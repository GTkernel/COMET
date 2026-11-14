"""Retrieve document evidence from versioned local semantic indexes."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time
import threading
import urllib.error
import urllib.request

MODEL = "embeddinggemma:300m"
CHUNKER_VERSION = "character-window-v1"
INDEX_VERSION = 1
COLLECTION = "mission_evidence_v1"
MAX_EXCERPT_BYTES = 1200


class SemanticUnavailable(ValueError):
    pass


def _request(route: str, payload=None, *, deadline=None):
    remaining = 30.0 if deadline is None else deadline - time.monotonic()
    if remaining <= 0:
        raise SemanticUnavailable("retrieval_deadline")
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/" + route,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=min(30.0, remaining)) as response:
            return json.load(response)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise SemanticUnavailable(f"embedding_service_unavailable: {exc}") from exc


def _digest(*, deadline=None):
    tags = _request("tags", deadline=deadline)
    match = next((item for item in tags.get("models", []) if item.get("name") == MODEL), None)
    if not match or not match.get("digest"):
        raise SemanticUnavailable(f"embedding_model_missing: {MODEL}")
    if (not match.get("size") or (match.get("details") or {}).get("format") != "gguf"
            or match.get("remote_host") or match.get("remote_model")):
        raise SemanticUnavailable("local_embedding_model_required")
    details = _request("show", {"model": MODEL}, deadline=deadline)
    if (details.get("remote_host") or details.get("remote_model")
            or (details.get("details") or {}).get("format") != "gguf"):
        raise SemanticUnavailable("local_embedding_model_required")
    return str(match["digest"])


def _embed(texts, *, deadline=None):
    # EmbeddingGemma's documented retrieval task prefixes differ for query/document.
    response = _request("embed", {"model": MODEL, "input": texts, "truncate": False,
                                  "keep_alive": "10m"}, deadline=deadline)
    vectors = response.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts) or not vectors:
        raise SemanticUnavailable("invalid_embedding_response")
    if any(not isinstance(v, list) for v in vectors):
        raise SemanticUnavailable("invalid_embedding_values")
    dimension = len(vectors[0])
    if not dimension or any(len(v) != dimension or not any(v) or any(isinstance(x, bool) or not isinstance(x, (int, float))
                           or not math.isfinite(x) for x in v) for v in vectors):
        raise SemanticUnavailable("invalid_embedding_values")
    return vectors


def _client(path):
    try:
        import chromadb
        return chromadb.PersistentClient(path=str(path))
    except Exception as exc:
        raise SemanticUnavailable(f"chroma_unavailable: {exc}") from exc


def _index_metadata(digest, dimension, chunk_size, chunk_overlap):
    return {"index_version": INDEX_VERSION, "embedding_model": MODEL, "embedding_digest": digest,
            "embedding_dimension": dimension, "chunker_version": CHUNKER_VERSION,
            "chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "hnsw:space": "cosine"}


def ingest_semantic(documents_dir, vector_store_dir, *, chunk_size=500, chunk_overlap=50):
    """Build a replacement collection, then publish its pointer after complete ingestion."""
    from src.rag.document_ingestion import (
        SUPPORTED_EXTENSIONS, _document_metadata, _read_document_text, _chunk_text, _vector_metadata)
    from src.db import utc_now
    import uuid

    started = time.monotonic()
    if chunk_size < 1 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Expected 0 <= chunk_overlap < chunk_size")
    source, destination = Path(documents_dir), Path(vector_store_dir)
    if not source.is_dir():
        raise ValueError(f"Mission document directory does not exist: {source}")
    digest = _digest()
    client = _client(destination)
    paths = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
    # A separate generation also preserves the previous semantic index on failed reindex.
    name = COLLECTION + "_" + uuid.uuid4().hex
    collection = None
    document_count = chunk_count = 0
    try:
        for path in paths:
            metadata = _document_metadata(path)
            chunks = _chunk_text(_read_document_text(path), metadata, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for begin in range(0, len(chunks), 32):
                current = chunks[begin:begin + 32]
                vectors = _embed(["title: " + metadata.file_name + " | text: " + c.text for c in current])
                if collection is None:
                    collection = client.create_collection(name=name, metadata=_index_metadata(digest, len(vectors[0]), chunk_size, chunk_overlap))
                if len(vectors[0]) != collection.metadata["embedding_dimension"]:
                    raise SemanticUnavailable("embedding_dimension_changed_during_ingestion")
                collection.upsert(ids=[c.chunk_id for c in current], documents=[c.text for c in current],
                                  embeddings=vectors, metadatas=[_vector_metadata(metadata, c, utc_now()) for c in current])
                chunk_count += len(current)
            document_count += 1
        if collection is None:
            raise SemanticUnavailable("no_document_evidence")
        if _digest() != digest:
            raise SemanticUnavailable("embedding_model_changed_during_ingestion")
        pointer = {"collection": name, "metadata": collection.metadata}
        destination.mkdir(parents=True, exist_ok=True)
        temporary = destination / ("semantic-index-" + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(pointer), encoding="utf-8")
        temporary.replace(destination / "semantic-index.json")
    except Exception:
        # Keep every previously published index, including the legacy collection.
        if collection is not None:
            client.delete_collection(name=name)
        raise
    return {"documents_found": len(paths), "documents_ingested": document_count, "documents_skipped": 0,
            "chunks_ingested": chunk_count, "vector_store": "chroma_semantic", "collection": name,
            "index_metadata": collection.metadata, "seconds": time.monotonic() - started}


def _identifier_match(text, identifiers):
    return any(re.search(r"(?<!\w)" + re.escape(identifier) + r"(?!\w)", text, re.IGNORECASE)
               for identifier in identifiers if identifier)


def _retrieve_semantic(query: str, vector_store_dir, *, identifiers=(), deadline=None, min_similarity=.35,
                      include_candidates=False):
    """Retrieve bounded semantic evidence and report missing or incompatible services."""
    from src.rag.document_ingestion import _retrieved_chunk
    started = time.monotonic()
    result = {"chunks": [], "status": "no_relevant_evidence", "seconds": 0.0, "index_metadata": None}
    try:
        if not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1:
            raise ValueError("min_similarity must be finite and between -1 and 1")
        pointer_path = Path(vector_store_dir) / "semantic-index.json"
        if not pointer_path.is_file():
            raise SemanticUnavailable("semantic_reindex_required")
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        metadata = pointer["metadata"]
        digest = _digest(deadline=deadline)
        if metadata.get("embedding_model") != MODEL or metadata.get("embedding_digest") != digest or metadata.get("index_version") != INDEX_VERSION or metadata.get("chunker_version") != CHUNKER_VERSION:
            raise SemanticUnavailable("semantic_index_incompatible_reindex_required")
        client = _client(vector_store_dir)
        collection = client.get_collection(pointer["collection"])
        if collection.metadata != metadata:
            raise SemanticUnavailable("semantic_index_metadata_mismatch")
        vector = _embed(["task: search result | query: " + query], deadline=deadline)[0]
        if len(vector) != metadata["embedding_dimension"]:
            raise SemanticUnavailable("semantic_index_dimension_mismatch")
        count = collection.count()
        if not count:
            raise SemanticUnavailable("no_document_evidence")
        found = collection.query(query_embeddings=[vector], n_results=min(8, count),
                                 include=["documents", "metadatas", "distances"])
        candidates = {}
        for chunk_id, text, info, distance in zip(found["ids"][0], found["documents"][0], found["metadatas"][0], found["distances"][0]):
            similarity = 1 - float(distance)
            if not math.isfinite(similarity):
                raise SemanticUnavailable("invalid_retrieval_score")
            exact = _identifier_match(text, identifiers)
            if similarity >= min_similarity or exact:
                entry = _retrieved_chunk(chunk_id=chunk_id, text=text, metadata=info, score=similarity)
                entry.update(semantic_similarity=similarity, exact_identifier_match=exact)
                candidates[chunk_id] = entry
        # Exact identifier lookup can recover evidence outside the semantic nearest eight.
        for identifier in list(dict.fromkeys(identifiers))[:8]:
            if not identifier:
                continue
            exact_rows = collection.get(where_document={"$contains": identifier}, limit=8,
                                        include=["documents", "metadatas"])
            for chunk_id, text, info in zip(exact_rows["ids"], exact_rows["documents"], exact_rows["metadatas"]):
                if chunk_id not in candidates and _identifier_match(text, [identifier]):
                    entry = _retrieved_chunk(chunk_id=chunk_id, text=text, metadata=info, score=0.0)
                    entry.update(semantic_similarity=None, exact_identifier_match=True)
                    candidates[chunk_id] = entry
        selected = sorted(candidates.values(), key=lambda c: (c["exact_identifier_match"], c["score"], c["chunk_id"]), reverse=True)[:8]
        if include_candidates:
            result["candidates"] = selected
        budget = MAX_EXCERPT_BYTES
        for entry in selected:
            if deadline is not None and time.monotonic() >= deadline:
                raise SemanticUnavailable("retrieval_deadline")
            # Bound excerpt size by UTF-8 bytes and keep each quote intact.
            size = len(entry["text"].encode("utf-8"))
            if size > budget:
                continue
            if not entry["document_version"]:
                raise SemanticUnavailable("unversioned_document_evidence")
            result["chunks"].append(entry)
            budget -= size
            if len(result["chunks"]) == 3:
                break
        result.update(status="ok" if result["chunks"] else "no_relevant_evidence", index_metadata=metadata)
    except (SemanticUnavailable, OSError, ValueError, KeyError, TypeError) as exc:
        result.update(chunks=[], status=str(exc))
        result.pop("candidates", None)
    except Exception as exc:
        result.update(chunks=[], status=f"chroma_retrieval_unavailable: {exc}")
        result.pop("candidates", None)
    result["seconds"] = time.monotonic() - started
    return result


_RETRIEVAL_SLOT = threading.Lock()


def retrieve_semantic(query: str, vector_store_dir, *, identifiers=(), deadline=None, min_similarity=.35,
                      include_candidates=False):
    """Bound caller latency even if local Chroma or a socket stalls."""
    started = time.monotonic()
    deadline = deadline if deadline is not None else started + 30
    failure = {"chunks": [], "status": "retrieval_deadline", "seconds": 0., "index_metadata": None}
    if not _RETRIEVAL_SLOT.acquire(timeout=max(0, deadline-time.monotonic())):
        return {**failure, "seconds": time.monotonic()-started}
    done, box = threading.Event(), {}
    def perform():
        try:
            box.update(_retrieve_semantic(query, vector_store_dir, identifiers=identifiers, deadline=deadline,
                                         min_similarity=min_similarity, include_candidates=include_candidates))
        finally:
            _RETRIEVAL_SLOT.release()
            done.set()
    threading.Thread(target=perform, daemon=True, name="comet-retrieval").start()
    if not done.wait(max(0, deadline-time.monotonic())):
        return {**failure, "seconds": time.monotonic()-started}
    return box
