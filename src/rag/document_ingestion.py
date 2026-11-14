from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.paths import RAG_DOCS_DIR, RAG_VECTOR_DIR
from src.db import utc_now

# Also should probably put this in .env
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
EMBEDDING_DIMENSION = 64


@dataclass(frozen=True)
class DocumentMetadata:
    source_path: str
    file_name: str
    file_size: int
    modified_time: str
    content_hash: str
    document_version: str


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    page_number: int | None = None


def ingest_documents(
    documents_dir: str | Path = RAG_DOCS_DIR,
    *,
    vector_store_dir: str | Path = RAG_VECTOR_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    db_path: str | Path | None = None,
    semantic: bool = False,
) -> dict[str, Any]:
    if semantic:
        from src.rag.semantic import ingest_semantic
        return ingest_semantic(documents_dir, vector_store_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    docs_path = Path(documents_dir)
    vector_path = Path(vector_store_dir)
    if not docs_path.exists() or not docs_path.is_dir():
        raise ValueError(
            f"Mission document directory does not exist: {docs_path}. "
            "Create it and add .txt, .md, or .pdf files before running ingest-docs."
        )
    vector_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "documents_found": 0,
        "documents_ingested": 0,
        "documents_skipped": 0,
        "chunks_ingested": 0,
        "vector_store": None,
    }

    vector_store = _VectorStore(vector_path)
    summary["vector_store"] = vector_store.kind

    document_paths = sorted(
        path
        for path in docs_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    summary["documents_found"] = len(document_paths)

    for document_path in document_paths:
        metadata = _document_metadata(document_path)
        text = _read_document_text(document_path)
        chunks = _chunk_text(text, metadata, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ingested_at = utc_now()
        vector_store.replace_document(metadata, chunks, ingested_at)

        summary["documents_ingested"] += 1
        summary["chunks_ingested"] += len(chunks)

    return summary


def _document_metadata(document_path: Path) -> DocumentMetadata:
    stat = document_path.stat()
    content_hash = _sha256(document_path)
    return DocumentMetadata(
        source_path=str(document_path.resolve()),
        file_name=document_path.name,
        file_size=stat.st_size,
        modified_time=str(stat.st_mtime_ns),
        content_hash=content_hash,
        document_version=content_hash[:12],
    )


def _read_document_text(document_path: Path) -> str:
    suffix = document_path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return document_path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        return _extract_pdf_text(document_path)
    raise ValueError(f"Unsupported document type: {document_path.suffix}")


def _extract_pdf_text(document_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise ValueError(
            "PDF document ingestion requires pypdf. "
            "Install it with: python -m pip install pypdf"
        ) from exc

    try:
        reader = PdfReader(str(document_path))
    except Exception as exc:
        raise ValueError(f"Could not read PDF document {document_path}: {exc}") from exc

    page_texts: list[str] = []
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            page_texts.append(f"\n\n[Page {page_index}]\n{text.strip()}")
    return "".join(page_texts).strip()


def _chunk_text(
    text: str,
    metadata: DocumentMetadata,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    if not text:
        return []

    chunks: list[TextChunk] = []
    start = 0
    chunk_index = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        if chunk_text.strip():
            chunk_id = _chunk_id(metadata, chunk_index, start, end, chunk_text)
            page_number = _infer_chunk_page_number(text, start, end)
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    page_number=page_number,
                )
            )
            chunk_index += 1
        if end == len(text):
            break
        start += step
    return chunks


def _infer_chunk_page_number(text: str, start_char: int, end_char: int) -> int | None:
    prefix = text[:end_char]
    markers = []
    search_start = 0
    marker = "[Page "
    while True:
        index = prefix.find(marker, search_start)
        if index == -1:
            break
        close = prefix.find("]", index)
        if close == -1:
            break
        page_value = prefix[index + len(marker) : close]
        if page_value.isdigit() and index <= end_char:
            markers.append((index, int(page_value)))
        search_start = close + 1

    before_or_inside = [page for index, page in markers if index <= start_char]
    if before_or_inside:
        return before_or_inside[-1]
    if markers:
        return markers[0][1]
    return None


def _chunk_id(
    metadata: DocumentMetadata,
    chunk_index: int,
    start_char: int,
    end_char: int,
    chunk_text: str,
) -> str:
    digest = hashlib.sha256(
        f"{metadata.source_path}|{metadata.document_version}|{chunk_index}|{start_char}|{end_char}|{chunk_text}".encode(
            "utf-8"
        )
    ).hexdigest()
    return f"chunk_{digest[:24]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _VectorStore:
    def __init__(self, persist_dir: Path) -> None:
        self.persist_dir = persist_dir
        try:
            import chromadb

            self.client = chromadb.PersistentClient(path=str(persist_dir))
            self.collection = self.client.get_or_create_collection("mission_documents")
            self.kind = "chroma"
        except Exception:
            self.client = None
            self.collection = None
            self.kind = "jsonl_fallback"

    def replace_document(
        self,
        metadata: DocumentMetadata,
        chunks: list[TextChunk],
        ingested_at: str,
    ) -> None:
        if self.collection is not None:
            try:
                self.collection.delete(where={"source_path": metadata.source_path})
            except Exception:
                pass
            if chunks:
                self.collection.upsert(
                    ids=[chunk.chunk_id for chunk in chunks],
                    documents=[chunk.text for chunk in chunks],
                    embeddings=[_local_embedding(chunk.text) for chunk in chunks],
                    metadatas=[
                        _vector_metadata(metadata, chunk, ingested_at)
                        for chunk in chunks
                    ],
                )
            return

        fallback_path = self.persist_dir / "mission_documents.jsonl"
        existing = []
        if fallback_path.exists():
            existing = [
                json.loads(line)
                for line in fallback_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        existing = [record for record in existing if record.get("source_path") != metadata.source_path]
        existing.extend(
            {
                "id": chunk.chunk_id,
                "text": chunk.text,
                "embedding": _local_embedding(chunk.text),
                "source_file": metadata.file_name,
                "source_path": metadata.source_path,
                "document_version": metadata.document_version,
                "chunk_index": chunk.chunk_index,
                "ingested_at": ingested_at,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "page_number": chunk.page_number,
            }
            for chunk in chunks
        )
        fallback_path.write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in existing),
            encoding="utf-8",
        )

    def query(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        query_embedding = _local_embedding(query_text)
        if self.collection is not None:
            count = self.collection.count()
            if count == 0:
                return []
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )
            ids = results.get("ids", [[]])[0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            return [
                _retrieved_chunk(
                    chunk_id=str(chunk_id),
                    text=str(text),
                    metadata=dict(metadata or {}),
                    score=1.0 / (1.0 + float(distance)),
                )
                for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances)
            ]

        fallback_path = self.persist_dir / "mission_documents.jsonl"
        if not fallback_path.exists():
            return []
        scored = []
        for line in fallback_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            text = str(record.get("text", ""))
            score = _cosine_similarity(query_embedding, record.get("embedding", _local_embedding(text)))
            scored.append(
                _retrieved_chunk(
                    chunk_id=str(record["id"]),
                    text=text,
                    metadata=record,
                    score=score,
                )
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]


def _retrieved_chunk(
    *,
    chunk_id: str,
    text: str,
    metadata: dict[str, Any],
    score: float,
) -> dict[str, Any]:
    start_char = metadata.get("start_char")
    end_char = metadata.get("end_char")
    return {
        "chunk_id": chunk_id,
        "chunk_index": int(metadata.get("chunk_index", 0)),
        "text": text,
        "score": score,
        "source_file": str(metadata.get("source_file", "")),
        "source_path": str(metadata.get("source_path", "")),
        "document_version": str(metadata.get("document_version", "")),
        "text_span": f"{start_char}:{end_char}" if start_char is not None and end_char is not None else None,
        "metadata": metadata,
    }


def _vector_metadata(
    metadata: DocumentMetadata,
    chunk: TextChunk,
    ingested_at: str,
) -> dict[str, str | int]:
    payload: dict[str, str | int] = {
        "source_file": metadata.file_name,
        "source_path": metadata.source_path,
        "document_version": metadata.document_version,
        "chunk_index": chunk.chunk_index,
        "ingested_at": ingested_at,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
    }
    if chunk.page_number is not None:
        payload["page_number"] = chunk.page_number
    return payload


def _local_embedding(text: str, dimension: int = EMBEDDING_DIMENSION) -> list[float]:
    vector = [0.0] * dimension
    tokens = text.lower().split()
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
