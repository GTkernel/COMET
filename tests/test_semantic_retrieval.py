import json
from pathlib import Path
import time

import pytest

from src.rag import semantic
from src.rag.document_ingestion import ingest_documents


class Collection:
    def __init__(self, metadata):
        self.metadata = metadata
        self.rows = {}

    def upsert(self, ids, documents, embeddings, metadatas):
        self.rows.update({i: (d, m, v) for i, d, m, v in zip(ids, documents, metadatas, embeddings)})

    def count(self):
        return len(self.rows)

    def query(self, query_embeddings, n_results, include):
        rows = list(self.rows.items())[:n_results]
        return {"ids": [[r[0] for r in rows]], "documents": [[r[1][0] for r in rows]],
                "metadatas": [[r[1][1] for r in rows]], "distances": [[.2 if "pressure" in r[1][0] else .9 for r in rows]]}

    def get(self, where_document, limit, include):
        rows = [(i, row) for i, row in self.rows.items() if where_document["$contains"] in row[0]][:limit]
        return {"ids": [r[0] for r in rows], "documents": [r[1][0] for r in rows], "metadatas": [r[1][1] for r in rows]}


class Client:
    def __init__(self):
        self.collections = {}

    def create_collection(self, name, metadata):
        self.collections[name] = Collection(metadata)
        return self.collections[name]

    def get_collection(self, name):
        return self.collections[name]

    def delete_collection(self, name):
        del self.collections[name]


@pytest.fixture
def indexed(tmp_path, monkeypatch):
    client = Client()
    monkeypatch.setattr(semantic, "_client", lambda p: client)
    monkeypatch.setattr(semantic, "_digest", lambda **kw: "digest-one")
    monkeypatch.setattr(semantic, "_embed", lambda texts, **kw: [[1., 0., 0.] for _ in texts])
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "manual.md").write_text("channel_12 pressure observations. No root cause established.")
    (docs / "unrelated.txt").write_text("Catalog backups and storage retention.")
    index = tmp_path / "vectors"
    result = ingest_documents(docs, vector_store_dir=index, semantic=True)
    return docs, index, client, result


def test_semantic_ingestion_and_relevance_gate(indexed):
    docs, index, client, result = indexed
    assert result["vector_store"] == "chroma_semantic"
    assert result["chunks_ingested"] == 2
    answer = semantic.retrieve_semantic("pressure", index)
    assert answer["status"] == "ok"
    assert [c["source_file"] for c in answer["chunks"]] == ["manual.md"]
    assert answer["chunks"][0]["document_version"]
    assert answer["index_metadata"]["embedding_digest"] == "digest-one"


def test_reindex_preserves_previous_and_legacy_collections(indexed):
    docs, index, client, result = indexed
    client.collections["mission_documents"] = Collection({})
    again = ingest_documents(docs, vector_store_dir=index, semantic=True)
    assert again["collection"] != result["collection"]
    assert {result["collection"], "mission_documents", again["collection"]} <= set(client.collections)
    pointer = json.loads((index / "semantic-index.json").read_text())
    assert pointer["collection"] == again["collection"]


def test_failed_reindex_keeps_published_index(indexed, monkeypatch):
    docs, index, client, result = indexed
    pointer = (index / "semantic-index.json").read_bytes()
    def fail(*a, **kw):
        raise semantic.SemanticUnavailable("embedding unavailable")
    monkeypatch.setattr(semantic, "_embed", fail)
    with pytest.raises(ValueError, match="embedding unavailable"):
        ingest_documents(docs, vector_store_dir=index, semantic=True)
    assert (index / "semantic-index.json").read_bytes() == pointer
    assert result["collection"] in client.collections


def test_missing_index_model_and_changed_digest_degrade(indexed, monkeypatch, tmp_path):
    docs, index, client, result = indexed
    assert semantic.retrieve_semantic("q", tmp_path / "absent")["status"] == "semantic_reindex_required"
    monkeypatch.setattr(semantic, "_digest", lambda **kw: "different-model")
    answer = semantic.retrieve_semantic("pressure", index)
    assert answer["chunks"] == [] and "incompatible" in answer["status"]
    def missing(**kw):
        raise semantic.SemanticUnavailable("embedding_model_missing")
    monkeypatch.setattr(semantic, "_digest", missing)
    assert semantic.retrieve_semantic("pressure", index)["status"] == "embedding_model_missing"


def test_exact_identifier_boundaries_not_fuzzy(indexed):
    docs, index, client, result = indexed
    collection = client.get_collection(result["collection"])
    collection.rows["exact"] = ("channel_7 operational note", {"source_file": "ids", "document_version": "v1"}, [1, 0, 0])
    collection.rows["wrong"] = ("channel_70 operational note", {"source_file": "ids", "document_version": "v1"}, [1, 0, 0])
    answer = semantic.retrieve_semantic("q", index, identifiers=["channel_7"], min_similarity=.95)
    assert [c["chunk_id"] for c in answer["chunks"]] == ["exact"]
    assert answer["chunks"][0]["exact_identifier_match"]


def test_excerpt_budget_and_dimension_contract(indexed, monkeypatch):
    docs, index, client, result = indexed
    collection = client.get_collection(result["collection"])
    for i in range(5):
        collection.rows[str(i)] = ("pressure " + "é" * 200, {"source_file": "unicode", "document_version": "v1"}, [1, 0, 0])
    answer = semantic.retrieve_semantic("pressure", index)
    assert len(answer["chunks"]) <= 3
    assert sum(len(c["text"].encode()) for c in answer["chunks"]) <= 1200
    monkeypatch.setattr(semantic, "_embed", lambda *a, **kw: [[1., 0.]])
    assert "dimension_mismatch" in semantic.retrieve_semantic("q", index)["status"]


def test_network_deadline_and_local_endpoint(monkeypatch):
    calls = []
    def capture(request, timeout):
        calls.append(request.full_url)
        raise OSError("no local service")
    monkeypatch.setattr(semantic.urllib.request, "urlopen", capture)
    with pytest.raises(ValueError, match="retrieval_deadline"):
        semantic._request("tags", deadline=time.monotonic() - 1)
    assert calls == []
    with pytest.raises(ValueError, match="embedding_service_unavailable"):
        semantic._request("tags")
    assert calls == ["http://127.0.0.1:11434/api/tags"]


@pytest.mark.parametrize("embeddings", [[], [None], [[0., 0.]], [[float("nan")]], [[True]], [[0, 1], [0]]])
def test_embedding_payload_validation(monkeypatch, embeddings):
    monkeypatch.setattr(semantic, "_request", lambda *a, **kw: {"embeddings": embeddings})
    with pytest.raises(ValueError, match="invalid_embedding"):
        semantic._embed(["one"])


@pytest.mark.parametrize("remote", ["tag", "show"])
def test_remote_embedding_alias_rejected_before_embedding(monkeypatch, remote):
    calls = []
    def request(route, payload=None, **kwargs):
        calls.append(route)
        if route == "tags":
            model = {"name": semantic.MODEL, "digest": "sha", "size": 100,
                     "details": {"format": "gguf"}}
            if remote == "tag":
                model["remote_host"] = "https://remote.invalid"
            return {"models": [model]}
        if route == "show":
            return {"details": {"format": "gguf"}, "remote_host": "https://remote.invalid"}
        pytest.fail("No embedding request may send text to a remote alias")
    monkeypatch.setattr(semantic, "_request", request)
    with pytest.raises(semantic.SemanticUnavailable, match="local_embedding_model_required"):
        semantic._digest()
    assert "embed" not in calls
