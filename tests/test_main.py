import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture(autouse=True)
def fake_backends(monkeypatch):
    monkeypatch.setattr(main, "qdrant_client", MagicMock())
    monkeypatch.setattr(main, "RAG_API_KEY", None)
    openai = MagicMock()
    openai.embeddings.create.side_effect = lambda model, input: MagicMock(
        data=[MagicMock(embedding=[0.0] * 1536) for _ in (input if isinstance(input, list) else [input])]
    )
    monkeypatch.setattr(main, "get_openai_client", lambda: openai)
    return openai


client = TestClient(main.app)


def test_chunk_text_overlap_and_skips_tiny():
    chunks = main.chunk_text("a" * 1900)
    assert [len(c) for c in chunks] == [1000, 1000, 300]
    assert main.chunk_text("a" * 40) == []


def test_api_key_required_when_configured(monkeypatch):
    monkeypatch.setattr(main, "RAG_API_KEY", "secret")
    body = {"query": "q", "company_id": "c", "accessible_doc_ids": []}
    assert client.post("/search", json=body).status_code == 401
    assert client.post("/search", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/search", json=body, headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/health").status_code == 200


def test_search_with_no_accessible_docs_returns_nothing(fake_backends):
    r = client.post("/search", json={"query": "q", "company_id": "c", "accessible_doc_ids": []})
    assert r.json() == {"success": True, "results": []}
    main.qdrant_client.search.assert_not_called()


def test_search_filters_by_company_and_docs():
    main.qdrant_client.search.return_value = []
    client.post("/search", json={"query": "q", "company_id": "c1", "accessible_doc_ids": ["d1", "d2"]})
    flt = main.qdrant_client.search.call_args.kwargs["query_filter"]
    keys = {c.key: c.match for c in flt.must}
    assert keys["company_id"].value == "c1"
    assert keys["document_id"].any == ["d1", "d2"]


def test_search_limit_is_bounded():
    r = client.post("/search", json={"query": "q", "company_id": "c", "accessible_doc_ids": ["d"], "limit": 100000})
    assert r.status_code == 422


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "http://127.0.0.1/x.pdf",
    "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5/x.pdf",
    "http://[::1]/x.pdf",
])
def test_index_rejects_non_public_urls(url):
    r = client.post("/index", data={"file_url": url, "document_id": "d", "company_id": "c"})
    assert r.status_code == 400
    main.qdrant_client.upsert.assert_not_called()


def test_index_replaces_previous_version():
    text = ("hello world " * 100).encode()
    r = client.post(
        "/index",
        files={"file": ("a.txt", text)},
        data={"document_id": "d1", "company_id": "c1"},
    )
    assert r.status_code == 200, r.text
    main.qdrant_client.delete.assert_called_once()
    main.qdrant_client.upsert.assert_called_once()


def test_index_unsupported_type_is_client_error():
    r = client.post("/index", files={"file": ("a.exe", b"x" * 100)}, data={"document_id": "d", "company_id": "c"})
    assert r.status_code == 400


def test_delete_is_scoped_to_company():
    r = client.post("/delete", json={"document_id": "d1", "company_id": "c1"})
    assert r.status_code == 200
    flt = main.qdrant_client.delete.call_args.kwargs["points_selector"]
    assert {c.key: c.match.value for c in flt.must} == {"document_id": "d1", "company_id": "c1"}
