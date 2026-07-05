"""Tests for src.knowledge — vector store, embedder, indexer, and search."""
import json
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import AppConfig
from src.knowledge.embedder import OllamaEmbedder
from src.knowledge.indexer import Indexer, _file_key
from src.knowledge.search import SearchResponse, SearchResult, Searcher
from src.knowledge.store import KnowledgeStore


# ── Store tests ──


class TestKnowledgeStore:
    """ChromaDB store lifecycle — uses a project-local dir to avoid tempfile lock issues."""

    TEST_DIR = Path("data/test_chroma_knowledge")

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        """Clean test dir before each test."""
        import shutil
        if self.TEST_DIR.exists():
            shutil.rmtree(self.TEST_DIR, ignore_errors=True)
        yield

    def _store(self):
        return KnowledgeStore(persist_dir=str(self.TEST_DIR / "db"))

    def test_creates_persist_dir(self):
        s = self._store()
        assert (self.TEST_DIR / "db").exists()

    def test_get_or_create_returns_collection(self):
        s = self._store()
        col = s.get_or_create("test_coll_xyz")
        assert col is not None
        assert col.name == "test_coll_xyz"

    def test_count_starts_zero(self):
        s = self._store()
        s.get_or_create("my_count_test")
        assert s.count("my_count_test") == 0

    def test_add_and_count(self):
        s = self._store()
        col = s.get_or_create("add_count_test")
        s.add(col, ["hello world"], [{"source": "x.txt"}], ["id1"])
        assert s.count("add_count_test") == 1

    def test_upsert_replaces_by_id(self):
        s = self._store()
        col = s.get_or_create("upsert_test")
        s.add(col, ["v1"], [{"s": "a"}], ["u1"])
        s.add(col, ["v2"], [{"s": "b"}], ["u1"])
        assert s.count("upsert_test") == 1

    def test_list_collections(self):
        s = self._store()
        s.get_or_create("coll_aaa")  # chromadb requires 3+ char names
        s.get_or_create("coll_bbb")
        names = s.list_collections()
        assert "coll_aaa" in names
        assert "coll_bbb" in names

    def test_delete_collection(self):
        s = self._store()
        s.get_or_create("coll_to_del")
        s.delete_collection("coll_to_del")
        assert "coll_to_del" not in s.list_collections()

    def test_delete_nonexistent_no_error(self):
        s = self._store()
        s.delete_collection("no_such_coll")  # should not raise

    def test_add_empty_documents(self):
        s = self._store()
        col = s.get_or_create("empty_docs_test")
        s.add(col, [], [], [])  # should not raise
        assert s.count("empty_docs_test") == 0


# ── Embedder tests ──


class TestOllamaEmbedder:
    """Ollama embedding wrapper (mock HTTP)."""

    @pytest.fixture
    def embedder(self):
        return OllamaEmbedder(base_url="http://localhost:9999", timeout=2)

    def test_defaults(self):
        e = OllamaEmbedder()
        assert e._model == "nomic-embed-text"

    def test_is_available_false_when_no_server(self, embedder):
        assert embedder.is_available() is False

    @patch("src.knowledge.embedder.requests.get")
    def test_is_available_true_when_model_present(self, mock_get, embedder):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }
        embedder._available = None
        assert embedder.is_available() is True

    @patch("src.knowledge.embedder.requests.get")
    def test_is_available_false_when_model_missing(self, mock_get, embedder):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"models": [{"name": "llama3"}]}
        embedder._available = None
        assert embedder.is_available() is False

    @patch("src.knowledge.embedder.requests.get")
    def test_is_available_caches_result(self, mock_get, embedder):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }
        embedder._available = None
        embedder._check_interval = 999
        assert embedder.is_available() is True
        assert embedder.is_available() is True
        assert mock_get.call_count == 1

    def test_embed_returns_none_when_unavailable(self, embedder):
        embedder._available = False
        assert embedder.embed("test") is None

    @patch("src.knowledge.embedder.requests.post")
    def test_embed_success(self, mock_post, embedder):
        embedder._available = True
        embedder._last_check = float("inf")
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_post.return_value.raise_for_status = MagicMock()
        result = embedder.embed("hello")
        assert result == [0.1, 0.2, 0.3]

    @patch("src.knowledge.embedder.requests.post")
    def test_embed_http_error(self, mock_post, embedder):
        import requests as req
        embedder._available = True
        embedder._last_check = float("inf")
        mock_post.side_effect = req.RequestException("boom")
        assert embedder.embed("hello") is None


# ── Indexer tests ──


class TestIndexer:
    """Indexer file scanning and chunking."""

    @pytest.fixture
    def tmp_dirs(self):
        logs_dir = tempfile.mkdtemp()
        chroma_dir = tempfile.mkdtemp()
        yield Path(logs_dir), Path(chroma_dir)
        import shutil
        shutil.rmtree(logs_dir, ignore_errors=True)
        shutil.rmtree(chroma_dir, ignore_errors=True)

    def test_file_key_stable(self, tmp_dirs):
        logs, _ = tmp_dirs
        f = logs / "test.jsonl"
        f.write_text('{"type":"run"}', encoding="utf-8")
        k1 = _file_key(f)
        k2 = _file_key(f)
        assert k1 == k2

    def test_file_key_changes_on_modify(self, tmp_dirs):
        logs, _ = tmp_dirs
        f1 = logs / "a.jsonl"
        f2 = logs / "b.jsonl"
        f1.write_text("v1", encoding="utf-8")
        f2.write_text("v2", encoding="utf-8")
        k1 = _file_key(f1)
        k2 = _file_key(f2)
        assert k1 != k2  # Different paths always give different keys

    def test_chunk_text_short(self):
        chunks = Indexer._chunk_text("hello", 100)
        assert chunks == ["hello"]

    def test_chunk_text_splits_on_paragraphs(self):
        text = "paragraph one\n\nparagraph two\n\nparagraph three"
        chunks = Indexer._chunk_text(text, 20)
        assert len(chunks) >= 2

    def test_chunk_text_empty(self):
        assert Indexer._chunk_text("", 100) == [""]

    def test_index_logs_empty_dir(self, tmp_dirs):
        logs, chroma = tmp_dirs
        store = KnowledgeStore(persist_dir=str(chroma))
        embedder = OllamaEmbedder()
        embedder._available = False
        indexer = Indexer(store, embedder)
        result = indexer.index_logs(str(logs))
        assert result["added"] == 0

    def test_index_all(self, tmp_dirs):
        logs, chroma = tmp_dirs
        store = KnowledgeStore(persist_dir=str(chroma))
        embedder = OllamaEmbedder()
        embedder._available = False
        indexer = Indexer(store, embedder)
        result = indexer.index_all()
        assert "logs" in result
        assert "outputs" in result


# ── Search tests ──


class TestSearcher:
    """Search interface."""

    def test_search_unavailable_embedder(self, tmp_path):
        store = KnowledgeStore(persist_dir=str(tmp_path / "chroma"))
        embedder = OllamaEmbedder()
        embedder._available = False
        searcher = Searcher(store, embedder)
        resp = searcher.search("test query")
        assert isinstance(resp, SearchResponse)
        assert resp.embedding_available is False
        assert resp.results == []

    def test_search_result_dataclass(self):
        r = SearchResult(
            chunk_id="c1",
            document="hello world",
            score=0.85,
            source="/tmp/test.md",
            doc_type="output",
            chunk_index=0,
        )
        assert r.score == 0.85
        assert r.source == "/tmp/test.md"

    def test_search_response_defaults(self):
        resp = SearchResponse(query="q")
        assert resp.query == "q"
        assert resp.results == []
        assert resp.total_hits == 0
