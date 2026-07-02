"""Batch indexer — walk data/logs/ and output/ and ingest into ChromaDB.

ponytail: simple O(n) scan on each indexer run, keyed by (path, mtime, size) to
skip unchanged files. Upgrade path: inotify-driven incremental updates for
production-scale watch directories.
"""
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.knowledge.embedder import OllamaEmbedder
from src.knowledge.store import (
    COLLECTION_OUTPUTS,
    COLLECTION_RUNLOG,
    KnowledgeStore,
)

logger = logging.getLogger(__name__)

# ── Cache key helpers ──


def _file_key(path: Path) -> Optional[str]:
    """Generate a content-based cache key: (path, mtime, size)."""
    try:
        stat = path.stat()
        raw = f"{path.resolve()}|{stat.st_mtime}|{stat.st_size}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
    except OSError:
        return None


def _load_index_state(cache_path: Path) -> Dict[str, str]:
    """Load the last-known index state: {chroma_doc_id: cache_key}."""
    if not cache_path.exists():
        return {}
    try:
        lines = cache_path.read_text(encoding="utf-8").strip().split("\n")
        state = {}
        for line in lines:
            if "|" in line:
                doc_id, key = line.split("|", 1)
                state[doc_id] = key
        return state
    except Exception:
        return {}


def _save_index_state(cache_path: Path, state: Dict[str, str]) -> None:
    """Persist the index state."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}|{v}" for k, v in state.items()]
    cache_path.write_text("\n".join(lines), encoding="utf-8")


class Indexer:
    """Scans target directories and indexes text content into ChromaDB.

    Usage:
        indexer = Indexer(store, embedder)
        result = indexer.index_logs("data/logs")
        # => {"added": 5, "skipped": 12, "errors": 0}
    """

    def __init__(self, store: KnowledgeStore, embedder: OllamaEmbedder):
        self.store = store
        self.embedder = embedder
        self._cache_dir = Path("data/chroma/.index_cache")

    def index_logs(self, logs_dir: str = "data/logs") -> dict:
        """Index all RunLog JSONL files.

        Each file produces one document with its full text content.
        """
        return self._index_directory(
            directory=Path(logs_dir),
            pattern="*.jsonl",
            collection_name=COLLECTION_RUNLOG,
            doc_type="runlog",
            chunk_size=8000,  # chars per doc
        )

    def index_outputs(self, outputs_dir: str = "output") -> dict:
        """Index all output .md/.txt files."""
        stats = self._index_directory(
            directory=Path(outputs_dir),
            pattern="*.md",
            collection_name=COLLECTION_OUTPUTS,
            doc_type="output",
            chunk_size=8000,
        )
        stats2 = self._index_directory(
            directory=Path(outputs_dir),
            pattern="*.txt",
            collection_name=COLLECTION_OUTPUTS,
            doc_type="output",
            chunk_size=8000,
        )
        for k in ("added", "skipped", "errors"):
            stats[k] = stats.get(k, 0) + stats2.get(k, 0)
        return stats

    def index_all(self) -> dict:
        """Convenience: index both logs and outputs."""
        return {
            "logs": self.index_logs(),
            "outputs": self.index_outputs(),
        }

    # ── internals ──

    def _index_directory(
        self,
        directory: Path,
        pattern: str,
        collection_name: str,
        doc_type: str,
        chunk_size: int,
    ) -> dict:
        """Walk a directory, read matching files, chunk, embed, and upsert."""
        stats = {"added": 0, "skipped": 0, "errors": 0}

        if not directory.exists():
            return stats

        cache_path = self._cache_dir / f"{collection_name}.state"
        old_state = _load_index_state(cache_path)
        new_state: Dict[str, str] = {}

        files = sorted(directory.glob(pattern))
        col = self.store.get_or_create(collection_name)

        for file_path in files:
            cache_key = _file_key(file_path)
            if cache_key is None:
                continue

            # Check if already indexed and unchanged
            doc_id = f"{doc_type}:{file_path.name}"
            old_key = old_state.get(doc_id)
            if old_key == cache_key:
                stats["skipped"] += 1
                new_state[doc_id] = cache_key
                continue

            # Read and chunk
            try:
                text = file_path.read_text(encoding="utf-8")
            except Exception:
                stats["errors"] += 1
                continue

            chunks = self._chunk_text(text, chunk_size)
            if not chunks:
                new_state[doc_id] = cache_key or "empty"
                continue

            # Embed each chunk
            chunk_ids: List[str] = []
            chunk_docs: List[str] = []
            chunk_metas: List[dict] = []
            chunk_embs: List[List[float]] = []

            for i, chunk in enumerate(chunks):
                cid = f"{doc_id}:chunk{i}"
                emb = self.embedder.embed(chunk)
                if emb is None:
                    stats["errors"] += 1
                    continue

                chunk_ids.append(cid)
                chunk_docs.append(chunk)
                chunk_metas.append({
                    "source": str(file_path),
                    "doc_type": doc_type,
                    "chunk_index": i,
                })
                chunk_embs.append(emb)

            if chunk_ids:
                self.store.add(col, chunk_docs, chunk_metas, chunk_ids, chunk_embs)
                stats["added"] += len(chunk_ids)

            new_state[doc_id] = cache_key or "reindexed"

        # Prune deleted files from Chroma
        removed_ids = set(old_state.keys()) - set(new_state.keys())
        for rid in removed_ids:
            try:
                # Delete all chunks for this doc
                col.delete(where={"source": rid})
            except Exception:
                pass

        _save_index_state(cache_path, new_state)
        return stats

    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> List[str]:
        """Split text into ~chunk_size-character chunks on paragraph boundaries."""
        if len(text) <= chunk_size:
            return [text]

        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if current_len + len(para) > chunk_size and current:
                chunks.append("\n\n".join(current))
                current = [para]
                current_len = len(para)
            else:
                current.append(para)
                current_len += len(para)

        if current:
            chunks.append("\n\n".join(current))

        return chunks
