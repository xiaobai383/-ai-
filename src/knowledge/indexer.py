"""批量索引器 — 遍历 data/logs/ 和 output/ 目录并将内容写入 ChromaDB。

ponytail: 每次运行执行简单 O(n) 扫描，通过 (path, mtime, size) 键跳过未变更文件。
升级路径：在生产规模监控目录中采用 inotify 驱动的增量更新。
"""
import hashlib
import json
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

# ── 缓存键辅助工具 ──


def _file_key(path: Path) -> Optional[str]:
    """生成基于内容的缓存键：(path, mtime, size)。"""
    try:
        stat = path.stat()
        raw = f"{path.resolve()}|{stat.st_mtime}|{stat.st_size}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
    except OSError:
        return None


def _load_index_state(cache_path: Path) -> Dict[str, str]:
    """加载上次已知的索引状态：{chroma_doc_id: cache_key}。"""
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
    """持久化索引状态。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}|{v}" for k, v in state.items()]
    cache_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_runlog_file(path: Path) -> str:
    """解析 JSONL 格式的 RunLog 文件，提取可搜索的纯文本。

    提取 user_query、各步骤的 output_preview 和结果文件内容。
    返回拼接后的纯文本，供 embedding 使用。
    """
    content = path.read_text(encoding="utf-8")
    lines = [l for l in content.strip().split("\n") if l]
    if not lines:
        return ""

    parts = []
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if obj.get("type") == "run":
            query = obj.get("user_query", "")
            if query:
                parts.append(f"用户问题: {query}")
        elif obj.get("type") == "step":
            name = obj.get("name", "")
            preview = obj.get("output_preview", "")
            if preview and name != "limit_check":
                parts.append(f"[{name}] {preview}")

    # 如果有结果文件，也读取其内容
    result_path = None
    try:
        header = json.loads(lines[0])
        result_path = header.get("result_path")
    except (json.JSONDecodeError, IndexError):
        pass

    if result_path:
        rp = Path(result_path)
        if rp.exists():
            try:
                result_text = rp.read_text(encoding="utf-8")
                if result_text.strip():
                    parts.append(f"结果:\n{result_text}")
            except Exception:
                pass

    return "\n\n".join(parts)


class Indexer:
    """扫描目标目录并将文本内容索引到 ChromaDB。

    用法:
        indexer = Indexer(store, embedder)
        result = indexer.index_logs("data/logs")
        # => {"added": 5, "skipped": 12, "errors": 0}
    """

    def __init__(self, store: KnowledgeStore, embedder: OllamaEmbedder):
        self.store = store
        self.embedder = embedder
        self._cache_dir = Path("data/chroma/.index_cache")

    def index_logs(self, logs_dir: str = "data/logs") -> dict:
        """索引所有 RunLog JSONL 文件。

        解析 JSONL 结构，提取用户查询和各步骤输出，生成可搜索的纯文本。
        """
        stats = {"added": 0, "skipped": 0, "errors": 0}
        logs_path = Path(logs_dir)
        if not logs_path.exists():
            return stats

        cache_path = self._cache_dir / f"{COLLECTION_RUNLOG}.state"
        old_state = _load_index_state(cache_path)
        new_state: Dict[str, str] = {}

        files = sorted(logs_path.glob("*.jsonl"))
        # 排除 notifications.jsonl
        files = [f for f in files if f.name.startswith("run-")]
        col = self.store.get_or_create(COLLECTION_RUNLOG)

        for file_path in files:
            cache_key = _file_key(file_path)
            if cache_key is None:
                continue

            doc_id = f"runlog:{file_path.name}"
            old_key = old_state.get(doc_id)
            if old_key == cache_key:
                stats["skipped"] += 1
                new_state[doc_id] = cache_key
                continue

            try:
                text = _parse_runlog_file(file_path)
            except Exception:
                stats["errors"] += 1
                continue

            if not text:
                new_state[doc_id] = cache_key or "empty"
                continue

            chunks = self._chunk_text(text, 8000)
            if not chunks:
                new_state[doc_id] = cache_key or "empty"
                continue

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
                    "doc_type": "runlog",
                    "chunk_index": i,
                })
                chunk_embs.append(emb)

            if chunk_ids:
                self.store.add(col, chunk_docs, chunk_metas, chunk_ids, chunk_embs)
                stats["added"] += len(chunk_ids)

            new_state[doc_id] = cache_key or "reindexed"

        # 清理已删除的文件
        removed_ids = set(old_state.keys()) - set(new_state.keys())
        for rid in removed_ids:
            try:
                col.delete(where={"source": rid})
            except Exception:
                pass

        _save_index_state(cache_path, new_state)
        return stats

    def index_outputs(self, outputs_dir: str = "output") -> dict:
        """索引所有输出的 .md/.txt 文件。"""
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
        """便捷方法：同时索引日志和输出。"""
        return {
            "logs": self.index_logs(),
            "outputs": self.index_outputs(),
        }

    # ── 内部方法 ──

    def _index_directory(
        self,
        directory: Path,
        pattern: str,
        collection_name: str,
        doc_type: str,
        chunk_size: int,
    ) -> dict:
        """遍历目录，读取匹配文件，分块、embedding 并更新插入。"""
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

            # 检查是否已索引且未变更
            doc_id = f"{doc_type}:{file_path.name}"
            old_key = old_state.get(doc_id)
            if old_key == cache_key:
                stats["skipped"] += 1
                new_state[doc_id] = cache_key
                continue

            # 读取并分块
            try:
                text = file_path.read_text(encoding="utf-8")
            except Exception:
                stats["errors"] += 1
                continue

            chunks = self._chunk_text(text, chunk_size)
            if not chunks:
                new_state[doc_id] = cache_key or "empty"
                continue

            # 为每个块生成 embedding
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

        # 清理 Chroma 中已删除的文件
        removed_ids = set(old_state.keys()) - set(new_state.keys())
        for rid in removed_ids:
            try:
                # 删除该文档的所有块
                col.delete(where={"source": rid})
            except Exception:
                pass

        _save_index_state(cache_path, new_state)
        return stats

    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> List[str]:
        """按段落边界将文本拆分为约 chunk_size 字符大小的块。"""
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
