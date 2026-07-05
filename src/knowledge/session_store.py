"""会话与消息存储 — JSON 文件持久化。

会话列表存 data/chat_sessions.json，每个会话的消息存 data/chat_messages/{session_id}.json。

为什么用 JSON 而不是 ChromaDB：会话消息的读取模式是「按 session_id 全量取 + 时间排序」，
这是关系型场景，JSON 最简最直接。ChromaDB 留给文件内容做语义检索（RAG），
体现「什么场景选什么存储」的选型判断。

ponytail: 全量读写 JSON。个人用量级（百级会话、千级消息）完全够用。
升级路径：超过万级消息可换 SQLite。
"""
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


class SessionStore:
    """会话 + 消息的 JSON 持久化层。"""

    def __init__(self, base_dir: str = "data"):
        self._base = Path(base_dir)
        self._sessions_file = self._base / "chat_sessions.json"
        self._messages_dir = self._base / "chat_messages"
        self._base.mkdir(parents=True, exist_ok=True)
        self._messages_dir.mkdir(parents=True, exist_ok=True)

    # ── 会话列表 ──

    def _load_sessions(self) -> List[Dict[str, Any]]:
        """读取会话列表 JSON。"""
        if not self._sessions_file.exists():
            return []
        try:
            return json.loads(self._sessions_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_sessions(self, sessions: List[Dict[str, Any]]) -> None:
        """写入会话列表 JSON。"""
        self._sessions_file.write_text(
            json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def create_session(
        self, mode: str = "privacy_enhanced", title: str = "新建会话"
    ) -> Dict[str, Any]:
        """创建新会话，返回会话元数据。"""
        now_ms = int(time.time() * 1000)
        session = {
            "id": f"sess-{uuid.uuid4().hex[:12]}",
            "title": title,
            "mode": mode,
            "created_at": now_ms,
            "updated_at": now_ms,
            "message_count": 0,
        }
        sessions = self._load_sessions()
        sessions.append(session)
        self._save_sessions(sessions)
        # 创建空消息文件
        self._save_messages(session["id"], [])
        return session

    def list_sessions(self) -> List[Dict[str, Any]]:
        """返回全部会话，按 updated_at 倒序（最近活跃在前）。"""
        sessions = self._load_sessions()
        sessions.sort(key=lambda s: s.get("updated_at", 0), reverse=True)
        return sessions

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """取单个会话元数据。"""
        for s in self._load_sessions():
            if s["id"] == session_id:
                return s
        return None

    def delete_session(self, session_id: str) -> None:
        """删除会话（列表 + 消息文件）。"""
        sessions = self._load_sessions()
        sessions = [s for s in sessions if s["id"] != session_id]
        self._save_sessions(sessions)
        msg_file = self._messages_dir / f"{session_id}.json"
        if msg_file.exists():
            msg_file.unlink()

    def rename_session(self, session_id: str, title: str) -> None:
        """重命名会话。"""
        sessions = self._load_sessions()
        for s in sessions:
            if s["id"] == session_id:
                s["title"] = title
                s["updated_at"] = int(time.time() * 1000)
                break
        self._save_sessions(sessions)

    # ── 消息 ──

    def _msg_file(self, session_id: str) -> Path:
        return self._messages_dir / f"{session_id}.json"

    def _load_messages(self, session_id: str) -> List[Dict[str, Any]]:
        f = self._msg_file(session_id)
        if not f.exists():
            return []
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        self._msg_file(session_id).write_text(
            json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add_message(self, session_id: str, role: str, content: str,
                    redact_map: dict | None = None) -> Dict[str, Any]:
        """追加一条消息，返回该消息。同时更新会话 message_count + updated_at。"""
        msg = {
            "id": f"msg-{uuid.uuid4().hex[:8]}",
            "role": role,
            "content": content,
            "timestamp": int(time.time() * 1000),
            "redact_map": redact_map,
        }
        messages = self._load_messages(session_id)
        messages.append(msg)
        self._save_messages(session_id, messages)

        # 更新会话元数据
        sessions = self._load_sessions()
        for s in sessions:
            if s["id"] == session_id:
                s["message_count"] = len(messages)
                s["updated_at"] = msg["timestamp"]
                # 首条用户消息自动设为会话标题（截断）
                if s["title"] == "新建会话" and role == "user":
                    s["title"] = content[:30] + ("..." if len(content) > 30 else "")
                break
        self._save_sessions(sessions)
        return msg

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """取某会话全部消息，按时间正序（早的在前）。"""
        messages = self._load_messages(session_id)
        messages.sort(key=lambda m: m.get("timestamp", 0))
        return messages
