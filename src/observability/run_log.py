"""用于可审计代理运行的结构化执行日志（RunLog）。"""
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class StepLog:
    """代理运行中的单个步骤。"""

    step_id: int
    name: str
    input_preview: str
    output_preview: str
    duration_ms: int
    tokens_in: int = 0
    tokens_out: int = 0
    cost_yuan: float = 0.0
    status: str = "success"

    def to_jsonl(self) -> str:
        """将此步骤序列化为 JSONL 行。"""
        return json.dumps(
            {
                "type": "step",
                "step_id": self.step_id,
                "name": self.name,
                "input_preview": self.input_preview,
                "output_preview": self.output_preview,
                "duration_ms": self.duration_ms,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "cost_yuan": self.cost_yuan,
                "status": self.status,
            },
            ensure_ascii=False,
        )


@dataclass
class RunLog:
    """代理执行运行的完整日志。"""

    run_id: str
    user_query: str
    mode: str
    model: str
    steps: List[StepLog] = field(default_factory=list)
    result_path: str | None = None
    fallback: bool = False  # v1.0：是否使用了本地回退

    @property
    def total_tokens_in(self) -> int:
        return sum(s.tokens_in for s in self.steps)

    @property
    def total_tokens_out(self) -> int:
        return sum(s.tokens_out for s in self.steps)

    @property
    def total_cost_yuan(self) -> float:
        return sum(s.cost_yuan for s in self.steps)

    def to_jsonl(self) -> str:
        """将整个运行序列化为 JSONL（头部 + 每个步骤一行）。"""
        lines = [
            json.dumps(
                {
                    "type": "run",
                    "run_id": self.run_id,
                    "user_query": self.user_query,
                    "mode": self.mode,
                    "model": self.model,
                    "total_tokens_in": self.total_tokens_in,
                    "total_tokens_out": self.total_tokens_out,
                    "total_cost_yuan": self.total_cost_yuan,
                    "result_path": self.result_path,
                    "fallback": self.fallback,
                },
                ensure_ascii=False,
            )
        ]
        for step in self.steps:
            lines.append(step.to_jsonl())
        return "\n".join(lines) + "\n"

    @classmethod
    def from_jsonl(cls, path: Path) -> "RunLog":
        """从 JSONL 文件反序列化 RunLog。"""
        content = path.read_text(encoding="utf-8")
        lines = [l for l in content.strip().split("\n") if l]

        header = json.loads(lines[0])
        run = cls(
            run_id=header["run_id"],
            user_query=header["user_query"],
            mode=header["mode"],
            model=header["model"],
            result_path=header.get("result_path"),
            fallback=header.get("fallback", False),
        )

        for line in lines[1:]:
            obj = json.loads(line)
            if obj.get("type") == "step":
                run.steps.append(
                    StepLog(
                        step_id=obj["step_id"],
                        name=obj["name"],
                        input_preview=obj["input_preview"],
                        output_preview=obj["output_preview"],
                        duration_ms=obj["duration_ms"],
                        tokens_in=obj.get("tokens_in", 0),
                        tokens_out=obj.get("tokens_out", 0),
                        cost_yuan=obj.get("cost_yuan", 0.0),
                        status=obj.get("status", "success"),
                    )
                )

        return run

    def save_to_disk(self, path) -> Path:
        """将此 RunLog 持久化为 JSONL 文件。

        Args:
            path: 目标文件路径（目录或完整路径）。

        Returns:
            已写入文件的路径。
        """
        p = Path(path)
        if p.is_dir() or str(p).endswith("/") or str(p).endswith("\\"):
            p.mkdir(parents=True, exist_ok=True)
            p = p / f"{self.run_id}.jsonl"
        else:
            p.parent.mkdir(parents=True, exist_ok=True)

        p.write_text(self.to_jsonl(), encoding="utf-8")
        return p
