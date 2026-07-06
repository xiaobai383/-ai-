"""FastAPI 应用 — Hush 的 REST API。"""
import asyncio
import json
import logging
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path as PathLib
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agent.app import run_task
from src.config import AppConfig

logger = logging.getLogger(__name__)


def _build_chat_llm(config: AppConfig):
    """构建对话用 LLM（云端优先 + Ollama 兜底），未配置 api_key 返回 None。"""
    if not config.api_key or "sk-fake" in config.api_key:
        return None
    if config.fallback_enabled:
        from src.fallback.provider import FallbackChatModel
        return FallbackChatModel(
            primary_model=config.model_name,
            primary_base_url=config.model_base_url,
            api_key=config.api_key,
            fallback_base_url=config.fallback_ollama_base_url,
            fallback_model=config.fallback_ollama_model,
            timeout=config.fallback_timeout_seconds,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=config.model_name,
        api_key=config.api_key,
        base_url=config.model_base_url,
        temperature=0.3,
    )


def _build_local_llm(config: AppConfig):
    """构建纯本地 LLM（Ollama），不可用返回 None。"""
    from src.fallback.provider import _is_ollama_available
    if not _is_ollama_available(config.fallback_ollama_base_url):
        return None
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=config.fallback_ollama_model,
        api_key="ollama",
        base_url=config.fallback_ollama_base_url,
        temperature=0.3,
        timeout=60,
    )


def _log_chat_cost(session_id, query, response, usage, config: AppConfig):
    """记录对话费用到 data/logs/chat_costs.jsonl。"""
    from src.tools.cost import estimate_cost
    tokens_in = usage.get("tokens_in", 0)
    tokens_out = usage.get("tokens_out", 0)
    cost = estimate_cost(tokens_in, tokens_out, config.model_name) if (tokens_in or tokens_out) else 0.0
    entry = {
        "timestamp": int(time.time() * 1000),
        "session_id": session_id,
        "query": query[:200],
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_yuan": cost,
        "model": config.model_name,
        "fallback": usage.get("used_fallback", False),
    }
    log_path = PathLib("data/logs/chat_costs.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Pydantic 模型 ──

class TaskRequest(BaseModel):
    """任务请求载荷。"""
    query: str
    mode: str = "privacy_enhanced"
    files: List[str] = []
    output_format: str = "markdown"


class TaskResponse(BaseModel):
    """任务执行结果。"""
    run_id: str
    status: str
    result_path: Optional[str] = None
    result_preview: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_yuan: float = 0.0
    duration_ms: int = 0
    steps: List[dict] = []


class HealthResponse(BaseModel):
    """健康检查响应。"""
    status: str
    version: str
    model: str


class WatchStatusResponse(BaseModel):
    """文件夹监控状态。"""
    running: bool
    dirs: List[str]
    trigger_mode: str


class ScheduledJobRequest(BaseModel):
    """创建定时任务请求。"""
    name: str
    description: str = ""
    schedule_preset: str
    query: str
    mode: str = "privacy_enhanced"
    file_paths: List[str] = []
    workflow_template: Optional[str] = None
    output_format: str = "markdown"


class ScheduledJobResponse(BaseModel):
    """定时任务信息。"""
    job_id: str
    name: str
    description: str
    schedule_preset: Optional[str] = None
    query: str
    mode: str
    enabled: bool
    last_run: Optional[str] = None
    last_status: Optional[str] = None


class BatchRequest(BaseModel):
    """批处理请求。"""
    query: str
    files: List[str]
    mode: str = "privacy_enhanced"


class BatchResponse(BaseModel):
    """批处理结果。"""
    batch_id: str
    total_files: int
    completed: int
    failed: int
    total_tokens_in: int
    total_tokens_out: int
    total_cost_yuan: float
    total_duration_ms: int
    success_rate: float
    errors: List[str]


# ── 会话/对话相关模型 ──

class SettingsResponse(BaseModel):
    """设置信息响应。"""
    model_name: str
    model_base_url: str
    api_key_masked: str
    fallback_enabled: bool
    fallback_ollama_base_url: str
    fallback_ollama_model: str


class SettingsUpdate(BaseModel):
    """更新设置请求（所有字段可选）。"""
    model_name: Optional[str] = None
    model_base_url: Optional[str] = None
    api_key: Optional[str] = None
    fallback_enabled: Optional[bool] = None
    fallback_ollama_base_url: Optional[str] = None
    fallback_ollama_model: Optional[str] = None


class SessionCreate(BaseModel):
    """创建会话请求。"""
    title: str = "新建会话"
    mode: str = "privacy_enhanced"


class SessionRename(BaseModel):
    """重命名会话/切换模式请求。"""
    title: str
    mode: Optional[str] = None


class MessageSend(BaseModel):
    """发送对话消息请求。"""
    message: str


# ── 应用工厂 ──

def create_api_app(config: AppConfig) -> FastAPI:
    """创建并配置 FastAPI 应用。

    Args:
        config: 应用配置。

    Returns:
        配置好的 FastAPI 实例。
    """
    app = FastAPI(
        title="Hush API",
        description="隐私优先的个人 AI 工作流助手",
        version="1.0.0",
    )

    # 将配置存入应用状态
    app.state.config = config

    # CORS：Vue 前端（:5173）跨域访问 FastAPI（:8000）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ponytail: API 处理文件需要允许的路径白名单。Gradio 端只加了系统临时目录，
    # API 端此前完全没初始化，导致 allowed_paths 为空、任何路径都被拒。
    # 启动时一次性注入：临时目录（/tasks/upload 落盘处）+ 项目根 + data/uploads。
    _tmp = str(PathLib(tempfile.gettempdir()))
    if _tmp not in config.allowed_paths:
        config.allowed_paths.append(_tmp)
    _project_root = str(PathLib(__file__).resolve().parent.parent.parent)
    if _project_root not in config.allowed_paths:
        config.allowed_paths.append(_project_root)
    _uploads = str(PathLib("data/uploads").resolve())
    if _uploads not in config.allowed_paths:
        config.allowed_paths.append(_uploads)

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """健康检查端点。"""
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            model=config.model_name,
        )

    @app.get("/settings", response_model=SettingsResponse)
    async def get_settings():
        """获取当前设置。"""
        key = config.api_key
        masked = key[:8] + "****" + key[-4:] if len(key) > 12 else ("****" if key else "")
        return SettingsResponse(
            model_name=config.model_name,
            model_base_url=config.model_base_url,
            api_key_masked=masked,
            fallback_enabled=config.fallback_enabled,
            fallback_ollama_base_url=config.fallback_ollama_base_url,
            fallback_ollama_model=config.fallback_ollama_model,
        )

    @app.patch("/settings", response_model=SettingsResponse)
    async def update_settings(req: SettingsUpdate):
        """更新设置并持久化到 config.yaml。"""
        if req.model_name is not None:
            config.model_name = req.model_name
        if req.model_base_url is not None:
            config.model_base_url = req.model_base_url
        if req.api_key is not None:
            config.api_key = req.api_key
        if req.fallback_enabled is not None:
            config.fallback_enabled = req.fallback_enabled
        if req.fallback_ollama_base_url is not None:
            config.fallback_ollama_base_url = req.fallback_ollama_base_url
        if req.fallback_ollama_model is not None:
            config.fallback_ollama_model = req.fallback_ollama_model
        config.save_to_yaml()
        key = config.api_key
        masked = key[:8] + "****" + key[-4:] if len(key) > 12 else ("****" if key else "")
        return SettingsResponse(
            model_name=config.model_name,
            model_base_url=config.model_base_url,
            api_key_masked=masked,
            fallback_enabled=config.fallback_enabled,
            fallback_ollama_base_url=config.fallback_ollama_base_url,
            fallback_ollama_model=config.fallback_ollama_model,
        )

    @app.post("/tasks", response_model=TaskResponse)
    async def create_task(request: TaskRequest):
        """执行新任务。

        Args:
            request: 包含查询、模式和文件路径的任务请求。

        Returns:
            包含 RunLog 详细信息的任务执行结果。
        """
        start_time = time.time()

        # 如果已配置，则构建 LLM
        llm = None
        if config.api_key and "sk-fake" not in config.api_key:
            from src.fallback.provider import FallbackChatModel

            if config.fallback_enabled:
                llm = FallbackChatModel(
                    primary_model=config.model_name,
                    primary_base_url=config.model_base_url,
                    api_key=config.api_key,
                    fallback_base_url=config.fallback_ollama_base_url,
                    fallback_model=config.fallback_ollama_model,
                    timeout=config.fallback_timeout_seconds,
                )
            else:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=config.model_name,
                    api_key=config.api_key,
                    base_url=config.model_base_url,
                    temperature=0.3,
                )

        try:
            run_log = run_task(
                query=request.query,
                files=request.files,
                mode=request.mode,
                config=config,
                llm=llm,
                auto_confirm=True,
                output_format=request.output_format,
            )

            duration_ms = int((time.time() - start_time) * 1000)

            # 读取结果预览（如果有的话）
            result_preview = None
            if run_log.result_path:
                from pathlib import Path
                result_file = Path(run_log.result_path)
                if result_file.exists():
                    content = result_file.read_text(encoding="utf-8")
                    result_preview = content[:500] + ("..." if len(content) > 500 else "")

            return TaskResponse(
                run_id=run_log.run_id,
                status="completed" if run_log.result_path else "failed",
                result_path=run_log.result_path,
                result_preview=result_preview,
                tokens_in=run_log.total_tokens_in,
                tokens_out=run_log.total_tokens_out,
                cost_yuan=run_log.total_cost_yuan,
                duration_ms=duration_ms,
                steps=[
                    {
                        "name": s.name,
                        "status": s.status,
                        "duration_ms": s.duration_ms,
                        "tokens_in": s.tokens_in,
                        "tokens_out": s.tokens_out,
                        "cost_yuan": s.cost_yuan,
                        "output_preview": s.output_preview,
                    }
                    for s in run_log.steps
                ],
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/tasks/upload", response_model=TaskResponse)
    async def create_task_with_upload(
        query: str = Form(...),
        mode: str = Form("privacy_enhanced"),
        output_format: str = Form("markdown"),
        files: List[UploadFile] = File(...),
    ):
        """通过文件上传执行任务。

        Args:
            query: 用户任务描述。
            mode: 处理模式。
            output_format: 期望的输出格式。
            files: 上传的文件。

        Returns:
            任务执行结果。
        """
        # 临时保存上传的文件
        import tempfile
        from pathlib import Path

        temp_dir = Path(tempfile.mkdtemp())
        file_paths = []

        for uploaded_file in files:
            file_path = temp_dir / uploaded_file.filename
            content = await uploaded_file.read()
            file_path.write_bytes(content)
            file_paths.append(str(file_path))

        # 创建任务请求
        request = TaskRequest(
            query=query,
            mode=mode,
            files=file_paths,
            output_format=output_format,
        )

        # 执行任务
        result = await create_task(request)

        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        return result

    @app.post("/tasks/stream")
    async def stream_task(
        query: str = Form(...),
        mode: str = Form("privacy_enhanced"),
        output_format: str = Form("markdown"),
        files: List[UploadFile] = File(...),
    ):
        """SSE 流式执行任务：上传文件 → run_task → 实时推送步骤/token/done。

        事件协议：{type:"step",...} / {type:"token",text} / {type:"done",...} / {type:"error",message}
        """
        # 先把上传文件落盘到临时目录，拿到服务器路径
        temp_dir = PathLib(tempfile.mkdtemp())
        file_paths = []
        for uploaded in files:
            fp = temp_dir / uploaded.filename
            fp.write_bytes(await uploaded.read())
            file_paths.append(str(fp))

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def push(evt):
            loop.call_soon_threadsafe(queue.put_nowait, evt)

        def run_in_thread():
            try:
                llm = _build_chat_llm(config)
                run_log = run_task(
                    query=query,
                    files=file_paths,
                    mode=mode,
                    config=config,
                    llm=llm,
                    auto_confirm=True,
                    output_format=output_format,
                    step_callback=lambda s: push({**s, "type": "step"}),
                    stream_callback=lambda t: push({"type": "token", "text": t}),
                )
                result_preview = None
                if run_log.result_path:
                    rf = PathLib(run_log.result_path)
                    if rf.exists():
                        content = rf.read_text(encoding="utf-8")
                        result_preview = content[:500] + ("..." if len(content) > 500 else "")
                push({
                    "type": "done",
                    "run_id": run_log.run_id,
                    "result_path": run_log.result_path,
                    "result_preview": result_preview,
                    "tokens_in": run_log.total_tokens_in,
                    "tokens_out": run_log.total_tokens_out,
                    "cost_yuan": run_log.total_cost_yuan,
                    "fallback": run_log.fallback,
                    "steps": [asdict(s) for s in run_log.steps],
                })
            except Exception as e:
                push({"type": "error", "message": str(e)})
            finally:
                push(None)
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

        threading.Thread(target=run_in_thread, daemon=True).start()

        async def event_gen():
            while True:
                evt = await queue.get()
                if evt is None:
                    break
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    @app.get("/config")
    async def get_config():
        """获取当前配置（非敏感部分）。

        Returns:
            公开的配置值。
        """
        return {
            "model": config.model_name,
            "max_file_size_mb": config.max_file_size_mb,
            "max_tokens_per_request": config.max_tokens_per_request,
            "max_cost_per_request_yuan": config.max_cost_per_request_yuan,
            "allowed_paths": config.allowed_paths,
            "redaction_enabled": config.redaction_enabled,
        }

    # ── v0.3：监控端点 ──

    @app.get("/watch/status", response_model=WatchStatusResponse)
    async def watch_status():
        """获取当前文件夹监控状态。"""
        from src.monitor.watcher import _active_watcher

        if _active_watcher is not None and _active_watcher.is_running:
            return WatchStatusResponse(
                running=True,
                dirs=config.watch_dirs,
                trigger_mode=config.watch_trigger_mode,
            )
        return WatchStatusResponse(
            running=False,
            dirs=config.watch_dirs,
            trigger_mode=config.watch_trigger_mode,
        )

    @app.post("/watch/start")
    async def watch_start():
        """启动文件夹监控。"""
        from src.monitor.watcher import start_watcher, _active_watcher

        if _active_watcher is not None and _active_watcher.is_running:
            return {"status": "already_running", "message": "Watcher is already running"}

        def _on_change(files: List[str]) -> None:
            """处理文件变更 — 为每个变更文件触发 run_task。"""
            query = "请总结以下变更文件的内容："

            for f in files:
                try:
                    run_task(
                        query=query,
                        files=[f],
                        mode=config.watch_mode,
                        config=config,
                        auto_confirm=True,
                        output_format=config.default_output_format,
                    )
                except Exception:
                    pass

            # 发送通知
            if config.notifications_enabled:
                try:
                    from src.notify.dispatch import notify
                    notify(
                        title="📁 文件变更处理完成",
                        message=f"处理了 {len(files)} 个文件",
                        engine=config.notifications_engine,
                        log_file=config.notifications_log_file,
                    )
                except Exception:
                    pass

        start_watcher(config, _on_change)
        return {"status": "started", "message": "Watcher started"}

    @app.post("/watch/stop")
    async def watch_stop():
        """停止文件夹监控。"""
        from src.monitor.watcher import stop_watcher
        stop_watcher()
        return {"status": "stopped", "message": "Watcher stopped"}

    # ── v0.3：调度器端点 ──

    _engine: Any = None

    def _ensure_engine():
        """延迟初始化调度器引擎。"""
        nonlocal _engine
        if _engine is None:
            from src.scheduler.engine import SchedulerEngine

            def _exec_cb(job: Any) -> None:
                """执行定时任务。"""
                try:
                    run_task(
                        query=job.query,
                        files=job.file_paths,
                        mode=job.mode,
                        config=config,
                        auto_confirm=True,
                        output_format=job.output_format,
                    )

                    if config.notifications_enabled:
                        from src.notify.dispatch import notify
                        notify(
                            title="⏰ 定时任务完成",
                            message=f"{job.name}: {job.query[:60]}",
                            engine=config.notifications_engine,
                            log_file=config.notifications_log_file,
                        )
                except Exception:
                    if config.notifications_enabled:
                        from src.notify.dispatch import notify
                        notify(
                            title="❌ 定时任务失败",
                            message=f"{job.name}: {job.query[:60]}",
                            engine=config.notifications_engine,
                            log_file=config.notifications_log_file,
                        )

            _engine = SchedulerEngine(config, _exec_cb)
            _engine.start()
        return _engine

    @app.get("/scheduler/jobs", response_model=List[ScheduledJobResponse])
    async def list_scheduled_jobs():
        """列出所有定时任务。"""
        engine = _ensure_engine()
        jobs = engine.list_jobs()
        return [
            ScheduledJobResponse(
                job_id=j["job_id"],
                name=j.get("name", ""),
                description=j.get("description", ""),
                schedule_preset=j.get("schedule_preset"),
                query=j.get("query", ""),
                mode=j.get("mode", "privacy_enhanced"),
                enabled=j.get("enabled", True),
                last_run=j.get("last_run"),
                last_status=j.get("last_status"),
            )
            for j in jobs
        ]

    @app.post("/scheduler/jobs", response_model=ScheduledJobResponse)
    async def create_scheduled_job(request: ScheduledJobRequest):
        """从预设创建新的定时任务。"""
        from src.scheduler.jobs import PRESET_SCHEDULES, ScheduledJob, JobStore

        preset = PRESET_SCHEDULES.get(request.schedule_preset)
        if not preset:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown preset '{request.schedule_preset}'. Available: {list(PRESET_SCHEDULES.keys())}",
            )

        job = ScheduledJob(
            name=request.name,
            description=request.description,
            schedule_preset=request.schedule_preset,
            trigger_type=preset["trigger"],
            trigger_kwargs={k: v for k, v in preset.items() if k not in ("label", "trigger")},
            query=request.query,
            mode=request.mode,
            file_paths=request.file_paths,
            workflow_template=request.workflow_template,
            output_format=request.output_format,
        )

        # 持久化
        store = JobStore(config.scheduler_jobs_dir)
        store.save(job)

        # 注册到引擎
        engine = _ensure_engine()
        engine.add_job(job)

        return ScheduledJobResponse(
            job_id=job.job_id,
            name=job.name,
            description=job.description,
            schedule_preset=job.schedule_preset,
            query=job.query,
            mode=job.mode,
            enabled=job.enabled,
            last_run=job.last_run,
            last_status=job.last_status,
        )

    @app.delete("/scheduler/jobs/{job_id}")
    async def delete_scheduled_job(job_id: str):
        """删除定时任务。"""
        from src.scheduler.jobs import JobStore

        engine = _ensure_engine()
        ok = engine.remove_job(job_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

        # 移除持久化
        store = JobStore(config.scheduler_jobs_dir)
        store.delete(job_id)
        return {"status": "deleted", "job_id": job_id}

    @app.post("/scheduler/jobs/{job_id}/pause")
    async def pause_scheduled_job(job_id: str):
        """暂停定时任务。"""
        engine = _ensure_engine()
        ok = engine.pause_job(job_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return {"status": "paused", "job_id": job_id}

    @app.post("/scheduler/jobs/{job_id}/resume")
    async def resume_scheduled_job(job_id: str):
        """恢复已暂停的定时任务。"""
        engine = _ensure_engine()
        ok = engine.resume_job(job_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return {"status": "resumed", "job_id": job_id}

    # ── v0.3：批处理端点 ──

    @app.post("/batch", response_model=BatchResponse)
    async def batch_run(request: BatchRequest):
        """跨多个文件执行批处理任务。"""
        from src.batch.runner import BatchRunner

        runner = BatchRunner(config)
        report = runner.run(
            query=request.query,
            files=request.files,
            mode=request.mode,
            auto_confirm=True,
        )
        return BatchResponse(
            batch_id=report.batch_id,
            total_files=report.total_files,
            completed=report.completed,
            failed=report.failed,
            total_tokens_in=report.total_tokens_in,
            total_tokens_out=report.total_tokens_out,
            total_cost_yuan=report.total_cost_yuan,
            total_duration_ms=report.total_duration_ms,
            success_rate=report.success_rate,
            errors=report.errors,
        )

    # ── v0.3：通知端点 ──

    @app.get("/notifications")
    async def list_notifications(limit: int = 50):
        """从 JSONL 日志中列出最近的通知历史。"""
        log_path = PathLib(config.notifications_log_file)
        if not log_path.exists():
            return {"notifications": [], "total": 0}

        entries = []
        try:
            lines = log_path.read_text(encoding="utf-8").strip().split("\n")
            for line in lines[-limit:]:
                if line.strip():
                    entries.append(json.loads(line))
        except Exception:
            pass

        return {"notifications": entries, "total": len(entries)}

    # ── v1.0：回放端点 ──

    @app.get("/replay/runs")
    async def list_runs(
        mode: Optional[str] = None,
        status: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ):
        """列出历史 RunLog，支持可选筛选。"""
        from src.replay.loader import RunLogLoader

        loader = RunLogLoader()
        result = loader.list_all(
            mode_filter=mode,
            status_filter=status,
            search=search,
            limit=limit,
            offset=offset,
        )
        return {
            "total": result.total,
            "items": [
                {
                    "run_id": r.run_id,
                    "user_query": r.user_query,
                    "mode": r.mode,
                    "model": r.model,
                    "total_tokens_in": r.total_tokens_in,
                    "total_tokens_out": r.total_tokens_out,
                    "total_cost_yuan": r.total_cost_yuan,
                    "step_count": r.step_count,
                    "fallback": r.fallback,
                    "status": r.status,
                }
                for r in result.items
            ],
        }

    @app.get("/replay/runs/{run_id}")
    async def get_run_detail(run_id: str):
        """通过 run_id 获取完整的 RunLog 详情。"""
        from src.replay.loader import RunLogLoader

        loader = RunLogLoader()
        run_log = loader.load_by_id(run_id)
        if run_log is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

        return {
            "run_id": run_log.run_id,
            "user_query": run_log.user_query,
            "mode": run_log.mode,
            "model": run_log.model,
            "fallback": run_log.fallback,
            "result_path": run_log.result_path,
            "total_tokens_in": run_log.total_tokens_in,
            "total_tokens_out": run_log.total_tokens_out,
            "total_cost_yuan": run_log.total_cost_yuan,
            "steps": [
                {
                    "step_id": s.step_id,
                    "name": s.name,
                    "input_preview": s.input_preview,
                    "output_preview": s.output_preview,
                    "duration_ms": s.duration_ms,
                    "tokens_in": s.tokens_in,
                    "tokens_out": s.tokens_out,
                    "cost_yuan": s.cost_yuan,
                    "status": s.status,
                }
                for s in run_log.steps
            ],
        }

    # ── v1.0：知识库/搜索端点 ──

    @app.get("/knowledge/search")
    async def search_knowledge(
        query: str,
        top_k: int = 5,
        collection: Optional[str] = None,
    ):
        """对已索引的知识进行语义搜索。"""
        from src.knowledge.embedder import OllamaEmbedder
        from src.knowledge.search import Searcher
        from src.knowledge.store import KnowledgeStore

        store = KnowledgeStore(persist_dir=config.knowledge_chroma_dir)
        embedder = OllamaEmbedder(
            base_url=config.knowledge_embed_base_url,
            model=config.knowledge_embed_model,
        )
        searcher = Searcher(store, embedder)

        colls = [collection] if collection else None
        resp = searcher.search(query, top_k=top_k, collections=colls)

        return {
            "query": resp.query,
            "embedding_available": resp.embedding_available,
            "total_hits": resp.total_hits,
            "results": [
                {
                    "document": r.document[:300],
                    "score": r.score,
                    "source": r.source,
                    "doc_type": r.doc_type,
                }
                for r in resp.results
            ],
        }

    # ── v1.0：仪表盘统计端点 ──

    @app.get("/dashboard/stats")
    async def dashboard_stats(days: int = 7):
        """聚合仪表盘统计数据。"""
        from src.dashboard.aggregator import DashboardAggregator

        agg = DashboardAggregator()
        stats = agg.aggregate(days=days)

        return {
            "total_tasks": stats.total_tasks,
            "successful_tasks": stats.successful_tasks,
            "failed_tasks": stats.failed_tasks,
            "total_tokens_in": stats.total_tokens_in,
            "total_tokens_out": stats.total_tokens_out,
            "total_cost_yuan": stats.total_cost_yuan,
            "total_duration_ms": stats.total_duration_ms,
            "fallback_count": stats.fallback_count,
            "mode_distribution": stats.mode_distribution,
            "model_distribution": stats.model_distribution,
            "daily": [
                {
                    "date": d.date,
                    "task_count": d.task_count,
                    "tokens_in": d.tokens_in,
                    "tokens_out": d.tokens_out,
                    "cost_yuan": d.cost_yuan,
                }
                for d in stats.daily
            ],
            "recent_tasks": stats.recent_tasks,
        }

    # ── v1.2：多轮对话 API（会话管理 + 消息 + 文件上传）──

    from src.knowledge.session_store import SessionStore

    @app.get("/sessions")
    async def list_sessions():
        """列出所有会话，按最近活跃倒序。"""
        store = SessionStore()
        sessions = [{**s, "session_id": s["id"]} for s in store.list_sessions()]
        return {"sessions": sessions, "total": len(sessions)}

    @app.post("/sessions")
    async def create_session(req: SessionCreate):
        """创建新会话。"""
        store = SessionStore()
        session = store.create_session(mode=req.mode, title=req.title)
        return {**session, "session_id": session["id"]}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        """获取单个会话元数据。"""
        store = SessionStore()
        s = store.get_session(session_id)
        if not s:
            raise HTTPException(status_code=404, detail="Session not found")
        return {**s, "session_id": s["id"]}

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        """删除会话（列表 + 消息文件）。"""
        store = SessionStore()
        store.delete_session(session_id)
        return {"status": "deleted", "session_id": session_id}

    @app.patch("/sessions/{session_id}")
    async def rename_session(session_id: str, req: SessionRename):
        """重命名会话，可选切换隐私模式。"""
        store = SessionStore()
        store.rename_session(session_id, req.title)
        if req.mode is not None:
            store.update_session_mode(session_id, req.mode)
        s = store.get_session(session_id)
        return {**s, "session_id": s["id"]} if s else None

    @app.get("/sessions/{session_id}/messages")
    async def get_messages(session_id: str):
        """获取会话全部消息，按时间正序。展示时还原脱敏占位符。"""
        from src.workflow.postprocess import restore_redactions
        store = SessionStore()
        if not store.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        msgs = store.get_messages(session_id)
        display = []
        for m in msgs:
            content = m["content"]
            rm = m.get("redact_map")
            if rm:
                content = restore_redactions(content, rm)
            display.append({"role": m["role"], "content": content, "timestamp": m["timestamp"]})
        return {"messages": display, "total": len(display)}

    @app.post("/sessions/{session_id}/upload")
    async def upload_files(session_id: str, files: List[UploadFile] = File(...)):
        """上传文件到指定会话（向量化存 ChromaDB，供 RAG 检索）。"""
        from src.knowledge.indexer import index_session_files

        store = SessionStore()
        if not store.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        # 保存上传文件到临时目录
        import tempfile
        tmp_dir = PathLib(tempfile.mkdtemp())
        saved = []
        for f in files:
            fp = tmp_dir / f.filename
            fp.write_bytes(await f.read())
            saved.append(fp)

        session = store.get_session(session_id)
        mode = session.get("mode", "privacy_enhanced") if session else "privacy_enhanced"
        added = index_session_files(saved, session_id, config, mode=mode)

        # 清理临时文件
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        return {"session_id": session_id, "chunks_indexed": added}

    @app.get("/sessions/{session_id}/cost")
    async def get_session_cost(session_id: str):
        """获取会话的 token 消耗统计和账户余额。"""
        from src.tools.cost import fetch_real_balance

        store = SessionStore()
        if not store.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        # 从日志中汇总该会话的消耗
        log_path = PathLib("data/logs/chat_costs.jsonl")
        total_tokens_in = 0
        total_tokens_out = 0
        total_cost = 0.0
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("session_id") == session_id:
                        total_tokens_in += entry.get("tokens_in", 0)
                        total_tokens_out += entry.get("tokens_out", 0)
                        total_cost += entry.get("cost_yuan", 0.0)
                except Exception:
                    continue

        # 查询真实余额
        balance = fetch_real_balance(config.api_key, config.model_base_url)

        return {
            "session_id": session_id,
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "cost_yuan": round(total_cost, 6),
            "balance_yuan": balance,
            "budget_yuan": config.budget_yuan,
        }

    @app.post("/sessions/{session_id}/messages")
    async def send_message(session_id: str, req: MessageSend):
        """发送消息并获取回复（非流式，含 RAG 检索 + 长期记忆）。

        流程：脱敏 → smart_retrieve（文件检索 + 意图路由 + 长期记忆）
              → chat_stream（LLM 生成）→ 持久化 → 记录费用。
        """
        from src.agent.chat import chat_stream, smart_retrieve
        from src.workflow.postprocess import restore_redactions
        from src.tools.redaction import detect_sensitive, redact

        store = SessionStore()
        if not store.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        message = req.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="消息不能为空")

        # 取历史消息
        history = store.get_messages(session_id)

        # 脱敏用户输入
        user_redact_map = {}
        if config.redaction_enabled:
            matches = detect_sensitive(message, config.redaction_rules)
            if matches:
                message, user_redact_map = redact(message, matches)

        # 根据会话模式构建 LLM
        session = store.get_session(session_id)
        session_mode = session.get("mode", "privacy_enhanced") if session else "privacy_enhanced"
        if session_mode == "local_fallback":
            llm = _build_local_llm(config)
            if llm is None:
                raise HTTPException(status_code=503, detail="本地模型未配置或 Ollama 未运行，请先安装并启动 Ollama")
        else:
            llm = _build_chat_llm(config)
            if llm is None:
                raise HTTPException(status_code=503, detail="未配置 LLM（api_key 为空），无法生成回复")

        # 意图感知检索
        file_context, long_term_memory, file_redact_map = smart_retrieve(
            session_id, message, config, llm, history
        )
        merged_redact_map = {**user_redact_map, **file_redact_map}

        # 生成回复（收集完整文本，非流式）
        accumulated = ""
        usage = {}
        try:
            for chunk in chat_stream(
                history, message, llm, file_context, long_term_memory, usage_out=usage
            ):
                accumulated += chunk
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"生成失败：{exc}")

        # 展示用文本（还原脱敏）
        display_text = restore_redactions(accumulated, merged_redact_map)

        # 持久化：存脱敏后内容
        store.add_message(session_id, "user", message, redact_map=user_redact_map or None)
        store.add_message(session_id, "assistant", accumulated, redact_map=merged_redact_map or None)

        # 持久化对话记忆
        if getattr(config, 'conversation_memory_enabled', True):
            try:
                from src.agent.memory import store_conversation_turn
                from src.knowledge.embedder import OllamaEmbedder
                from src.knowledge.store import KnowledgeStore
                store_conv = KnowledgeStore(persist_dir=config.knowledge_chroma_dir)
                embedder = OllamaEmbedder(
                    base_url=config.knowledge_embed_base_url,
                    model=config.knowledge_embed_model,
                )
                store_conversation_turn(store_conv, embedder, session_id, message, accumulated)
            except Exception as e:
                logger.warning("对话记忆持久化失败: %s", e)

        # 记录费用
        _log_chat_cost(session_id, message, accumulated, usage, config)

        return {
            "session_id": session_id,
            "reply": display_text,
            "used_fallback": usage.get("used_fallback", False),
            "tokens_in": usage.get("tokens_in", 0),
            "tokens_out": usage.get("tokens_out", 0),
        }

    @app.post("/sessions/{session_id}/messages/stream")
    async def send_message_stream(session_id: str, req: MessageSend):
        """SSE 流式发送消息，逐字推送 LLM 回复。

        复用 smart_retrieve + chat_stream；token 事件逐字推送，done 事件含 token/费用。
        持久化（SessionStore + 对话记忆 + 费用）在流结束后统一执行。
        """
        from src.agent.chat import chat_stream, smart_retrieve
        from src.tools.redaction import detect_sensitive, redact
        from src.workflow.postprocess import restore_redactions

        store = SessionStore()
        if not store.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        message = req.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="消息不能为空")

        history = store.get_messages(session_id)

        # 根据会话模式构建 LLM
        session = store.get_session(session_id)
        session_mode = session.get("mode", "privacy_enhanced") if session else "privacy_enhanced"
        if session_mode == "local_fallback":
            llm = _build_local_llm(config)
            if llm is None:
                raise HTTPException(status_code=503, detail="本地模型未配置或 Ollama 未运行，请先安装并启动 Ollama")
        else:
            llm = _build_chat_llm(config)
            if llm is None:
                raise HTTPException(status_code=503, detail="未配置 LLM（api_key 为空），无法生成回复")

        # 脱敏用户输入
        user_redact_map = {}
        if config.redaction_enabled:
            matches = detect_sensitive(message, config.redaction_rules)
            if matches:
                message, user_redact_map = redact(message, matches)

        file_context, long_term_memory, file_redact_map = smart_retrieve(
            session_id, message, config, llm, history
        )
        merged_redact_map = {**user_redact_map, **file_redact_map}

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def push(evt):
            loop.call_soon_threadsafe(queue.put_nowait, evt)

        def run_in_thread():
            import time as _t
            accumulated = ""
            usage = {}
            try:
                # 思考过程步骤推送（供前端步骤时间线展示「思考过程」）
                push({"type": "step", "step_id": 1, "name": "redact_input", "status": "success",
                      "duration_ms": 0, "input_preview": "",
                      "output_preview": f"脱敏 {len(user_redact_map)} 项",
                      "tokens_in": 0, "tokens_out": 0, "cost_yuan": 0})
                push({"type": "step", "step_id": 2, "name": "retrieve", "status": "success",
                      "duration_ms": 0, "input_preview": message[:80],
                      "output_preview": f"文件上下文 {len(file_context)} 字",
                      "tokens_in": 0, "tokens_out": 0, "cost_yuan": 0})
                t0 = _t.time()
                # 流式逐块推送；占位符实时还原为原文，避免用户看到 ORG_1 之类占位符。
                # ponytail: 每 chunk 推送已累积文本的完整还原版（O(n²)，对话回复量级可接受）；
                #           升级路径：改为增量 diff 推送以降低长回复带宽。
                for chunk in chat_stream(
                    history, message, llm, file_context, long_term_memory, usage_out=usage
                ):
                    accumulated += chunk
                    push({"type": "token", "text": restore_redactions(accumulated, merged_redact_map)})
                push({"type": "step", "step_id": 3, "name": "llm_call", "status": "success",
                      "duration_ms": int((_t.time() - t0) * 1000), "input_preview": message[:80],
                      "output_preview": accumulated[:200],
                      "tokens_in": usage.get("tokens_in", 0), "tokens_out": usage.get("tokens_out", 0),
                      "cost_yuan": 0})
                display_text = restore_redactions(accumulated, merged_redact_map)
                # 持久化
                store.add_message(session_id, "user", message, redact_map=user_redact_map or None)
                store.add_message(session_id, "assistant", accumulated, redact_map=merged_redact_map or None)
                # 对话记忆
                if getattr(config, "conversation_memory_enabled", True):
                    try:
                        from src.agent.memory import store_conversation_turn
                        from src.knowledge.embedder import OllamaEmbedder
                        from src.knowledge.store import KnowledgeStore

                        store_conv = KnowledgeStore(persist_dir=config.knowledge_chroma_dir)
                        embedder = OllamaEmbedder(
                            base_url=config.knowledge_embed_base_url,
                            model=config.knowledge_embed_model,
                        )
                        store_conversation_turn(store_conv, embedder, session_id, message, accumulated)
                    except Exception as e:
                        logger.warning("对话记忆持久化失败: %s", e)
                _log_chat_cost(session_id, message, accumulated, usage, config)
                push({
                    "type": "done",
                    "reply": display_text,
                    "used_fallback": usage.get("used_fallback", False),
                    "tokens_in": usage.get("tokens_in", 0),
                    "tokens_out": usage.get("tokens_out", 0),
                })
            except Exception as e:
                push({"type": "error", "message": str(e)})
            finally:
                push(None)

        threading.Thread(target=run_in_thread, daemon=True).start()

        async def event_gen():
            while True:
                evt = await queue.get()
                if evt is None:
                    break
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    # 生产模式：若前端已 build（frontend/dist 存在），直接由 FastAPI serve 静态文件，
    # 这样只需 `python app.py` 一个进程即可同时提供 API 与前端 UI（无需 npm run dev）。
    _dist = PathLib(__file__).resolve().parents[2] / "frontend" / "dist"
    if _dist.is_dir():
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse

        app.mount("/assets", StaticFiles(directory=str(_dist / "assets")), name="assets")

        @app.get("/{full_path:path}")
        async def _serve_spa(full_path: str):
            return FileResponse(str(_dist / "index.html"))

    return app
