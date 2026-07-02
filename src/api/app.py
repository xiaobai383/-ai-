"""FastAPI 应用 — AI 工作流助手的 REST API。"""
import json
import time
from pathlib import Path as PathLib
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from src.agent.app import run_task
from src.config import AppConfig
from src.workflow.templates import load_workflow_template, list_templates


# ── Pydantic 模型 ──

class TaskRequest(BaseModel):
    """任务请求载荷。"""
    query: str
    mode: str = "privacy_enhanced"
    files: List[str] = []
    workflow_template: Optional[str] = None
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


class TemplateInfo(BaseModel):
    """工作流模板信息。"""
    name: str
    description: str
    steps: List[str]


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


# ── 应用工厂 ──

def create_api_app(config: AppConfig) -> FastAPI:
    """创建并配置 FastAPI 应用。

    Args:
        config: 应用配置。

    Returns:
        配置好的 FastAPI 实例。
    """
    app = FastAPI(
        title="个人 AI 工作流助手 API",
        description="可控上传、可审计执行的 AI 工作流助手",
        version="1.0.0",
    )

    # 将配置存入应用状态
    app.state.config = config

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """健康检查端点。"""
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            model=config.model_name,
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

        # 如果指定了工作流模板，则加载它
        workflow_steps = None
        if request.workflow_template:
            try:
                template = load_workflow_template(request.workflow_template)
                workflow_steps = template.get("steps", [])
            except FileNotFoundError:
                raise HTTPException(
                    status_code=404,
                    detail=f"Workflow template '{request.workflow_template}' not found",
                )

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

    @app.get("/templates", response_model=List[TemplateInfo])
    async def get_templates():
        """列出所有可用的工作流模板。

        Returns:
            模板信息列表。
        """
        templates = list_templates()
        return [
            TemplateInfo(
                name=t["name"],
                description=t.get("description", ""),
                steps=[s.get("name", "") for s in t.get("steps", [])],
            )
            for t in templates
        ]

    @app.get("/templates/{template_name}", response_model=TemplateInfo)
    async def get_template(template_name: str):
        """获取指定的工作流模板。

        Args:
            template_name: 模板名称。

        Returns:
            模板详细信息。
        """
        try:
            template = load_workflow_template(template_name)
            return TemplateInfo(
                name=template["name"],
                description=template.get("description", ""),
                steps=[s.get("name", "") for s in template.get("steps", [])],
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Template '{template_name}' not found",
            )

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
            """处理文件变更 — 为每个文件触发 run_task。"""
            for f in files:
                try:
                    run_task(
                        query=config.watch_workflow_template,
                        files=[f],
                        mode=config.watch_mode,
                        config=config,
                        auto_confirm=True,
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
                    )
                except Exception:
                    pass

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

    return app
