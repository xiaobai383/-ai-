"""FastAPI application — REST API for the AI Workflow Assistant."""
import json
import time
from pathlib import Path as PathLib
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from src.agent.app import run_task
from src.config import AppConfig
from src.workflow.templates import load_workflow_template, list_templates


# ── Pydantic models ──

class TaskRequest(BaseModel):
    """Task request payload."""
    query: str
    mode: str = "privacy_enhanced"
    files: List[str] = []
    workflow_template: Optional[str] = None
    output_format: str = "markdown"


class TaskResponse(BaseModel):
    """Task execution result."""
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
    """Workflow template information."""
    name: str
    description: str
    steps: List[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model: str


class WatchStatusResponse(BaseModel):
    """Folder watcher status."""
    running: bool
    dirs: List[str]
    trigger_mode: str


class ScheduledJobRequest(BaseModel):
    """Create scheduled job request."""
    name: str
    description: str = ""
    schedule_preset: str
    query: str
    mode: str = "privacy_enhanced"
    file_paths: List[str] = []
    workflow_template: Optional[str] = None
    output_format: str = "markdown"


class ScheduledJobResponse(BaseModel):
    """Scheduled job information."""
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
    """Batch processing request."""
    query: str
    files: List[str]
    mode: str = "privacy_enhanced"


class BatchResponse(BaseModel):
    """Batch processing result."""
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


# ── App factory ──

def create_api_app(config: AppConfig) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Application configuration.

    Returns:
        Configured FastAPI instance.
    """
    app = FastAPI(
        title="个人 AI 工作流助手 API",
        description="可控上传、可审计执行的 AI 工作流助手",
        version="1.0.0",
    )

    # Store config in app state
    app.state.config = config

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            model=config.model_name,
        )

    @app.post("/tasks", response_model=TaskResponse)
    async def create_task(request: TaskRequest):
        """Execute a new task.

        Args:
            request: Task request with query, mode, and file paths.

        Returns:
            Task execution result with RunLog details.
        """
        start_time = time.time()

        # Load workflow template if specified
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

        # Build LLM if configured
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

            # Read result preview if available
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
        """Execute a task with file upload.

        Args:
            query: User task description.
            mode: Processing mode.
            output_format: Desired output format.
            files: Uploaded files.

        Returns:
            Task execution result.
        """
        # Save uploaded files temporarily
        import tempfile
        from pathlib import Path

        temp_dir = Path(tempfile.mkdtemp())
        file_paths = []

        for uploaded_file in files:
            file_path = temp_dir / uploaded_file.filename
            content = await uploaded_file.read()
            file_path.write_bytes(content)
            file_paths.append(str(file_path))

        # Create task request
        request = TaskRequest(
            query=query,
            mode=mode,
            files=file_paths,
            output_format=output_format,
        )

        # Execute task
        result = await create_task(request)

        # Cleanup temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        return result

    @app.get("/templates", response_model=List[TemplateInfo])
    async def get_templates():
        """List available workflow templates.

        Returns:
            List of template information.
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
        """Get a specific workflow template.

        Args:
            template_name: Name of the template.

        Returns:
            Template details.
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
        """Get current configuration (non-sensitive parts).

        Returns:
            Public configuration values.
        """
        return {
            "model": config.model_name,
            "max_file_size_mb": config.max_file_size_mb,
            "max_tokens_per_request": config.max_tokens_per_request,
            "max_cost_per_request_yuan": config.max_cost_per_request_yuan,
            "allowed_paths": config.allowed_paths,
            "redaction_enabled": config.redaction_enabled,
        }

    # ── v0.3: Watch endpoints ──

    @app.get("/watch/status", response_model=WatchStatusResponse)
    async def watch_status():
        """Get current folder watcher status."""
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
        """Start the folder watcher."""
        from src.monitor.watcher import start_watcher, _active_watcher

        if _active_watcher is not None and _active_watcher.is_running:
            return {"status": "already_running", "message": "Watcher is already running"}

        def _on_change(files: List[str]) -> None:
            """Handle file changes — trigger run_task for each file."""
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
        """Stop the folder watcher."""
        from src.monitor.watcher import stop_watcher
        stop_watcher()
        return {"status": "stopped", "message": "Watcher stopped"}

    # ── v0.3: Scheduler endpoints ──

    _engine: Any = None

    def _ensure_engine():
        """Lazy-init the scheduler engine."""
        nonlocal _engine
        if _engine is None:
            from src.scheduler.engine import SchedulerEngine

            def _exec_cb(job: Any) -> None:
                """Execute a scheduled job."""
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
        """List all scheduled jobs."""
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
        """Create a new scheduled job from a preset."""
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

        # Persist
        store = JobStore(config.scheduler_jobs_dir)
        store.save(job)

        # Register with engine
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
        """Delete a scheduled job."""
        from src.scheduler.jobs import JobStore

        engine = _ensure_engine()
        ok = engine.remove_job(job_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

        # Remove persistence
        store = JobStore(config.scheduler_jobs_dir)
        store.delete(job_id)
        return {"status": "deleted", "job_id": job_id}

    @app.post("/scheduler/jobs/{job_id}/pause")
    async def pause_scheduled_job(job_id: str):
        """Pause a scheduled job."""
        engine = _ensure_engine()
        ok = engine.pause_job(job_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return {"status": "paused", "job_id": job_id}

    @app.post("/scheduler/jobs/{job_id}/resume")
    async def resume_scheduled_job(job_id: str):
        """Resume a paused scheduled job."""
        engine = _ensure_engine()
        ok = engine.resume_job(job_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return {"status": "resumed", "job_id": job_id}

    # ── v0.3: Batch endpoint ──

    @app.post("/batch", response_model=BatchResponse)
    async def batch_run(request: BatchRequest):
        """Execute a batch task across multiple files."""
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

    # ── v0.3: Notifications endpoint ──

    @app.get("/notifications")
    async def list_notifications(limit: int = 50):
        """List recent notification history from JSONL log."""
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

    # ── v1.0: Replay endpoints ──

    @app.get("/replay/runs")
    async def list_runs(
        mode: Optional[str] = None,
        status: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ):
        """List historical RunLogs with optional filters."""
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
        """Get full RunLog detail by run_id."""
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

    # ── v1.0: Knowledge / Search endpoint ──

    @app.get("/knowledge/search")
    async def search_knowledge(
        query: str,
        top_k: int = 5,
        collection: Optional[str] = None,
    ):
        """Semantic search across indexed knowledge."""
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

    # ── v1.0: Dashboard stats endpoint ──

    @app.get("/dashboard/stats")
    async def dashboard_stats(days: int = 7):
        """Aggregate dashboard statistics."""
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
