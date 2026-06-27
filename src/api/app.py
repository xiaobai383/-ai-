"""FastAPI application — REST API for the AI Workflow Assistant."""
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
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
        version="0.2.0",
    )

    # Store config in app state
    app.state.config = config

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version="0.2.0",
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

    return app
