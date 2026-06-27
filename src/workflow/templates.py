"""Workflow template management — load, list, and execute YAML-based workflows."""
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Default templates directory
TEMPLATES_DIR = Path(__file__).parent.parent.parent / "workflows"


def _ensure_templates_dir() -> Path:
    """Ensure the templates directory exists."""
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    return TEMPLATES_DIR


def list_templates() -> List[Dict[str, Any]]:
    """List all available workflow templates.

    Returns:
        List of template metadata dicts with 'name', 'description', 'steps'.
    """
    templates_dir = _ensure_templates_dir()
    templates = []

    for template_file in templates_dir.glob("*.yaml"):
        try:
            with open(template_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            templates.append({
                "name": data.get("name", template_file.stem),
                "description": data.get("description", ""),
                "steps": data.get("steps", []),
            })
        except Exception:
            continue

    return templates


def load_workflow_template(name: str) -> Dict[str, Any]:
    """Load a workflow template by name.

    Args:
        name: Template name (without .yaml extension).

    Returns:
        Template data dict.

    Raises:
        FileNotFoundError: If template not found.
    """
    templates_dir = _ensure_templates_dir()
    template_path = templates_dir / f"{name}.yaml"

    if not template_path.exists():
        raise FileNotFoundError(f"Workflow template '{name}' not found at {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Ensure required fields
    if "name" not in data:
        data["name"] = name
    if "steps" not in data:
        data["steps"] = []

    return data


def create_template(name: str, description: str, steps: List[Dict[str, Any]]) -> Path:
    """Create a new workflow template.

    Args:
        name: Template name.
        description: Template description.
        steps: List of step definitions.

    Returns:
        Path to the created template file.
    """
    templates_dir = _ensure_templates_dir()
    template_path = templates_dir / f"{name}.yaml"

    template_data = {
        "name": name,
        "description": description,
        "steps": steps,
    }

    with open(template_path, "w", encoding="utf-8") as f:
        yaml.dump(template_data, f, allow_unicode=True, default_flow_style=False)

    return template_path


def delete_template(name: str) -> bool:
    """Delete a workflow template.

    Args:
        name: Template name.

    Returns:
        True if deleted, False if not found.
    """
    templates_dir = _ensure_templates_dir()
    template_path = templates_dir / f"{name}.yaml"

    if template_path.exists():
        template_path.unlink()
        return True
    return False
