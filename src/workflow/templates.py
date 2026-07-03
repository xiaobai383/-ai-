"""工作流模板管理 —— 加载、列出和执行基于 YAML 的工作流。"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# 默认模板目录
TEMPLATES_DIR = Path(__file__).parent.parent.parent / "workflows"


def _ensure_templates_dir(dir_override: str | None = None) -> Path:
    """确保模板目录存在。"""
    target = Path(dir_override) if dir_override else TEMPLATES_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def list_templates(templates_dir: str | None = None) -> List[Dict[str, Any]]:
    """列出所有可用的工作流模板。

    返回：
        模板元数据字典列表，包含 'name'、'description'、'steps' 字段。
    """
    templates_dir = _ensure_templates_dir(templates_dir)
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


def load_workflow_template(name: str, templates_dir: str | None = None) -> Dict[str, Any]:
    """按名称加载工作流模板。

    参数：
        name: 模板名称（不含 .yaml 扩展名）。
        templates_dir: 可选的模板目录覆盖。

    返回：
        模板数据字典。

    异常：
        FileNotFoundError: 如果未找到模板。
    """
    templates_dir = _ensure_templates_dir(templates_dir)
    template_path = templates_dir / f"{name}.yaml"

    if not template_path.exists():
        raise FileNotFoundError(f"Workflow template '{name}' not found at {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # 确保必要字段存在
    if "name" not in data:
        data["name"] = name
    if "steps" not in data:
        data["steps"] = []

    return data


def create_template(name: str, description: str, steps: List[Dict[str, Any]]) -> Path:
    """创建工作流模板。

    参数：
        name: 模板名称。
        description: 模板描述。
        steps: 步骤定义列表。

    返回：
        创建的模板文件路径。
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
    """删除工作流模板。

    参数：
        name: 模板名称。

    返回：
        如果删除成功返回 True，如果未找到返回 False。
    """
    templates_dir = _ensure_templates_dir()
    template_path = templates_dir / f"{name}.yaml"

    if template_path.exists():
        template_path.unlink()
        return True
    return False
