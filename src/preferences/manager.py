"""用户偏好管理 — 存储和读取用户设置。"""
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class UserPreferences:
    """用户偏好设置。"""

    # 默认处理模式
    default_mode: str = "privacy_enhanced"

    # 输出格式偏好
    default_output_format: str = "markdown"
    include_metadata: bool = True
    include_timestamp: bool = True

    # 界面偏好
    show_token_count: bool = True
    show_cost_estimate: bool = True
    auto_preview: bool = True

    # 工作流偏好
    favorite_templates: list = field(default_factory=list)
    recent_templates: list = field(default_factory=list)

    # 通知偏好
    notify_on_completion: bool = True
    notify_on_cost_threshold: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreferences":
        """从字典创建，忽略未知键。"""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


class PreferenceManager:
    """管理用户偏好，基于文件持久化。"""

    def __init__(self, preferences_dir: str | Path | None = None):
        """初始化偏好管理器。

        Args:
            preferences_dir: 存储偏好的目录。默认为 'data/preferences'。
        """
        if preferences_dir is None:
            preferences_dir = Path("data/preferences")

        self.preferences_dir = Path(preferences_dir)
        self.preferences_dir.mkdir(parents=True, exist_ok=True)

        self._preferences_file = self.preferences_dir / "user_preferences.json"
        self._cache: Optional[UserPreferences] = None

    def load(self) -> UserPreferences:
        """从文件加载用户偏好。

        Returns:
            UserPreferences 实例。
        """
        if self._cache is not None:
            return self._cache

        if self._preferences_file.exists():
            try:
                with open(self._preferences_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._cache = UserPreferences.from_dict(data)
            except Exception:
                self._cache = UserPreferences()
        else:
            self._cache = UserPreferences()

        return self._cache

    def save(self, preferences: UserPreferences) -> None:
        """保存用户偏好到文件。

        Args:
            preferences: 要保存的 UserPreferences。
        """
        self._cache = preferences

        with open(self._preferences_file, "w", encoding="utf-8") as f:
            json.dump(preferences.to_dict(), f, ensure_ascii=False, indent=2)

    def update(self, **kwargs) -> UserPreferences:
        """更新特定偏好字段。

        Args:
            **kwargs: 要更新的字段。

        Returns:
            更新后的 UserPreferences。
        """
        preferences = self.load()

        for key, value in kwargs.items():
            if hasattr(preferences, key):
                setattr(preferences, key, value)

        self.save(preferences)
        return preferences

    def add_recent_template(self, template_name: str, max_recent: int = 10) -> None:
        """将模板添加到最近使用列表。

        Args:
            template_name: 模板名称。
            max_recent: 保留的最近模板最大数量。
        """
        preferences = self.load()

        # 如果已存在则移除
        if template_name in preferences.recent_templates:
            preferences.recent_templates.remove(template_name)

        # 添加到最前面
        preferences.recent_templates.insert(0, template_name)

        # 裁剪到最大数量
        preferences.recent_templates = preferences.recent_templates[:max_recent]

        self.save(preferences)

    def toggle_favorite_template(self, template_name: str) -> bool:
        """切换模板的收藏状态。

        Args:
            template_name: 模板名称。

        Returns:
            如果添加到收藏则返回 True，如果移除则返回 False。
        """
        preferences = self.load()

        if template_name in preferences.favorite_templates:
            preferences.favorite_templates.remove(template_name)
            self.save(preferences)
            return False
        else:
            preferences.favorite_templates.append(template_name)
            self.save(preferences)
            return True

    def reset(self) -> UserPreferences:
        """重置偏好为默认值。

        Returns:
            默认的 UserPreferences。
        """
        default = UserPreferences()
        self.save(default)
        return default
