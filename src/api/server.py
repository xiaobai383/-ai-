"""FastAPI 服务器启动器。"""
import uvicorn

from src.api.app import create_api_app
from src.config import AppConfig


def launch_api(config: AppConfig | None = None, host: str | None = None, port: int | None = None):
    """启动 FastAPI 服务器。

    Args:
        config: 可选的 AppConfig。如果为 None，则从默认位置加载。
        host: 服务器主机地址。
        port: 服务器端口号。
    """
    if config is None:
        config = AppConfig.from_yaml_and_env()

    app = create_api_app(config)
    uvicorn.run(app, host=host or config.api_host, port=port or config.api_port)


if __name__ == "__main__":
    launch_api(host=None, port=None)
