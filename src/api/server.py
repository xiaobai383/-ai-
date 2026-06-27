"""FastAPI server launcher."""
import uvicorn

from src.api.app import create_api_app
from src.config import AppConfig


def launch_api(config: AppConfig | None = None, host: str = "127.0.0.1", port: int = 8000):
    """Launch the FastAPI server.

    Args:
        config: Optional AppConfig. Loads from default locations if None.
        host: Server host address.
        port: Server port number.
    """
    if config is None:
        config = AppConfig.from_yaml_and_env()

    app = create_api_app(config)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    launch_api()
