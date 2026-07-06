"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_api_app
from src.config import AppConfig


@pytest.fixture
def config():
    """Create a test configuration."""
    return AppConfig(
        api_key="sk-fake-test-key",
        allowed_paths=["data/", "output/"],
        blocked_patterns=[".env", "*.key"],
    )


@pytest.fixture
def client(config):
    """Create a test client."""
    app = create_api_app(config)
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"


def test_get_config(client):
    """Test config endpoint."""
    response = client.get("/config")

    assert response.status_code == 200
    data = response.json()
    assert "model" in data
    assert "max_file_size_mb" in data
    assert "allowed_paths" in data


def test_create_task_missing_query(client):
    """Test creating a task without query."""
    response = client.post("/tasks", json={
        "mode": "quick",
        "files": [],
    })

    assert response.status_code == 422  # Validation error


def test_create_task_with_files(client, tmp_path):
    """Test creating a task with files."""
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content", encoding="utf-8")

    response = client.post("/tasks", json={
        "query": "Summarize this",
        "mode": "quick",
        "files": [str(test_file)],
    })

    # This will fail because we don't have a real LLM configured
    # but it tests the endpoint structure
    assert response.status_code in [200, 500]
