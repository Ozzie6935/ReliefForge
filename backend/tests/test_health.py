"""Health endpoint and compute detection tests.

These tests must pass whether or not PyTorch is installed.
"""

import builtins
import sys
import types

from fastapi.testclient import TestClient

from app.core.compute import detect_compute
from app.main import app

client = TestClient(app)


def test_health_returns_200() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200


def test_health_status_ok() -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"


def test_health_application_name() -> None:
    body = client.get("/api/health").json()
    assert body["application"] == "CNC Relief Studio"
    assert body["api_version"] == "1"


def test_health_includes_compute() -> None:
    body = client.get("/api/health").json()
    compute = body["compute"]
    assert set(compute) == {"preferred_device", "torch_installed", "mps_available"}
    assert isinstance(compute["torch_installed"], bool)
    assert isinstance(compute["mps_available"], bool)


def test_preferred_device_is_mps_or_cpu() -> None:
    body = client.get("/api/health").json()
    assert body["compute"]["preferred_device"] in {"mps", "cpu"}


def test_detect_compute_without_torch(monkeypatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named 'torch'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    result = detect_compute()

    assert result.torch_installed is False
    assert result.mps_available is False
    assert result.preferred_device == "cpu"


def test_detect_compute_prefers_mps_when_available(monkeypatch) -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: True)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = detect_compute()

    assert result.torch_installed is True
    assert result.mps_available is True
    assert result.preferred_device == "mps"


def test_detect_compute_falls_back_to_cpu_when_mps_unavailable(monkeypatch) -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = detect_compute()

    assert result.torch_installed is True
    assert result.mps_available is False
    assert result.preferred_device == "cpu"
