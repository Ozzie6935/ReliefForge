"""Response models for the health endpoint."""

from typing import Literal

from pydantic import BaseModel


class ComputeInfo(BaseModel):
    preferred_device: Literal["mps", "cpu"]
    torch_installed: bool
    mps_available: bool


class HealthResponse(BaseModel):
    status: Literal["ok"]
    application: str
    api_version: str
    compute: ComputeInfo
