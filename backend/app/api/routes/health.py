"""Health and compute capability endpoint."""

from fastapi import APIRouter

from app.core.compute import detect_compute
from app.core.config import API_VERSION, APPLICATION_NAME
from app.schemas.health import ComputeInfo, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    capability = detect_compute()
    return HealthResponse(
        status="ok",
        application=APPLICATION_NAME,
        api_version=API_VERSION,
        compute=ComputeInfo(
            preferred_device=capability.preferred_device,
            torch_installed=capability.torch_installed,
            mps_available=capability.mps_available,
        ),
    )
