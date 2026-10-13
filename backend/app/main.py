"""CNC Relief Studio API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.depth import router as depth_router
from app.api.routes.health import router as health_router
from app.api.routes.projects import router as projects_router
from app.api.routes.relief import router as relief_router
from app.api.routes.segmentation import router as segmentation_router
from app.core.config import API_VERSION, APPLICATION_NAME, CORS_ORIGINS

app = FastAPI(title=APPLICATION_NAME, version=API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Accept", "Content-Type"],
    expose_headers=[
        "Content-Disposition",
        "X-Relief-Stl-Units",
        "X-Relief-Export-Certificate",
        "X-Relief-Self-Intersection-Checked",
        "X-Relief-Topology-Limitation",
    ],
)

app.include_router(health_router, prefix="/api")
app.include_router(projects_router, prefix="/api")
app.include_router(depth_router, prefix="/api")
app.include_router(segmentation_router, prefix="/api")
app.include_router(relief_router, prefix="/api")
