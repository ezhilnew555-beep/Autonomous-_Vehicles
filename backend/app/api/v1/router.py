"""API v1 router — aggregates all versioned endpoints."""
from fastapi import APIRouter
from app.api.v1.endpoints.video import router as video_router
from app.api.v1.endpoints.evaluation import router as evaluation_router
from app.api.v1.endpoints.dashboard import router as dashboard_router
from app.api.v1.endpoints.scenarios import router as scenarios_router
from app.api.v1.endpoints.analysis import router as analysis_router

api_v1_router = APIRouter()
api_v1_router.include_router(video_router, prefix="/video", tags=["Video"])
api_v1_router.include_router(evaluation_router, prefix="", tags=["Evaluation"])
api_v1_router.include_router(dashboard_router, prefix="/dashboard", tags=["Dashboard"])
api_v1_router.include_router(scenarios_router, prefix="/scenarios", tags=["Scenarios"])
api_v1_router.include_router(analysis_router, prefix="/analysis", tags=["Analysis"])


