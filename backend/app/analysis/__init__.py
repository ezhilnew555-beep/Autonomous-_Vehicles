"""
Analysis module — unified road analysis, object counting, event management, and reporting.
"""
from app.analysis.object_counter import ObjectCounter, CANONICAL_CATEGORIES, canonicalize_class
from app.analysis.event_manager import EventManager, TimelineEvent
from app.analysis.analysis_state import (
    RoadStatus,
    ROAD_STATUS_DESCRIPTIONS,
    map_decision_to_plain_action,
    map_planning_status_to_label,
    determine_road_status,
)
from app.analysis.analysis_summary import AnalysisSummaryGenerator
from app.analysis.analysis_result import (
    AnalysisResult,
    AnalysisResultStore,
    analysis_store,
    VideoInfoModel,
    SafetyAnalysisModel,
    RoadInformationModel,
    ObjectCountsModel,
    PathInfoModel,
    PerformanceMetricsModel,
    SummaryModel,
    TechnicalDetailsModel,
)
from app.analysis.analysis_orchestrator import AnalysisOrchestrator

__all__ = [
    "ObjectCounter",
    "CANONICAL_CATEGORIES",
    "canonicalize_class",
    "EventManager",
    "TimelineEvent",
    "RoadStatus",
    "ROAD_STATUS_DESCRIPTIONS",
    "map_decision_to_plain_action",
    "map_planning_status_to_label",
    "determine_road_status",
    "AnalysisSummaryGenerator",
    "AnalysisResult",
    "AnalysisResultStore",
    "analysis_store",
    "ObjectCountsModel",
    "PathInfoModel",
    "PerformanceMetricsModel",
    "SummaryModel",
    "TechnicalDetailsModel",
    "AnalysisOrchestrator",
]
