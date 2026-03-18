"""Endpoints for collecting manual Re-ID/detection feedback from the UI."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.infrastructure.database.base import get_db
from app.infrastructure.database.repositories.tracking_repository import TrackingRepository


router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackDecision(str, Enum):
    """Thumbs up/down decisions supported by the frontend UI."""

    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"


class ReIDFeedbackRequest(BaseModel):
    """Payload describing manual verification of a re-identification match."""

    global_person_id: str = Field(..., description="Primary SpotOn global person identifier.")
    candidate_person_id: Optional[str] = Field(
        None,
        description="Matched/global ID that is being compared against the selected person.",
    )
    match_id: Optional[str] = Field(
        None,
        description="Identifier for the match event shown to the user (if tracked by the UI).",
    )
    environment_id: str = Field(..., description="Environment identifier associated with the cameras.")
    camera_id: str = Field(..., description="Camera identifier where the user observed the person.")
    frame_number: Optional[int] = Field(
        None, ge=0, description="Frame number for the rendered bounding box, if applicable."
    )
    decision: FeedbackDecision = Field(
        ..., description="Frontend thumb-up or thumb-down decision from the user."
    )
    event_timestamp: Optional[datetime] = Field(
        None,
        description="Timestamp when the match occurred. Defaults to the receipt time if omitted.",
    )
    session_id: Optional[str] = Field(
        None, description="Playback or analytics session identifier for traceability."
    )
    source: Optional[str] = Field(
        None,
        description="Source of the feedback (front-end client identifier) if different from default.",
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional UI-provided confidence to prioritize review queues.",
    )
    notes: Optional[str] = Field(None, description="Optional free-form operator notes.")
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional structured context (bounding box IDs, UI state, etc.).",
    )


class ReIDFeedbackResponse(BaseModel):
    """Acknowledgement returned once feedback is persisted."""

    feedback_id: UUID
    global_person_id: str
    candidate_person_id: Optional[str]
    match_id: Optional[str]
    decision: FeedbackDecision
    recorded_at: datetime
    metadata: Dict[str, Any]


class ReIDFeedbackRecord(BaseModel):
    """Single feedback entry returned from the GET endpoint."""

    feedback_id: UUID
    global_person_id: str
    candidate_person_id: Optional[str]
    match_id: Optional[str]
    decision: FeedbackDecision
    camera_id: str
    environment_id: str
    frame_number: Optional[int]
    session_id: Optional[str]
    event_timestamp: datetime
    recorded_at: datetime
    source: Optional[str]
    confidence: Optional[float]
    notes: Optional[str]
    metadata: Dict[str, Any]


class ReIDFeedbackListResponse(BaseModel):
    """Paginated list of feedback entries."""

    total: int
    items: List[ReIDFeedbackRecord]


def get_tracking_repo(db: Session = Depends(get_db)) -> TrackingRepository:
    """FastAPI dependency providing a tracking repository instance."""

    return TrackingRepository(db)


@router.post(
    "/reidentification",
    response_model=ReIDFeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record thumbs up/down feedback for a re-identification match",
)
async def submit_reidentification_feedback(
    payload: ReIDFeedbackRequest,
    repo: TrackingRepository = Depends(get_tracking_repo),
) -> ReIDFeedbackResponse:
    """Persist re-identification feedback from the frontend UI."""

    event = await repo.record_reidentification_feedback(
        global_person_id=payload.global_person_id,
        candidate_person_id=payload.candidate_person_id,
        match_id=payload.match_id,
        camera_id=payload.camera_id,
        environment_id=payload.environment_id,
        frame_number=payload.frame_number,
        event_timestamp=payload.event_timestamp,
        session_id=payload.session_id,
        decision=payload.decision.value,
        source=payload.source,
        confidence=payload.confidence,
        notes=payload.notes,
        metadata=payload.metadata,
    )

    return ReIDFeedbackResponse(
        feedback_id=event.id,
        global_person_id=event.global_person_id,
        candidate_person_id=event.candidate_person_id,
        match_id=event.match_id,
        decision=FeedbackDecision(event.decision),
        recorded_at=event.created_at,
        metadata=event.feedback_metadata or {},
    )


@router.get(
    "/reidentification",
    response_model=ReIDFeedbackListResponse,
    summary="List recorded re-identification feedback events",
)
async def list_reidentification_feedback(
    global_person_id: Optional[str] = Query(None, description="Filter by SpotOn global ID."),
    candidate_person_id: Optional[str] = Query(None, description="Filter by candidate global ID."),
    match_id: Optional[str] = Query(None, description="Filter by specific match identifier."),
    camera_id: Optional[str] = Query(None, description="Restrict to a specific camera."),
    environment_id: Optional[str] = Query(None, description="Restrict to an environment."),
    session_id: Optional[str] = Query(None, description="Restrict to a playback session."),
    decision: Optional[FeedbackDecision] = Query(None, description="thumbs_up or thumbs_down."),
    start_time: Optional[datetime] = Query(None, description="Return feedback recorded after this timestamp."),
    end_time: Optional[datetime] = Query(None, description="Return feedback recorded before this timestamp."),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of records to return."),
    offset: int = Query(0, ge=0, description="Number of records to skip for pagination."),
    repo: TrackingRepository = Depends(get_tracking_repo),
) -> ReIDFeedbackListResponse:
    """Return a filtered, paginated list of feedback events for UI dashboards."""

    events, total = await repo.list_reidentification_feedback(
        global_person_id=global_person_id,
        candidate_person_id=candidate_person_id,
        match_id=match_id,
        camera_id=camera_id,
        environment_id=environment_id,
        session_id=session_id,
        decision=decision.value if decision else None,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset,
    )

    items = [
        ReIDFeedbackRecord(
            feedback_id=event.id,
            global_person_id=event.global_person_id,
            candidate_person_id=event.candidate_person_id,
            match_id=event.match_id,
            decision=FeedbackDecision(event.decision),
            camera_id=event.camera_id,
            environment_id=event.environment_id,
            frame_number=event.frame_number,
            session_id=event.session_id,
            event_timestamp=event.event_timestamp,
            recorded_at=event.created_at,
            source=event.feedback_source,
            confidence=event.confidence,
            notes=event.notes,
            metadata=event.feedback_metadata or {},
        )
        for event in events
    ]

    return ReIDFeedbackListResponse(total=total, items=items)
