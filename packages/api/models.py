from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class MinerSubmission(BaseModel):
    submission_id: str
    miner_id: str
    processing_date: str
    window_days: int
    alert_id: str
    score: float = Field(ge=0.0, le=1.0)
    model_version: str
    model_github_url: str
    score_metadata: Optional[str] = None


class BatchSubmission(BaseModel):
    submissions: List[MinerSubmission]


class ValidationScore(BaseModel):
    miner_id: str
    processing_date: str
    window_days: int
    tier1_integrity_score: float
    tier2_behavior_score: float
    tier3_gt_score: Optional[float] = None
    tier3_evolution_score: Optional[float] = None
    final_score: float
    validation_status: str
    validated_at: datetime