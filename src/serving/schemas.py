"""Pydantic request/response models for the recommendation API."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: str
    n_recommendations: int = Field(default=10, ge=1, le=50)
    context: Optional[dict[str, Any]] = None


class RecommendedItem(BaseModel):
    item_id: str
    title: str
    category: str
    price: float
    avg_rating: float
    score: float


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: list[RecommendedItem]
    retrieval_ms: float
    ranking_ms: float
    total_ms: float
    model_version: str


class FeedbackRequest(BaseModel):
    user_id: str
    item_id: str
    event_type: str  # "click" | "purchase" | "add_to_cart"
    session_id: Optional[str] = None
    context: Optional[dict[str, Any]] = None
