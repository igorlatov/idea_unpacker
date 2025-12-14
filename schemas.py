"""
Data schemas for the Idea Unpacker flow.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class UserInput(BaseModel):
    topic: str = Field(..., min_length=1, max_length=50)
    intent: str = Field(..., min_length=1, max_length=200)


class Idea(BaseModel):
    name: str
    description: str
    why_underexplored: str
    source: str  # Author name or "model-generated"
    is_model_generated: bool = False


class ScoredIdea(BaseModel):
    idea: Idea
    score_1: float = Field(..., ge=1, le=10)
    score_2: float = Field(..., ge=1, le=10)
    rationale_1: str
    rationale_2: str
    score_delta: float = 0
    combined_score: float = 0


class OutputFormat(str, Enum):
    POEM = "poem"
    QUOTES = "quotes"
    MICRO_ESSAY = "micro_essay"
    APHORISMS = "aphorisms"
    DIALOGUE = "dialogue"


class FormatSpec(BaseModel):
    format_type: OutputFormat
    rationale: str
    criteria: list[str] = Field(..., min_length=3, max_length=5)
    minimum_bar: float = Field(..., ge=1, le=10)


class Draft(BaseModel):
    content: str
    explainer: str
    word_count: int
    version: int


class Evaluation(BaseModel):
    scores: dict[str, float]  # criterion -> score
    total_score: float
    feedback: list[str] = Field(..., max_length=3)
    plateau_detected: bool = False


class FlowResult(BaseModel):
    success: bool
    final_draft: Optional[Draft] = None
    final_score: float = 0
    cycles_used: int = 0
    failure_reason: Optional[str] = None
    provenance: list[dict] = []
