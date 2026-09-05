"""Structured-output schema for the poverty-screening annotation task."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PovertyStatus = Literal["poor", "non_poor"]


class PovertyAnnotation(BaseModel):
    poverty_status: PovertyStatus = Field(
        description="Whether the household appears to be in relative poverty based on the note."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this classification.")
    explanation: str = Field(description="A concise, single-sentence rationale.")
