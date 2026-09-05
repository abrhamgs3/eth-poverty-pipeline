"""Pluggable LLM client interface for the poverty-screening stage.

``MockLLMClient`` is a controlled-accuracy simulator, same design as the
sibling ``llm-sentiment-pipeline`` and ``llm-econ-pipeline`` repos: it's
handed the true poverty label and flips it at a target rate, producing a
known, reproducible error rate to demonstrate the *correction* against.
It is not a real classifier and never reads the generated household note.

A real provider client (see ``OpenAIClient``) reads only the note text -
never the true label - and is what a real rapid-screening deployment
would actually look like.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .schema import PovertyAnnotation

SYSTEM_PROMPT = (
    "You are assisting a community health program in rural Ethiopia. Based "
    "on a brief field note about a household, classify whether the "
    "household is likely in relative poverty. Categorize exactly."
)


class LLMClient(Protocol):
    def annotate(self, note: str, true_label: str | None = None) -> PovertyAnnotation:
        ...


class MockLLMClient:
    """Label-informed accuracy simulator (see module docstring)."""

    def __init__(self, accuracy: float = 0.80, rng: np.random.Generator | None = None):
        self.accuracy = accuracy
        self.rng = rng if rng is not None else np.random.default_rng()

    def annotate(self, note: str, true_label: str | None = None) -> PovertyAnnotation:
        true_label = true_label or "non_poor"
        other = "non_poor" if true_label == "poor" else "poor"

        if self.rng.random() > self.accuracy:
            status = other
            explanation = "[Simulated error] Ambiguous cues in the note misread the household's circumstances."
            confidence = round(float(self.rng.uniform(0.55, 0.85)), 2)
        else:
            status = true_label
            explanation = (
                "Note describes markers consistent with material hardship."
                if status == "poor"
                else "Note describes markers consistent with relative stability."
            )
            confidence = round(float(self.rng.uniform(0.75, 0.98)), 2)

        return PovertyAnnotation(poverty_status=status, confidence=confidence, explanation=explanation)


class OpenAIClient:
    """Reference implementation for a real structured-output API call.

    Not wired up by default - selected via ``--client openai`` plus an API
    key. Requires ``pip install openai``. Unlike ``MockLLMClient``, this
    never sees ``true_label`` - a real classifier doesn't get the answer.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only when selected
            raise ImportError("OpenAIClient requires: pip install openai") from exc
        self.model = model
        self._client = OpenAI(api_key=api_key)

    def annotate(self, note: str, true_label: str | None = None) -> PovertyAnnotation:
        response = self._client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Field note:\n\n{note}"},
            ],
            response_format=PovertyAnnotation,
            temperature=0.2,
        )
        return response.choices[0].message.parsed
