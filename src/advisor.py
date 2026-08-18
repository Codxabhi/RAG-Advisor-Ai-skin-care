from __future__ import annotations

import json
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from src.config import Settings
from src.knowledge import Chunk
from src.llm import chat

DISCLAIMER = (
    "Educational cosmetic guidance only — not a diagnosis, prescription, or substitute "
    "for a dermatologist. Stop any product that burns or swells, and seek care for "
    "infections, sudden rashes, or suspected allergic reactions."
)

SYSTEM = """You are a careful cosmetic-science assistant for an app called AI Skin Care.
Use ONLY the retrieved notes plus the user's profile. If notes are missing, say so and stay conservative.
Never claim to diagnose disease. Flag pregnancy, broken skin, and severe acne as clinician territory.
Cite chunk ids like [niacinamide] when you use a fact.
Prefer short routines over long ones. Introduce one new active at a time.
Always remind the user about daily sunscreen when relevant.
"""


class RoutineStep(BaseModel):
    slot: str
    product_type: str
    why: str
    example_ingredients: list[str] = Field(default_factory=list)


class RoutinePlan(BaseModel):
    summary: str
    morning: list[RoutineStep]
    night: list[RoutineStep]
    weekly: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER


def format_context(hits: list[tuple[Chunk, float]]) -> str:
    blocks = []
    for chunk, score in hits:
        blocks.append(f"[{chunk.chunk_id}] (score={score:.3f})\n{chunk.text}")
    return "\n\n".join(blocks) if blocks else "(no retrieved notes)"


def profile_block(profile: dict[str, Any]) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in profile.items() if v)


def generate_routine(
    client: OpenAI,
    settings: Settings,
    profile: dict[str, Any],
    hits: list[tuple[Chunk, float]],
) -> RoutinePlan:
    user = f"""Create a simple AM/PM routine.

User profile:
{profile_block(profile)}

Retrieved notes:
{format_context(hits)}

Return JSON with keys:
summary, morning, night, weekly, avoid, citations, disclaimer.
morning/night arrays of objects: slot, product_type, why, example_ingredients.
citations: list of chunk ids you used.
Keep 3–5 steps per routine. No brand shopping lists.
"""
    raw = chat(
        client,
        settings,
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        json_mode=True,
    )
    try:
        data = json.loads(raw)
        return RoutinePlan.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        return RoutinePlan(
            summary=raw[:800] or "Could not parse a structured routine. Try again.",
            morning=[],
            night=[],
            citations=[c.chunk_id for c, _ in hits[:4]],
        )


def decode_ingredients(
    client: OpenAI,
    settings: Settings,
    inci_list: str,
    profile: dict[str, Any],
    hits: list[tuple[Chunk, float]],
) -> str:
    user = f"""Explain this ingredient list for a layperson. Group: helpful / potentially irritating / context-dependent.
Call out alcohols, fragrance, essential oils, and strong actives. Stay within the notes when possible.

Profile:
{profile_block(profile)}

INCI / ingredient text:
{inci_list}

Notes:
{format_context(hits)}
"""
    return chat(
        client,
        settings,
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        temperature=0.3,
    )


def answer_question(
    client: OpenAI,
    settings: Settings,
    question: str,
    profile: dict[str, Any],
    hits: list[tuple[Chunk, float]],
    image_data_url: str | None = None,
) -> str:
    text = f"""Answer the user. Cite chunk ids. If a photo is attached, comment only on visible surface clues (oiliness, dryness flakes, redness) and say you cannot diagnose.

Profile:
{profile_block(profile)}

Question:
{question}

Notes:
{format_context(hits)}
"""
    if image_data_url:
        user_content: Any = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
    else:
        user_content = text
    return chat(
        client,
        settings,
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_content}],
        temperature=0.35,
    )
