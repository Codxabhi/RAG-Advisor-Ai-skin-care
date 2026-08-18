from __future__ import annotations

from openai import OpenAI

from src.config import Settings


def make_client(settings: Settings) -> OpenAI:
    kwargs: dict = {"api_key": settings.api_key or "no-key"}
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    return OpenAI(**kwargs)


def chat(
    client: OpenAI,
    settings: Settings,
    messages: list[dict],
    *,
    temperature: float = 0.4,
    json_mode: bool = False,
) -> str:
    kwargs: dict = {
        "model": settings.chat_model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


def embed_texts(client: OpenAI, settings: Settings, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=settings.embedding_model, input=texts)
    ordered = sorted(response.data, key=lambda d: d.index)
    return [item.embedding for item in ordered]
