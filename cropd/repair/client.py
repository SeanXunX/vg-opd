from __future__ import annotations

import os

from cropd.repair.prompts import repair_messages

_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI

        _CLIENT = OpenAI(
            base_url=os.environ.get("CROPD_REPAIR_BASE_URL", "http://localhost:8001/v1"),
            api_key="EMPTY",
            timeout=float(os.environ.get("CROPD_REPAIR_TIMEOUT", "120")),
            max_retries=0,
        )
    return _CLIENT


def repair(question: str, y: str, criterion: dict, expert: str, max_tokens: int = 3072) -> str | None:
    try:
        resp = _client().chat.completions.create(
            model=expert,
            temperature=0.0,
            max_tokens=max_tokens,
            messages=repair_messages(question, y, criterion),
        )
        text = resp.choices[0].message.content
        return text.strip() if text and text.strip() else None
    except Exception:
        return None


def continue_from(question: str, prefix: str, expert: str, max_tokens: int = 2048) -> str | None:
    try:
        resp = _client().chat.completions.create(
            model=expert,
            temperature=0.0,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": question}, {"role": "assistant", "content": prefix}],
            extra_body={"add_generation_prompt": False, "continue_final_message": True},
        )
        return resp.choices[0].message.content
    except Exception:
        return None
