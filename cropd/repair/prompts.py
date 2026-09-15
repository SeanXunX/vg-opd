REPAIR_SYSTEM = """You are a precise solution repairer. You will be given a problem, a model's solution attempt, and ONE specific criterion the attempt fails.

Rewrite the solution changing AS LITTLE AS POSSIBLE so that the criterion is satisfied:
- Fix ONLY what is necessary for the given criterion; copy all other text verbatim, character for character.
- Do NOT restructure, reorder, expand, or polish anything else.
- Keep the same format, including the final \\boxed{...} line (update its content only if the criterion requires it).
- Output ONLY the full repaired solution text (no commentary, no diff markers)."""

REPAIR_USER_TMPL = """## Problem
{question}

## Solution attempt
{y}

## Failed criterion (fix exactly this)
{criterion_text}
(machine check: {verifier} {verifier_args})"""


def repair_messages(question: str, y: str, criterion: dict) -> list[dict]:
    import json

    return [
        {"role": "system", "content": REPAIR_SYSTEM},
        {"role": "user", "content": REPAIR_USER_TMPL.format(
            question=question, y=y, criterion_text=criterion.get("text", ""),
            verifier=criterion.get("verifier"),
            verifier_args=json.dumps(criterion.get("verifier_args", {}), ensure_ascii=False)[:300],
        )},
    ]
