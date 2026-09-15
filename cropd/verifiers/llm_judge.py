from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from . import Unknown, VerifierResult

PROMPT_VER = "judge_v0"
_CACHE_DIR = Path(os.environ.get("CROPD_JUDGE_CACHE", "data/cache/llm_judge"))
_LOW_CONF = (0.35, 0.65)
_VOTE_N = 3

_SYSTEM = (
    "You are a strict scientific grading assistant. Judge ONLY the single criterion given, "
    "against the student answer. Output STRICT JSON on one line: "
    '{"pass": true|false, "score": <0..1>, "evidence": "<shortest quote or reason>"} '
    "No other text. score reflects degree of satisfaction; pass = (criterion clearly satisfied)."
)

_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI

        _CLIENT = OpenAI(
            base_url=os.environ.get("CROPD_JUDGE_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("CROPD_JUDGE_API_KEY", "EMPTY"),
            timeout=float(os.environ.get("CROPD_JUDGE_TIMEOUT", "20")),
            max_retries=0,
        )
    return _CLIENT


def judge_version() -> str:
    return os.environ.get("CROPD_JUDGE_MODEL", "judge")


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_verdict(text: str) -> dict | None:
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group())
    except ValueError:
        return None
    if not isinstance(d.get("pass"), bool):
        return None
    try:
        d["score"] = min(1.0, max(0.0, float(d.get("score", float(d["pass"])))))
    except (TypeError, ValueError):
        d["score"] = float(d["pass"])
    return d


def _ask(criterion: str, answer: str, temperature: float) -> dict | None:
    resp = _client().chat.completions.create(
        model=judge_version(),
        temperature=temperature,
        max_tokens=300,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Criterion: {criterion}\n\nStudent answer:\n{answer}\n\nJSON verdict:"},
        ],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return _parse_verdict(resp.choices[0].message.content)


def _cache_key(cid: str, answer: str) -> Path:
    h = hashlib.sha256(
        "|".join([judge_version(), PROMPT_VER, cid, hashlib.sha256(answer.encode()).hexdigest()]).encode()
    ).hexdigest()
    return _CACHE_DIR / f"{h}.json"


def verify(response: str, args: dict) -> VerifierResult:
    criterion = str(args.get("criterion") or args.get("rubric") or "").strip()
    if not criterion:
        raise Unknown("empty criterion")
    cid = str(args.get("cid", "c?"))

    ck = _cache_key(cid, response)
    if ck.exists():
        d = json.loads(ck.read_text())
        return VerifierResult(d["pass"], d["score"], d.get("detail", "") + " (cached)")

    try:
        v = _ask(criterion, response, temperature=0.0) or _ask(criterion, response, temperature=0.2)
        if v is None:
            raise Unknown("unparseable verdict twice")
        if _LOW_CONF[0] < v["score"] < _LOW_CONF[1]:
            votes = [v] + [x for t in (0.5, 0.5) if (x := _ask(criterion, response, temperature=t))]
            n_pass = sum(x["pass"] for x in votes)
            v = {
                "pass": n_pass * 2 > len(votes),
                "score": sum(x["score"] for x in votes) / len(votes),
                "evidence": votes[0].get("evidence", ""),
            }
            v["vote"] = f"{n_pass}/{len(votes)}"
    except Unknown:
        raise
    except Exception as e:
        raise Unknown(f"judge call failed: {type(e).__name__}") from e

    detail = f"{judge_version()}@{PROMPT_VER}" + (f" vote={v['vote']}" if "vote" in v else "")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ck.write_text(json.dumps({"pass": v["pass"], "score": v["score"], "detail": detail}, ensure_ascii=False))
    return VerifierResult(v["pass"], v["score"], detail)
