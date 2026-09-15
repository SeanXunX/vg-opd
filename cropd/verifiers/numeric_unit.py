from __future__ import annotations

import math
import re

from . import Unknown, VerifierResult, extract_boxed, find_numbers, normalize_sci

_UREG = None


def _ureg():
    global _UREG
    if _UREG is None:
        import pint

        _UREG = pint.UnitRegistry()
    return _UREG


def _target_value(raw) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    nums = find_numbers(str(raw))
    if len(nums) != 1:
        raise Unknown(f"target not a single number: {raw!r}")
    return nums[0]


def _close(a: float, b: float, rel: float, abs_tol: float) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


_UNIT_RE = re.compile(r"^\s*(?:\\?,|\\ )?\s*\\?(?:text|mathrm|si|,)?\{?\s*([a-zA-Z\u03bc\u03a9\u00b0%][a-zA-Z\u03bc\u03a9\u00b0%\d\s.^{}/*\u00b7\\-]*?)\s*\}?\s*$")


def _try_convert(num: float, unit_str: str, target_unit: str) -> float | None:
    try:
        u = _ureg()
        src = unit_str.replace("\\", "").replace("\u00b7", "*").replace("^", "**").strip()
        q = num * u.parse_expression(src)
        return q.to(u.parse_expression(target_unit)).magnitude
    except Exception:
        return None


def verify(response: str, args: dict) -> VerifierResult:
    target = _target_value(args["value"])
    rel = float(args.get("rel_tol", 0.02))
    abs_tol = float(args.get("abs_tol", 1e-9 if target == 0 else 0.0))
    unit = args.get("unit") or None
    scope = args.get("scope", "final")

    if scope == "anywhere":
        nums = find_numbers(response)
        if not nums:
            return VerifierResult(False, 0.0, "no numbers in response")
        ok = any(_close(n, target, rel, abs_tol) for n in nums)
        return VerifierResult(ok, float(ok), f"anywhere: {len(nums)} candidates")

    boxed = extract_boxed(response)
    seg = boxed if boxed is not None else response
    nums = find_numbers(seg)
    if not nums:
        return VerifierResult(False, 0.0, "no number in final segment")
    cands = nums if boxed is not None else [nums[-1]]
    cand = cands[-1]
    for c in cands:
        if _close(c, target, rel, abs_tol):
            return VerifierResult(True, 1.0, f"match {c} ~ {target}")

    if unit and boxed is not None:
        m = normalize_sci(boxed)
        tail = m[m.rfind(str(cand) if str(cand) in m else "") :]
        um = _UNIT_RE.match(tail[len(str(cand)) :]) if tail else None
        if um:
            conv = _try_convert(cand, um.group(1), unit)
            if conv is not None and _close(conv, target, rel, abs_tol):
                return VerifierResult(True, 1.0, f"unit-converted {cand} {um.group(1)} -> {conv} {unit}")
    return VerifierResult(False, 0.0, f"mismatch {cand} vs {target}")
