from __future__ import annotations

from typing import Any

FINAL_TYPES = {"numeric", "expression", "mcq", "string"}
VERIFIERS = {"numeric_unit", "sympy_equiv", "rule"}
INTERMEDIATE_VERIFIERS = {"numeric_unit", "sympy_equiv"}
REQUIRED_ARGS = {
    "numeric_unit": {"value"},
    "sympy_equiv": {"target"},
    "rule": {"kind", "target"},
}
AXES = {"quantitative", "symbolic"}


def validate_annotation(a: Any) -> list[str]:
    if not isinstance(a, dict):
        return ["annotation is not a dict"]
    errs: list[str] = []

    fa = a.get("final_answer")
    if not isinstance(fa, dict):
        errs.append("final_answer missing/not dict")
    else:
        ty = fa.get("type")
        if ty not in FINAL_TYPES:
            errs.append(f"final_answer.type invalid: {ty!r}")
        if not str(fa.get("value", "")).strip():
            errs.append("final_answer.value empty")
        if ty == "numeric":
            try:
                float(str(fa.get("value")).strip())
            except (TypeError, ValueError):
                errs.append(f"numeric value unparseable: {fa.get('value')!r}")
            rt = fa.get("rel_tol", 0.02)
            if not (isinstance(rt, (int, float)) and 0 < rt <= 0.2):
                errs.append(f"rel_tol out of (0, 0.2]: {rt!r}")

    if a.get("primary_axis") not in AXES:
        errs.append(f"primary_axis invalid: {a.get('primary_axis')!r}")

    crits = a.get("intermediate_criteria")
    if not isinstance(crits, list) or not (0 <= len(crits) <= 4):
        errs.append("intermediate_criteria must be a list of 0..4")
    else:
        for i, c in enumerate(crits):
            if not isinstance(c, dict):
                errs.append(f"crit[{i}] not dict")
                continue
            v = c.get("verifier")
            if v not in INTERMEDIATE_VERIFIERS:
                errs.append(f"crit[{i}] verifier invalid: {v!r}")
            else:
                args = c.get("verifier_args")
                if not isinstance(args, dict) or not REQUIRED_ARGS[v] <= set(args):
                    errs.append(f"crit[{i}] verifier_args missing {REQUIRED_ARGS[v]}")
                elif v == "numeric_unit":
                    try:
                        float(str(args["value"]).strip())
                    except (TypeError, ValueError):
                        errs.append(f"crit[{i}] numeric value unparseable")
            if c.get("axis") not in AXES:
                errs.append(f"crit[{i}] axis invalid: {c.get('axis')!r}")
            if not str(c.get("text", "")).strip():
                errs.append(f"crit[{i}] text empty")
    return errs


def build_criteria(a: dict) -> list[dict]:
    fa = a["final_answer"]
    ty = fa["type"]
    primary = a["primary_axis"]

    if ty == "numeric":
        args: dict = {"value": float(str(fa["value"]).strip()), "rel_tol": float(fa.get("rel_tol") or 0.02)}
        if fa.get("unit"):
            args["unit"] = str(fa["unit"])
        final = {
            "cid": "final", "text": "Final numeric answer matches ground truth within tolerance.",
            "verifier": "numeric_unit", "verifier_args": args, "axis": "quantitative", "weight": 1.0,
        }
    elif ty == "expression":
        final = {
            "cid": "final", "text": "Final expression is mathematically equivalent to ground truth.",
            "verifier": "sympy_equiv", "verifier_args": {"target": str(fa["value"])},
            "axis": "symbolic", "weight": 1.0,
        }
    elif ty == "mcq":
        final = {
            "cid": "final", "text": "Selected option matches the correct choice.",
            "verifier": "rule", "verifier_args": {"kind": "mcq", "target": str(fa["value"]).strip()},
            "axis": primary, "weight": 1.0,
        }
    else:
        final = {
            "cid": "final", "text": "Final answer string matches ground truth (normalized).",
            "verifier": "rule", "verifier_args": {"kind": "string", "target": str(fa["value"])},
            "axis": primary, "weight": 1.0,
        }

    out = [final]
    for i, c in enumerate(a.get("intermediate_criteria", []), start=1):
        args = dict(c["verifier_args"])
        args["scope"] = "anywhere"
        w = c.get("weight", 0.5)
        w = w if isinstance(w, (int, float)) and 0 < w <= 1 else 0.5
        out.append(
            {
                "cid": f"c{i}", "text": str(c["text"]).strip(), "verifier": c["verifier"],
                "verifier_args": args, "axis": c["axis"], "weight": float(w),
            }
        )
    return out
