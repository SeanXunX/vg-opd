from __future__ import annotations

import re
import time

from . import Unknown, VerifierResult, extract_boxed


def _parse(s: str):
    from math_verify import parse

    got = parse(f"${s}$") + parse(s)
    if not got:
        raise Unknown(f"unparseable: {s[:80]!r}")
    return got


def _equiv(gold, cand) -> bool:
    from math_verify import verify as mv_verify

    return bool(mv_verify(gold, cand))


_MATH_SEG_RE = re.compile(
    r"\$\$(.+?)\$\$|\$(.+?)\$|\\\[(.+?)\\\]|\\\((.+?)\\\)", re.DOTALL
)


_WORD_RE = re.compile(r"^[a-z]+$|^[A-Z][a-z]+$")
_CLAUSE_SPLIT_RE = re.compile(r"[,;:]|(?<=[a-zA-Z])\.\s")


def _eq_neighborhood(clause: str) -> str | None:
    toks = clause.split()
    idx = next((i for i, t in enumerate(toks) if "=" in t), None)
    if idx is None:
        return None
    lo = idx
    while lo > 0 and not _WORD_RE.match(toks[lo - 1]):
        lo -= 1
    hi = idx
    while hi + 1 < len(toks) and not _WORD_RE.match(toks[hi + 1]):
        hi += 1
    seg = " ".join(toks[lo : hi + 1])
    return seg if len(seg) >= 3 else None


def _candidates_anywhere(response: str, limit: int = 60) -> list[str]:
    cands: list[str] = []
    for m in _MATH_SEG_RE.finditer(response):
        seg = next(g for g in m.groups() if g is not None).strip()
        if seg:
            cands.append(seg)
    for line in response.splitlines():
        line = line.strip()
        if "=" not in line or not (3 <= len(line) <= 400) or line.startswith(("#", "|")):
            continue
        cands.append(line[:200])
        for clause in _CLAUSE_SPLIT_RE.split(line):
            clause = clause.strip()
            if "=" not in clause or len(clause) < 3:
                continue
            cands.append(clause)
            toks = clause.split()
            i = 0
            while i < len(toks) and "=" not in toks[i] and _WORD_RE.match(toks[i]):
                i += 1
            if 0 < i < len(toks):
                cands.append(" ".join(toks[i:]))
            neigh = _eq_neighborhood(clause)
            if neigh:
                cands.append(neigh)
    boxed = extract_boxed(response)
    if boxed:
        cands.append(boxed)
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:limit]


_DISPLAY_RE = re.compile(r"^\s*(?:\\\[|\\\(|\$\$?)\s*|\s*(?:\\\]|\\\)|\$\$?)\s*$")
_TRAIL_UNIT_RE = re.compile(r"(?:\\,|\\;|\\ |~)\s*(?:\\(?:text|mathrm)\{[^{}]*\}|\\[A-Za-z]+)\s*$")
_LETTERS_RE = re.compile(r"^[A-Za-z]{2,6}$")
_GREEK = {
    "\u03b1": r"\alpha", "\u03b2": r"\beta", "\u03b3": r"\gamma", "\u03b4": r"\delta", "\u03b5": r"\epsilon",
    "\u03b7": r"\eta", "\u03b8": r"\theta", "\u03ba": r"\kappa", "\u03bb": r"\lambda", "\u03bc": r"\mu",
    "\u03bd": r"\nu", "\u03c0": r"\pi", "\u03c1": r"\rho", "\u03c3": r"\sigma", "\u03c4": r"\tau",
    "\u03c6": r"\phi", "\u03c7": r"\chi", "\u03c8": r"\psi", "\u03c9": r"\omega",
    "\u0394": r"\Delta", "\u03a9": r"\Omega", "\u0393": r"\Gamma", "\u039b": r"\Lambda", "\u03a6": r"\Phi",
    "\u0127": r"\hbar", "\u222b": r"\int ", "\u221a": r"\sqrt", "\u221e": r"\infty",
}


def _variants(s: str) -> list[str]:
    out: list[str] = []

    def add(x: str) -> None:
        x = x.strip()
        if x and x not in out:
            out.append(x)

    if re.search(r"\\\\[a-zA-Z([]", s):
        s = s.replace("\\\\", "\\")
    s = s.replace("$", "")
    if "\\boxed{" in s:
        inner = extract_boxed(s)
        if inner:
            s = inner
    if any(ch in s for ch in _GREEK):
        for ch, cmd in _GREEK.items():
            s = s.replace(ch, cmd + " ")
        s = re.sub(r"(\\[a-zA-Z]+)\s+", r"\1 ", s)
    s = re.sub(r"\^\(([^()]+)\)", r"^{\1}", s)
    s = s.replace("*", " ")

    add(s)
    stripped = _DISPLAY_RE.sub("", s.strip())
    stripped = _DISPLAY_RE.sub("", stripped)
    add(stripped)
    for base in list(out):
        add(re.sub(r"\\(?:text|mathrm)\{[^{}]*\}", " ", base))
    for base in list(out):
        if "=" in base:
            add(base.rsplit("=", 1)[1])
    for base in list(out):
        add(_TRAIL_UNIT_RE.sub("", base))
    for base in list(out):
        if _LETTERS_RE.match(base):
            add(" ".join(base))
    return out[:12]


def _any_equiv(target: str, candidate: str, deadline: float) -> bool:
    tvs, cvs = _variants(target), _variants(candidate)

    from .rule import _norm

    tns = {_norm(tv) for tv in tvs if len(_norm(tv)) >= 3}
    if any(_norm(cv) in tns for cv in cvs):
        return True

    golds = []
    for tv in tvs:
        try:
            golds.append(_parse(tv))
        except Exception:
            continue
    if not golds:
        raise Unknown(f"target unparseable in all variants: {target[:80]!r}")
    for cv in cvs:
        if time.monotonic() > deadline:
            return False
        try:
            c = _parse(cv)
        except Exception:
            continue
        for g in golds:
            if _equiv(g, c):
                return True
    return False


def verify(response: str, args: dict) -> VerifierResult:
    target = str(args["target"])
    scope = args.get("scope", "final")
    budget = float(args.get("timeout", 8.0))
    deadline = time.monotonic() + budget

    if scope == "anywhere":
        for cand in _candidates_anywhere(response):
            if time.monotonic() > deadline:
                return VerifierResult(None, None, "budget exhausted (anywhere scan)")
            try:
                if _any_equiv(target, cand, deadline):
                    return VerifierResult(True, 1.0, f"equiv at: {cand[:60]!r}")
            except Exception:
                continue
        return VerifierResult(False, 0.0, "no equivalent segment found")

    boxed = extract_boxed(response)
    if boxed is None:
        return VerifierResult(False, 0.0, "no \\boxed final answer")
    ok = _any_equiv(target, boxed, deadline)
    return VerifierResult(ok, float(ok), f"final {boxed[:60]!r} vs {target[:60]!r}")
