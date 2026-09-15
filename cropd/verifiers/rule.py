from __future__ import annotations

import re

from . import VerifierResult, extract_boxed, fold_unicode_math

_MCQ_RE = re.compile(r"(?<![A-Za-z])([A-J])(?![A-Za-z])")


def _norm(s: str) -> str:
    if re.search(r"\\\\[a-zA-Z([]", s):
        s = s.replace("\\\\", "\\")
    s = fold_unicode_math(s)
    s = s.replace("\u21cc", "<=>").replace("\u2192", "->").replace("\u2190", "<-")
    s = re.sub(r"[\u2013\u2014\u2212]", "-", s)
    s = re.sub(r"\\[()\[\]]", "", s)
    s = re.sub(r"\\(?:begin|end)\{[a-zA-Z*]+\}", "", s)
    s = re.sub(r"\\boxed", "", s)
    s = re.sub(r"\\(?:text|mathrm|ce|mathbf|mathit)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[,;!:]|\\\\|\\ ", "", s)
    s = re.sub(r"[\s\u00a0]+", "", s)
    s = re.sub(r"[,,;;..!!??'\"\u201c\u201d\u2018\u2019`&$*{}]", "", s)
    return s.lower()


def _string_match(cand: str, target: str) -> bool:
    cn, tn = _norm(cand), _norm(target)
    if not tn:
        return False
    if cn == tn or tn in cn or (cn and cn in tn):
        return True
    cn2, tn2 = cn.replace("-", ""), tn.replace("-", "")
    return bool(tn2) and (cn2 == tn2 or tn2 in cn2 or (cn2 and cn2 in tn2))


def verify(response: str, args: dict) -> VerifierResult:
    kind = args.get("kind", "string")
    target = str(args["target"])

    if kind == "mcq":
        seg = extract_boxed(response) or response
        letters = _MCQ_RE.findall(seg.upper())
        if not letters:
            return VerifierResult(False, 0.0, "no option letter found")
        ok = letters[-1] == target.strip().upper()
        return VerifierResult(ok, float(ok), f"picked {letters[-1]} vs {target}")

    tn = _norm(target)
    if not tn:
        return VerifierResult(None, None, "empty target after normalization")
    if kind == "contains":
        ok = tn in _norm(response)
        return VerifierResult(ok, float(ok), f"contains({target[:40]!r})={ok}")

    seg = extract_boxed(response)
    cand = seg if seg is not None else response
    ok = _string_match(cand, target)
    return VerifierResult(ok, float(ok), f"string vs {target[:40]!r}")
