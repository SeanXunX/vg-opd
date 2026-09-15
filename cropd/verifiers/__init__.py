from __future__ import annotations

import re
import signal
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class VerifierResult:
    passed: bool | None
    score: float | None
    detail: str = ""


class Unknown(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    def handler(signum, frame):
        raise TimeoutError

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def strip_think(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        return ""
    return text


def extract_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    i, depth = idx + len("\\boxed{"), 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None


_SUB = str.maketrans("\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089", "0123456789")
_SUP = {"\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3", "\u2074": "4", "\u2075": "5", "\u2076": "6", "\u2077": "7", "\u2078": "8", "\u2079": "9", "\u207b": "-", "\u207a": "+"}


def fold_unicode_math(s: str) -> str:
    s = s.translate(_SUB)
    for k, v in _SUP.items():
        s = s.replace(k, v)
    return s


def _sup_to_ascii(s: str) -> str:
    return "".join(_SUP.get(c, c) for c in s)


def normalize_sci(s: str) -> str:
    s = re.sub(
        r"(\d(?:\.\d+)?)\s*[\u00d7x\*]\s*10\s*([\u207b\u207a]?[\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079]+)",
        lambda m: m.group(1) + "e" + _sup_to_ascii(m.group(2)),
        s,
    )
    s = re.sub(r"(\d(?:\.\d+)?)\s*(?:[\u00d7x\*]|\\times)\s*10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?", r"\1e\2", s)
    s = re.sub(r"(?<![\d.eE])10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?", r"1e\1", s)
    s = fold_unicode_math(s)
    s = re.sub(r"(\d),(?=\d{3}(\D|$))", r"\1", s)
    return s


NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def find_numbers(s: str) -> list[float]:
    return [float(m.group()) for m in NUM_RE.finditer(normalize_sci(s))]


def run_verifier(name: str, response: str, args: dict) -> VerifierResult:
    from . import llm_judge, numeric_unit, rule, sympy_equiv

    registry = {
        "numeric_unit": numeric_unit.verify,
        "sympy_equiv": sympy_equiv.verify,
        "rule": rule.verify,
        "llm_judge": llm_judge.verify,
    }
    if name not in registry:
        return VerifierResult(None, None, f"unknown verifier: {name}")
    try:
        return registry[name](response, args)
    except (Unknown, TimeoutError) as e:
        return VerifierResult(None, None, f"unknown: {e}")
    except Exception as e:
        return VerifierResult(None, None, f"error: {type(e).__name__}: {e}")
