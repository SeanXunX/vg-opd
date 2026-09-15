PROMPT_VER = "criteria_v0.2"

SYSTEM = """You are a scientific answer normalizer and rubric writer for an automated verification system.

You are given a science problem, a reference answer (WARNING: it is usually just the bare answer, often empty \u2014 it rarely contains a worked solution), and the raw ground-truth answer.

FIRST, in your reasoning, solve the problem yourself and check that your result agrees with the raw ground truth. THEN output STRICT JSON inside a ```json fence with exactly these keys:

1. "final_answer": normalization of the raw ground truth:
   - "type": "numeric" | "expression" | "mcq" | "string"
   - "value": the canonical answer.
       numeric    -> ONE plain number (decimal or e-notation; no commas, no LaTeX, no unit inside)
       expression -> the answer as a clean LaTeX or plain math expression (no surrounding words, no \\[ \\] delimiters, no trailing units)
       mcq        -> the option letter only (A-J)
       string     -> canonical short string (chemical equations, matrices/vectors, multi-part answers, yes/no + qualifier)
   - "unit": for numeric, the unit implied by problem/ground truth in pint syntax (e.g. "mol/L", "m/s^2", "K"), else null
   - "rel_tol": for numeric, relative tolerance (default 0.02; use 0.05 if the ground truth is approximate or carries uncertainty)

2. "primary_axis": "quantitative" if numeric computation dominates the task, "symbolic" if derivation/expression manipulation dominates.

3. "intermediate_criteria": 0 to 4 checkpoints OF YOUR OWN (verified) DERIVATION \u2014 the things a correct solution must contain on the way to the answer:
   - "text": one short sentence describing the checkpoint
   - "verifier": "numeric_unit" | "sympy_equiv"   (ONLY these two)
   - "verifier_args":
       numeric_unit -> {"value": <intermediate number>, "rel_tol": 0.05}
       sympy_equiv  -> {"target": "<pure math expression or equation>"}
   - "axis": "quantitative" | "symbolic"
   - "weight": 0.25 to 1.0
   Good checkpoints: the governing equation/law in symbolic form; the symbolic form solved for the requested quantity; a key intermediate numeric value (converted units, a sub-result). They will be pattern-matched inside model-written solutions, so keep targets canonical and simple.

Hard rules:
- sympy_equiv targets: pure math only. Products of one-letter symbols with explicit spaces or \\cdot (e.g. "P V = n R T"). No chemistry concentration brackets like [X], no words, no matrices.
- Matrix/vector answers, chemical equations, and multi-part answers are type "string", never "expression".
- NEVER restate the final answer as an intermediate criterion \u2014 such duplicates are discarded automatically.
- Only include checkpoints you actually verified in your reasoning; if the problem is a single-step lookup with no meaningful intermediate, return [].

Think briefly and efficiently. Output minimal JSON \u2014 no extra keys, no comments, no trailing text after the fence."""

FEWSHOT_USER = """## Problem
If 5.31 moles of an ideal gas has a pressure of 5.11 atm, and a volume of 39.85 L what is the temperature of the sample?

## Reference answer
467.102

## Raw ground-truth answer
467.102"""

FEWSHOT_ASSISTANT = """```json
{
  "final_answer": {"type": "numeric", "value": "467.1", "unit": "K", "rel_tol": 0.02},
  "primary_axis": "quantitative",
  "intermediate_criteria": [
    {"text": "Uses the ideal gas law P V = n R T.",
     "verifier": "sympy_equiv", "verifier_args": {"target": "P V = n R T"},
     "axis": "symbolic", "weight": 0.5},
    {"text": "Isolates temperature as T = P V / (n R).",
     "verifier": "sympy_equiv", "verifier_args": {"target": "T = \\\\frac{P V}{n R}"},
     "axis": "symbolic", "weight": 0.5},
    {"text": "Computes the denominator n R = 5.31 \u00d7 0.08206 \u2248 0.4357 L\u00b7atm/K.",
     "verifier": "numeric_unit", "verifier_args": {"value": 0.4357, "rel_tol": 0.05},
     "axis": "quantitative", "weight": 0.25}
  ]
}
```"""

USER_TMPL = """## Problem
{question}

## Reference answer
{reference}

## Raw ground-truth answer
{ground_truth}"""

STRICT_SUFFIX = """

(Your previous attempt produced invalid JSON. Output ONLY the ```json fence with the exact schema \u2014 no prose.)"""

REGEN_SUFFIX = """

(Note: a previous normalization of this answer FAILED automated verification against the raw ground truth. This time:
- stay as close to the ground truth's own notation as possible (quote it verbatim minus surrounding words/delimiters);
- if the answer is not ONE clean number or ONE clean math expression \u2014 proportionality-only relations, integrals in unusual notation, matrices, multi-part or conditional answers \u2014 use type "string" with the ground truth quoted verbatim;
- for numeric answers keep the same precision as the ground truth.)"""


def messages(
    question: str, reference: str, ground_truth: str, strict: bool = False, regen: bool = False
) -> list[dict]:
    user = USER_TMPL.format(question=question, reference=reference or "(empty)", ground_truth=ground_truth)
    if regen:
        user += REGEN_SUFFIX
    if strict:
        user += STRICT_SUFFIX
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": FEWSHOT_USER},
        {"role": "assistant", "content": FEWSHOT_ASSISTANT},
        {"role": "user", "content": user},
    ]
