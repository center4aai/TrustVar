import ast
import random
import re
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field

from src.core.operators.base import (
    AbstractOperator,
    PreCheckResult,
    Tier,
    VariationResult,
)

TEMPLATE_SLOT_RE = re.compile(r"\{(\w+)(?:\|[^}]+)?\}")
_SLOT_VALUE_RE = re.compile(r"\{(\w+)\|([^}]+)\}")

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_MATH_EXPR_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)?"
)

_ANSWER_FN_KW_RE = re.compile(r"answer_fn\s*:\s*(lambda.*)$", re.IGNORECASE)

_MAX_ATTEMPTS = 50


class SlotType(str, Enum):
    INT = "int"
    FLOAT = "float"
    STR_ENUM = "str_enum"
    DATETIME = "datetime"
    NER = "ner"


class TemplateSlot(BaseModel):
    name: str
    type: SlotType = SlotType.INT
    constraints: List[str] = Field(default_factory=list)
    value_pool: Optional[List[Any]] = None
    description: str = ""


class TemplateSchema(BaseModel):
    """Describes a typed template for parametric substitution.

    Example YAML:
    ```yaml
    template: "If x = {a} and y = {b}, what is x + y?"
    slots:
      - name: a
        type: int
        constraints: ["a > 0", "a < 100"]
      - name: b
        type: int
        constraints: ["b > 0", "b < 100", "a != b"]
    template_constraints:
      - "(a, b) != (0, 0)"
    answer_fn: "lambda a, b: a + b"
    language: en
    ```
    """

    template: str = ""
    slots: List[TemplateSlot] = Field(default_factory=list)
    template_constraints: List[str] = Field(default_factory=list)
    answer_fn: Optional[str] = None
    language: str = "en"


_MATH_UNICODE_TO_PYTHON = {
    "≠": "!=",
    "≤": "<=",
    "≥": ">=",
    "∧": " and ",
    "∨": " or ",
    "¬": "not ",
    "×": "*",
    "÷": "/",
}

_TYPE_SYMBOLS = {"ℤ", "ℝ", "ℚ", "ℕ", "ℂ"}

_TYPE_SYMBOL_MAP = {
    "ℤ": SlotType.INT,
    "ℕ": SlotType.INT,
    "ℝ": SlotType.FLOAT,
    "ℚ": SlotType.FLOAT,
}  # ℂ → skip (unsupported)

_TYPE_ANNOTATION_RE = re.compile(r"(\w+)\s*(?:∈|in)\s*([ℤℝℚℕℂ])")


def _extract_type_annotations(raw: str) -> Dict[str, SlotType]:
    """Extract type annotations like ``a ∈ ℤ`` from constraint text.

    Returns a dict mapping variable name → :class:`SlotType`.

    ``ℤ`` / ``ℕ`` → ``SlotType.INT``
    ``ℝ`` / ``ℚ`` → ``SlotType.FLOAT``
    ``ℂ`` → skipped (not supported)
    """
    types = {}
    for m in _TYPE_ANNOTATION_RE.finditer(raw):
        var, symbol = m.group(1), m.group(2)
        st = _TYPE_SYMBOL_MAP.get(symbol)
        if st is not None:
            types[var] = st
    return types

_SAFE_CONSTRAINT_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "pow": pow,
    "round": round,
    "len": len,
    "sum": sum,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "range": range,
    "list": list,
    "tuple": tuple,
}


def _split_outer_commas(s: str) -> List[str]:
    """Split string by commas that are NOT inside parentheses."""
    parts = []
    depth = 0
    current = ""
    for ch in s:
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    rest = current.strip()
    if rest:
        parts.append(rest)
    return parts


def _parse_math_constraint(raw: str) -> List[str]:
    """Convert a math-notation constraint string to one or more Python expressions.

    Handles:
    - Unicode operators: ``≠ → !=``, ``≤ → <=``, ``≥ → >=``, ``∧ → and``, etc.
    - ``|var| → abs(var)``
    - Type annotations: ``a, b ∈ ℤ`` → filtered out (handled by ``type: int``)
    - Semicolon separator: combines multiple constraints in one string
    - Comma expansion: ``|a|, |b| < 100`` → ``abs(a) < 100``, ``abs(b) < 100``
      (respects parentheses, so ``(a, b) != (0, 0)`` stays intact)

    Backward compatible: a plain Python expression like ``"a > 0"`` passes through unchanged.
    """
    raw = raw.strip()
    if not raw:
        return []

    # Step 1: normalize Unicode math operators
    for uni, py in _MATH_UNICODE_TO_PYTHON.items():
        raw = raw.replace(uni, py)

    # Step 2: |expr| → abs(expr)
    raw = re.sub(r"\|([^|]+)\|", r"abs(\1)", raw)

    # Step 3: split on semicolons (compound constraints)
    parts = [p.strip() for p in raw.split(";") if p.strip()]

    result = []
    for part in parts:
        # Filter type annotations: a ∈ ℤ, x in ℝ, etc.
        if re.search(r"(?:in|∈)\s*([" + "".join(_TYPE_SYMBOLS) + r"])", part):
            continue

        # Find the leftmost operator
        multi_char_ops = ["!=", "<=", ">=", " in ", " not in "]
        single_char_ops = ["<", ">", "="]
        idx = -1
        chosen_op = ""

        for op in multi_char_ops:
            pos = part.find(op)
            if pos != -1 and (idx == -1 or pos < idx):
                idx = pos
                chosen_op = op

        for op in single_char_ops:
            pos = part.find(op)
            if pos != -1 and (idx == -1 or pos < idx):
                idx = pos
                chosen_op = op

        if idx == -1:
            result.append(part)
            continue

        left = part[:idx].strip()
        right = part[idx + len(chosen_op):].strip()

        if "," in left:
            items = _split_outer_commas(left)
            for item in items:
                if item:
                    result.append(f"{item} {chosen_op} {right}")
        else:
            result.append(part)

    return result


def _evaluate_constraint(constraint: str, values: Dict[str, Any]) -> bool:
    """Evaluate a constraint expression safely against slot values.

    Accepts both plain Python expressions (backward compatible) and
    mathematical notation:

    - Python: ``"abs(a) < 100"``, ``"a > 0"``, ``"(a, b) != (0, 0)"``
    - Math notation: ``"|a| < 100"``, ``"a, b ∈ ℤ; |a|, |b| < 100; (a, b) ≠ (0, 0)"``

    Slot names are available as variables. Safe builtins: abs, min, max,
    pow, round, len, sum, int, float, str, bool, range, list, tuple.
    """
    expressions = _parse_math_constraint(constraint)
    if not expressions:
        return True
    safe_locals = {**values, **_SAFE_CONSTRAINT_BUILTINS}
    for expr in expressions:
        try:
            if not bool(eval(expr, {"__builtins__": {}}, safe_locals)):
                return False
        except Exception:
            return False
    return True


def _execute_answer_fn(fn_str: str, values: Dict[str, Any]) -> Any:
    """Execute an answer-function string with given slot values."""
    try:
        fn = eval(fn_str, {"__builtins__": _SAFE_CONSTRAINT_BUILTINS}, {})
        return fn(**values)
    except Exception:
        return None


def _parse_slot_directives(text: str) -> Dict[str, str]:
    """Extract inline slot directives like {name|Paris,London,Berlin} or {x|1-100}."""
    directives = {}
    for match in _SLOT_VALUE_RE.finditer(text):
        directives[match.group(1)] = match.group(2)
    return directives


_TEXT_CONSTRAINT_SEP = ";;"


def _split_template(text: str) -> Tuple[str, str]:
    """Split text into template and inline constraints section.

    Format: ``<template> ;; <constraints>``

    The ``;;`` separator marks the constraints part, which is parsed
    via :func:`_parse_math_constraint`.  Returns ``(template, constraints_str)``.
    If no separator is found, constraints_str is ``""``.
    """
    if _TEXT_CONSTRAINT_SEP in text:
        idx = text.index(_TEXT_CONSTRAINT_SEP)
        return text[:idx].strip(), text[idx + len(_TEXT_CONSTRAINT_SEP):].strip()
    return text, ""


class MathExpression:
    __slots__ = ("a", "op", "b", "result", "original_text")

    def __init__(
        self,
        a: Union[int, float],
        op: str,
        b: Union[int, float],
        result: Optional[Union[int, float]] = None,
        original_text: str = "",
    ):
        self.a = a
        self.op = op
        self.b = b
        self.result = result
        self.original_text = original_text

    def compute_result(self) -> Optional[Union[int, float]]:
        if self.result is not None:
            return self.result
        try:
            if self.op == "+":
                return self.a + self.b
            if self.op == "-":
                return self.a - self.b
            if self.op == "*":
                return self.a * self.b
            if self.op == "/":
                if self.b == 0:
                    return None
                r = self.a / self.b
                return int(r) if isinstance(r, float) and r == int(r) else round(r, 4)
        except Exception:
            return None
        return None

    def render(self) -> str:
        def _fmt(v: Union[int, float]) -> str:
            if isinstance(v, float) and v == int(v):
                return str(int(v))
            return str(v)

        return f"{_fmt(self.a)} {self.op} {_fmt(self.b)} = "


def _detect_math_expressions(text: str) -> List[MathExpression]:
    results: List[MathExpression] = []
    seen: set = set()
    for m in _MATH_EXPR_RE.finditer(text):
        span = (m.start(), m.end())
        if span in seen:
            continue
        seen.add(span)
        try:
            a_str, op, b_str, r_str = (
                m.group(1),
                m.group(2),
                m.group(3),
                m.group(4),
            )
            a: Union[int, float] = float(a_str) if "." in a_str else int(a_str)
            b: Union[int, float] = float(b_str) if "." in b_str else int(b_str)
            result: Optional[Union[int, float]] = None
            if r_str:
                result = float(r_str) if "." in r_str else int(r_str)
            results.append(MathExpression(a=a, op=op, b=b, result=result, original_text=m.group(0)))
        except (ValueError, TypeError):
            continue
    return results


def _parse_answer_fn_from_constraints(constraints_str: str) -> Optional[str]:
    m = _ANSWER_FN_KW_RE.search(constraints_str)
    return m.group(1).strip() if m else None


def _infer_operand_range(value: Union[int, float]) -> Tuple[int, int]:
    """Return (lo, hi) bounds for integer sampling around *value*."""
    v = int(value) if isinstance(value, float) and value == int(value) else value
    if isinstance(v, float):
        abs_v = max(abs(v), 1.0)
        return (max(1, int(abs_v / 2)), int(abs_v * 3))
    abs_v = abs(v)
    if abs_v <= 0:
        return (1, 20)
    span = max(abs_v, 10) * 2
    lo = max(1, abs_v - abs_v // 2)
    hi = lo + span
    return (lo, max(hi, lo + 1))


def _compute_operation(a: Union[int, float], op: str, b: Union[int, float]) -> Optional[Union[int, float]]:
    try:
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            if b == 0:
                return None
            return a / b
    except Exception:
        return None
    return None


class TypedParametricSubstitutionOperator(AbstractOperator):
    operator_id = "typed_parametric_substitution"
    tier = Tier.A

    def __init__(self):
        self._value_pools: Dict[str, List] = self._load_value_pools()

    @staticmethod
    def _load_value_pools() -> Dict[str, List]:
        pools = {}
        for fname in ("en_names.yaml", "ru_names.yaml", "en_cities.yaml", "ru_cities.yaml"):
            path = _DATA_DIR / fname
            if not path.exists():
                raise FileNotFoundError(f"Required value pool not found: {path}")
            with open(path, encoding='utf-8') as f:
                key = fname.replace(".yaml", "")
                data = yaml.safe_load(f)
                pools[key] = data if isinstance(data, list) else []
        return pools

    async def check_preconditions(
        self,
        text: str,
        task_type: Optional[str] = None,
        language: Optional[str] = None,
        template_schema: Optional[TemplateSchema] = None,
        **kwargs,
    ) -> PreCheckResult:
        # If a template_schema is provided, check it explicitly
        if template_schema is not None:
            if not template_schema.slots:
                return PreCheckResult(passed=False, reason="Template has no slots defined")
            return PreCheckResult(
                passed=True,
                details={"slot_count": len(template_schema.slots), "schema_provided": True},
            )

        # Template text provided separately (instruction with {slots} + resolved variables)
        template_text = kwargs.get("template_text")
        if template_text:
            template_slots = TEMPLATE_SLOT_RE.findall(template_text)
            if len(template_slots) >= 1:
                return PreCheckResult(
                    passed=True,
                    details={
                        "slot_count": len(template_slots),
                        "schema_provided": False,
                        "mode": "template_text",
                    },
                )

        # Detect explicit {slots} from rendered text (strip inline constraints)
        template, _ = _split_template(text)
        slots = TEMPLATE_SLOT_RE.findall(template)
        if len(slots) >= 1:
            return PreCheckResult(
                passed=True,
                details={"slot_count": len(slots), "schema_provided": False},
            )

        # Detect math expressions in text (no explicit slots, but math patterns present)
        math_exprs = _detect_math_expressions(text)
        if math_exprs:
            return PreCheckResult(
                passed=True,
                details={
                    "mode": "math_expression",
                    "expression_count": len(math_exprs),
                    "schema_provided": False,
                },
            )

        return PreCheckResult(passed=False, reason="No template slots or math expressions found")

    async def apply(
        self,
        text: str,
        seed: Optional[int] = None,
        template_schema: Optional[TemplateSchema] = None,
        language: str = "en",
        **kwargs,
    ) -> VariationResult:
        rng = random.Random(seed)

        if template_schema is not None and template_schema.slots:
            return self._apply_from_schema(text, template_schema, rng)

        # Template text provided (instruction with {slots}) + resolved variables
        template_text = kwargs.get("template_text")
        template_vars = kwargs.get("template_variables")
        if template_text and template_vars:
            return self._apply_with_template(text, template_text, template_vars, rng, language)

        # Explicit {slots} in text + optional ;; constraints
        template_part, constraints_str = _split_template(text)
        if TEMPLATE_SLOT_RE.findall(template_part):
            return self._apply_heuristic(text, rng, language)

        # Auto-detect math expressions
        math_exprs = _detect_math_expressions(text)
        if math_exprs:
            return self._apply_math_variation(text, math_exprs, rng)

        return VariationResult(variant_text=text, metadata={"skipped": "no_applicable_mode"}, original_text=text)

    def _apply_from_schema(
        self,
        text: str,
        schema: TemplateSchema,
        rng: random.Random,
    ) -> VariationResult:
        values = {}
        for slot in schema.slots:
            val = self._sample_slot_value(slot, rng, values)
            if val is None:
                return VariationResult(
                    variant_text=text,
                    metadata={"skipped": f"no_valid_value_for_{slot.name}"},
                    original_text=text,
                )
            values[slot.name] = val

        # Check template-level cross-slot constraints
        for constraint in schema.template_constraints:
            if not _evaluate_constraint(constraint, values):
                return VariationResult(
                    variant_text=text,
                    metadata={"skipped": f"template_constraint_failed: {constraint}", "values": values},
                    original_text=text,
                )

        # Check all per-slot constraints
        for slot in schema.slots:
            for constraint in slot.constraints:
                if not _evaluate_constraint(constraint, values):
                    return VariationResult(
                        variant_text=text,
                        metadata={"skipped": f"constraint_failed: {constraint}", "values": values},
                        original_text=text,
                    )

        # Compute new gold answer
        answer = None
        if schema.answer_fn:
            answer = _execute_answer_fn(schema.answer_fn, values)

        variant = text
        for key, val in values.items():
            variant = re.sub(r"\{" + re.escape(key) + r"(?:\|[^}]+)?\}", str(val), variant)

        # Handle |directive slots that weren't in schema
        for match in _SLOT_VALUE_RE.finditer(variant):
            slot_name = match.group(1)
            if slot_name not in values:
                directives = _parse_slot_directives(text)
                raw = directives.get(slot_name, "")
                pool = [x.strip() for x in raw.split(",")]
                chosen = rng.choice(pool)
                variant = variant.replace(match.group(0), chosen)
                values[slot_name] = chosen

        return VariationResult(
            variant_text=variant,
            metadata={
                "substitutions": values,
                "new_gold": answer,
                "template_schema": schema.template,
            },
            original_text=text,
        )

    def _apply_heuristic(
        self,
        text: str,
        rng: random.Random,
        language: str = "en",
    ) -> VariationResult:
        template, constraints_str = _split_template(text)
        slots = [m.group(1) for m in re.finditer(TEMPLATE_SLOT_RE, template)]
        if not slots:
            return VariationResult(variant_text=text, metadata={"skipped": "no_slots"})

        all_constraints = _parse_math_constraint(constraints_str) if constraints_str else []
        type_annotations = _extract_type_annotations(constraints_str)
        answer_fn_str = _parse_answer_fn_from_constraints(constraints_str)

        # Filter out answer_fn directive from constraints before evaluation
        all_constraints = [
            c for c in all_constraints if not c.lower().startswith("answer_fn")
        ]

        values = {}
        for slot_name in slots:
            slot_type = type_annotations.get(slot_name, SlotType.INT)
            pool_key = self._infer_pool_key(slot_name, language)
            pool = self._value_pools.get(pool_key, [])

            if pool:
                candidates = [v for v in pool if str(v) != self._extract_current(template, template, slot_name)]
                values[slot_name] = rng.choice(candidates) if candidates else pool[0]
            else:
                values[slot_name] = self._sample_with_constraints(
                    slot_name, all_constraints, values, rng, slot_type,
                )

        # Check cross-slot constraints
        for c in all_constraints:
            if not _evaluate_constraint(c, values):
                return VariationResult(
                    variant_text=text,
                    metadata={"skipped": f"constraint_failed: {c}", "values": values, "mode": "heuristic"},
                    original_text=text,
                )

        variant = template
        for key, val in values.items():
            variant = re.sub(r"\{" + re.escape(key) + r"(?:\|[^}]+)?\}", str(val), variant)

        # Recompute gold answer if answer_fn is provided in constraints
        answer = None
        if answer_fn_str:
            answer = _execute_answer_fn(answer_fn_str, values)

        return VariationResult(
            variant_text=variant,
            metadata={
                "substitutions": values,
                "mode": "heuristic",
                "new_gold": answer,
            },
            original_text=text,
        )

    def _apply_math_variation(
        self,
        text: str,
        math_exprs: list,
        rng: random.Random,
    ) -> VariationResult:
        variant = text
        total_substitutions: Dict[str, Any] = {}
        new_gold_values: List[Union[int, float]] = []

        for expr in math_exprs:
            a_range = _infer_operand_range(expr.a)
            b_range = _infer_operand_range(expr.b)
            new_a: Union[int, float]
            new_b: Union[int, float]
            if isinstance(expr.a, int):
                new_a = rng.randint(*a_range)
            else:
                a_lo, a_hi = a_range
                new_a = round(rng.uniform(a_lo, a_hi), 2)
            if isinstance(expr.b, int):
                new_b = rng.randint(*b_range)
            else:
                b_lo, b_hi = b_range
                new_b = round(rng.uniform(b_lo, b_hi), 2)

            new_result = _compute_operation(new_a, expr.op, new_b)

            if expr.op == "/":
                while new_b == 0:
                    new_b = rng.randint(*b_range)

            new_expr = MathExpression(a=new_a, op=expr.op, b=new_b, result=new_result)
            rendered = new_expr.render()
            variant = variant.replace(expr.original_text, rendered, 1)

            total_substitutions[f"a_{len(total_substitutions)}"] = new_a
            total_substitutions[f"b_{len(total_substitutions)}"] = new_b
            total_substitutions[f"op_{len(total_substitutions)}"] = expr.op
            if new_result is not None:
                new_gold_values.append(new_result)

        return VariationResult(
            variant_text=variant,
            metadata={
                "substitutions": total_substitutions,
                "new_gold": new_gold_values[0] if len(new_gold_values) == 1 else (new_gold_values or None),
                "mode": "math_expression",
            },
            original_text=text,
        )

    def _apply_with_template(
        self,
        text: str,
        template_text: str,
        template_vars: Dict[str, Any],
        rng: random.Random,
        language: str = "en",
    ) -> VariationResult:
        template_slots = TEMPLATE_SLOT_RE.findall(template_text)
        new_vars: Dict[str, Any] = {}
        values: Dict[str, Any] = {}

        for slot_name in template_slots:
            orig_val = template_vars.get(slot_name)
            if orig_val is None:
                continue

            orig_str = str(orig_val)

            # Try to parse as math expression
            math_exprs = _detect_math_expressions(orig_str)
            if math_exprs and len(math_exprs) == 1:
                expr = math_exprs[0]
                a_range = _infer_operand_range(expr.a)
                b_range = _infer_operand_range(expr.b)
                if isinstance(expr.a, int):
                    new_a = rng.randint(*a_range)
                else:
                    new_a = round(rng.uniform(*a_range), 2)
                if isinstance(expr.b, int):
                    new_b = rng.randint(*b_range)
                else:
                    new_b = round(rng.uniform(*b_range), 2)
                if expr.op == "/":
                    while new_b == 0:
                        new_b = rng.randint(*b_range)
                new_result = _compute_operation(new_a, expr.op, new_b)
                new_expr = MathExpression(a=new_a, op=expr.op, b=new_b, result=new_result)
                new_vars[slot_name] = new_expr.render()
                values[slot_name] = new_expr.render()
                values[f"{slot_name}_result"] = new_result
            else:
                # Not a math expression — treat as generic slot
                pool_key = self._infer_pool_key(slot_name, language)
                pool = self._value_pools.get(pool_key, [])
                if pool:
                    candidates = [v for v in pool if str(v) != orig_str]
                    new_vars[slot_name] = rng.choice(candidates) if candidates else pool[0]
                else:
                    new_vars[slot_name] = self._sample_default(slot_name, rng)
                values[slot_name] = new_vars[slot_name]

        # Re-render template with new variables
        new_text = template_text
        all_vars = {**template_vars, **new_vars}
        for key, val in all_vars.items():
            new_text = re.sub(
                r"\{(\s*)?" + re.escape(key) + r"(\s*)?\}",
                str(val),
                new_text,
            )

        # Try to resolve remaining {slots} that weren't explicitly substituted
        for match in TEMPLATE_SLOT_RE.finditer(new_text):
            remaining = match.group(1)
            if remaining in new_vars:
                new_text = new_text.replace(match.group(0), str(new_vars[remaining]))

        # Extract gold answer: first {slot}_result found, or None
        new_gold = None
        for s in template_slots:
            result_key = f"{s}_result"
            if result_key in values:
                new_gold = values[result_key]
                break

        return VariationResult(
            variant_text=new_text,
            metadata={
                "substitutions": values,
                "new_gold": new_gold,
                "mode": "template_text",
            },
            original_text=text,
        )

    def _sample_slot_value(
        self,
        slot: TemplateSlot,
        rng: random.Random,
        current_values: Dict[str, Any],
    ) -> Optional[Any]:
        attempts = 0

        while attempts < _MAX_ATTEMPTS:
            attempts += 1
            if slot.value_pool:
                val = rng.choice(slot.value_pool)
            else:
                val = self._sample_by_type(slot.type, rng)

            if val is None:
                continue

            candidate_values = {**current_values, slot.name: val}
            all_ok = True
            for constraint in slot.constraints:
                if not _evaluate_constraint(constraint, candidate_values):
                    all_ok = False
                    break
            if all_ok:
                return val

        return None

    def _sample_with_constraints(
        self,
        slot_name: str,
        constraints: List[str],
        current_values: Dict[str, Any],
        rng: random.Random,
        slot_type: SlotType = SlotType.INT,
    ) -> Any:
        """Sample a value for *slot_name* that satisfies all *constraints*."""
        for _ in range(_MAX_ATTEMPTS):
            val = self._sample_default(slot_name, rng, slot_type)
            candidate = {**current_values, slot_name: val}
            ok = True
            for c in constraints:
                if not _evaluate_constraint(c, candidate):
                    ok = False
                    break
            if ok:
                return val
        return self._sample_default(slot_name, rng, slot_type)

    @staticmethod
    def _coerce_value(raw: str, st: SlotType) -> Any:
        if st == SlotType.INT:
            return int(raw)
        elif st == SlotType.FLOAT:
            return float(raw)
        return raw

    @staticmethod
    def _sample_by_type(st: SlotType, rng: random.Random) -> Any:
        if st == SlotType.INT:
            return rng.randint(1, 100)
        elif st == SlotType.FLOAT:
            return round(rng.uniform(0.1, 100.0), 2)
        elif st == SlotType.STR_ENUM:
            return rng.choice(["option_a", "option_b", "option_c"])
        elif st == SlotType.DATETIME:
            return f"{rng.randint(2000, 2025)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        elif st == SlotType.NER:
            return rng.choice(["Alice", "Bob", "Charlie", "Анна", "Пётр", "Мария", "Иван"])
        return rng.randint(1, 100)

    @staticmethod
    def _check_type(raw: str, slot: TemplateSlot) -> bool:
        st = slot.type
        if st == SlotType.INT:
            try:
                int(raw)
                return True
            except ValueError:
                return False
        elif st == SlotType.FLOAT:
            try:
                float(raw)
                return True
            except ValueError:
                return False
        elif st == SlotType.STR_ENUM:
            if slot.value_pool:
                return raw in slot.value_pool
            return bool(raw)
        elif st == SlotType.DATETIME:
            return bool(re.match(r"\d{4}-\d{2}-\d{2}", raw))
        elif st == SlotType.NER:
            return raw.isalpha() and raw[0].isupper()
        return True

    @staticmethod
    def _infer_pool_key(slot_name: str, language: str = "en") -> str:
        name_lower = slot_name.lower()
        lang = language if language in ("en", "ru") else "en"
        if name_lower in ("name", "person", "student", "teacher", "researcher"):
            return f"{lang}_names"
        if name_lower in ("city", "town", "capital", "country"):
            return f"{lang}_cities"
        return ""

    @staticmethod
    def _extract_current(original: str, variant: str, slot: str) -> str:
        """Extract the substituted value of a slot from the variant text."""
        slots = [m.group(1) for m in re.finditer(TEMPLATE_SLOT_RE, original)]
        if slot not in slots:
            return ""
        pattern_parts = []
        pos = 0
        for m in re.finditer(TEMPLATE_SLOT_RE, original):
            if m.start() > pos:
                pattern_parts.append(re.escape(original[pos:m.start()]))
            pattern_parts.append("(.+?)")
            pos = m.end()
        if pos < len(original):
            pattern_parts.append(re.escape(original[pos:]))
        pattern = "^" + "".join(pattern_parts) + "$"
        match = re.match(pattern, variant)
        if not match:
            return ""
        return match.group(slots.index(slot) + 1)

    @staticmethod
    def _sample_default(slot: str, rng: random.Random, slot_type: SlotType = SlotType.INT) -> Any:
        if slot_type == SlotType.FLOAT:
            return round(rng.uniform(-100.0, 100.0), 2)
        sample_map = {
            "a": rng.randint(1, 100),
            "b": rng.randint(1, 100),
            "n": rng.randint(1, 20),
            "k": rng.randint(0, 10),
            "x": rng.randint(-50, 50),
            "y": rng.randint(-50, 50),
            "name": rng.choice(["Alice", "Bob", "Charlie", "Diana", "Анна", "Мария", "Иван", "Пётр"]),
            "city": rng.choice(["Paris", "London", "Berlin", "Moscow", "Москва", "Париж", "Лондон"]),
            "number": rng.randint(1, 1000),
            "year": rng.randint(1900, 2024),
        }
        return sample_map.get(slot, rng.randint(1, 100))
