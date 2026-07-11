"""JSON-safe sanitisation for API responses.

Starlette's ``JSONResponse`` serialises with ``json.dumps(..., allow_nan=False)``
and without FastAPI's ``jsonable_encoder``. Raw Mongo/eval payloads routinely
carry values that crash that path: ``NaN``/``Inf`` floats (pervasive in
TrustVar CV/TSI/EAR on single-variant or zero-mean cells), numpy scalars/arrays,
``ObjectId`` and ``datetime``. ``to_json_safe`` recursively rewrites a payload
into JSON-serialisable primitives so the response can be emitted safely.

numpy is imported optionally: the API container installs only the lightweight
``[api]`` extra (no numpy), and Mongo returns native Python types there, so the
sanitiser must not hard-depend on numpy at import time. When numpy IS present
(eval/celery contexts), its scalar/array types are handled too.

Pure function: never mutates its input, always returns a fresh structure.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

try:  # numpy is absent in the lightweight API container — stay import-safe
    import numpy as np

    _NUMPY_INT = (np.integer,)
    _NUMPY_FLOAT = (np.floating,)
    _NUMPY_BOOL = (np.bool_,)
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - exercised only in numpy-free runtimes
    np = None  # type: ignore[assignment]
    _NUMPY_INT = ()
    _NUMPY_FLOAT = ()
    _NUMPY_BOOL = ()
    _HAS_NUMPY = False

try:  # bson is always present in this Mongo project, but stay defensive
    from bson import ObjectId
except ImportError:  # pragma: no cover - bson is a hard dependency in practice
    ObjectId = None  # type: ignore[assignment]


def _clean_float(value: float) -> float | None:
    """Map non-finite floats (NaN/Inf/-Inf) to None; pass finite floats through."""
    if math.isnan(value) or math.isinf(value):
        return None
    return float(value)


def to_json_safe(obj: Any) -> Any:
    """Return a deep copy of ``obj`` containing only JSON-serialisable values.

    Conversions:
        - dict           -> dict with sanitised values (keys coerced to str)
        - list/tuple/set  -> list of sanitised items
        - numpy integer   -> int
        - numpy floating  -> float, or None if NaN/Inf
        - python float    -> float, or None if NaN/Inf
        - numpy bool_     -> bool
        - numpy ndarray   -> sanitised list
        - ObjectId        -> str
        - datetime/date   -> ISO-8601 string
        - everything else -> returned unchanged
    """
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(v) for v in obj]
    if _HAS_NUMPY and isinstance(obj, np.ndarray):
        return [to_json_safe(v) for v in obj.tolist()]
    # bool must precede int: bool is a subclass of int
    if isinstance(obj, (bool, *_NUMPY_BOOL)):
        return bool(obj)
    if isinstance(obj, _NUMPY_INT):
        return int(obj)
    if isinstance(obj, _NUMPY_FLOAT):
        return _clean_float(float(obj))
    if isinstance(obj, float):
        return _clean_float(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if ObjectId is not None and isinstance(obj, ObjectId):
        return str(obj)
    return obj
