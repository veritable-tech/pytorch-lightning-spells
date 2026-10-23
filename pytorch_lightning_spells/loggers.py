import json
import numbers
from enum import Enum
from pprint import pprint
from datetime import date, datetime
from typing import Any
from collections.abc import MutableMapping
from typing_extensions import override

from pytorch_lightning.loggers import Logger


def _sanitize_value(x):
    if x is None or isinstance(x, (bool, int, float, str)):
        # No-op
        return x
    if isinstance(x, Enum):
        # Prefer the raw value; fallback to name
        return _sanitize_value(getattr(x, "value", x.name))
    if isinstance(x, (datetime, date)):
        return x.isoformat()
    if isinstance(x, (bytes, bytearray, memoryview)):
        try:
            return bytes(x).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return str(x)
    # Handle numpy/torch scalars
    if hasattr(x, "item") and callable(x.item):
        try:
            scalar = x.item()
            if isinstance(scalar, (bool, int, float, str)):
                return scalar
        except Exception:  # noqa: BLE001, S110
            pass
    # Handle numpy arrays / torch tensors via tolist
    if hasattr(x, "tolist") and callable(x.tolist):
        try:
            return x.tolist()
        except Exception:  # noqa: BLE001, S110
            pass
    # Generic numeric conversions (e.g., numpy numeric types)
    if isinstance(x, numbers.Integral):
        try:
            return int(x)
        except Exception:  # noqa: BLE001, S110
            pass
    if isinstance(x, numbers.Real):
        try:
            return float(x)
        except Exception:  # noqa: BLE001, S110
            pass
    # Sequences
    if isinstance(x, (list, tuple, set)):
        return [_sanitize_value(v) for v in list(x)]
    # Mappings
    if isinstance(x, MutableMapping):
        return {str(k): _sanitize_value(v) for k, v in x.items()}
    if hasattr(x, "items") and callable(x.items):
        try:
            return {str(k): _sanitize_value(v) for k, v in x.items()}  # type: ignore
        except Exception:  # noqa: BLE001, S110
            pass
    # Last resort: keep if JSON can handle it, else stringify
    try:
        json.dumps(x)
        return x
    except Exception:  # noqa: BLE001
        return str(x)


def sanitize_dict_like_object(obj: MutableMapping[str, Any]) -> dict[str, Any]:
    """Convert a dict-like object into a dict and ensure all values are JSON-serializable."""
    return {str(k): _sanitize_value(v) for k, v in obj.items()}


class ScreenLogger(Logger):
    """A logger that prints metrics to the screen.

    Suitable in situation where you want to check the training progress directly in the console.
    """

    def __init__(self):
        super().__init__()

    @override
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if any(key.startswith("val_") for key in metrics):
            print("\n\n")
        for key, val in metrics.items():
            if key.startswith("val_"):
                print(key, f"{val:.4f}")

    @override
    def log_hyperparams(self, params):
        if isinstance(params, MutableMapping):
            pprint(sanitize_dict_like_object(params))
        else:
            pprint(params)

    @property
    @override
    def name(self):
        return ""

    @property
    @override
    def version(self) -> int:
        return 0

    @property
    @override
    def save_dir(self) -> str | None:
        """Return the root directory where experiment logs get saved, or `None` if the logger does not save data
        locally."""
        return None
