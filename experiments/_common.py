"""Shared experiment I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def prepare_output_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(_json_value(data), indent=2) + "\n", encoding="utf-8")


def save_history(path: str | Path, history: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **history)


def save_figure(fig: Any, path: str | Path, *, dpi: int = 600) -> None:
    fig.savefig(path, format=Path(path).suffix.lstrip("."), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
