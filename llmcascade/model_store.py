"""Custom models + per-model overrides (enabled, weight/chance, key tier)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Literal

from llmcascade.secrets import data_dir

_LOCK = threading.Lock()
KeyTier = Literal["free", "paid"]


def _overrides_path() -> Path:
    return data_dir() / "model_overrides.json"


def _custom_path() -> Path:
    return data_dir() / "custom_models.json"


def _load(path: Path) -> dict[str, Any] | list[Any]:
    if not path.is_file():
        return {} if path.name.endswith("overrides.json") else []
    with _LOCK:
        raw = json.loads(path.read_text())
    return raw


def _save(path: Path, data: Any) -> None:
    with _LOCK:
        path.write_text(json.dumps(data, indent=2) + "\n")
        try:
            path.chmod(0o600)
        except OSError:
            pass


def load_overrides() -> dict[str, dict[str, Any]]:
    raw = _load(_overrides_path())
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name, entry in raw.items():
        if isinstance(entry, dict):
            out[str(name)] = entry
    return out


def save_overrides(data: dict[str, dict[str, Any]]) -> None:
    _save(_overrides_path(), data)


def get_override(name: str) -> dict[str, Any]:
    return dict(load_overrides().get(name) or {})


def set_override(
    name: str,
    *,
    enabled: bool | None = None,
    weight: int | None = None,
    key_tier: KeyTier | None = None,
) -> dict[str, Any]:
    data = load_overrides()
    entry = dict(data.get(name) or {})
    if enabled is not None:
        entry["enabled"] = bool(enabled)
    if weight is not None:
        entry["weight"] = max(1, int(weight))
    if key_tier is not None:
        entry["key_tier"] = "paid" if key_tier == "paid" else "free"
    data[name] = entry
    save_overrides(data)
    return entry


def load_custom_models() -> list[dict[str, Any]]:
    raw = _load(_custom_path())
    if not isinstance(raw, list):
        return []
    return [m for m in raw if isinstance(m, dict) and m.get("name")]


def save_custom_models(models: list[dict[str, Any]]) -> None:
    _save(_custom_path(), models)


def upsert_custom_model(model: dict[str, Any]) -> dict[str, Any]:
    name = str(model.get("name") or "").strip()
    if not name:
        raise ValueError("model name required")
    models = load_custom_models()
    out = []
    replaced = False
    for m in models:
        if m.get("name") == name:
            out.append(model)
            replaced = True
        else:
            out.append(m)
    if not replaced:
        out.append(model)
    save_custom_models(out)
    return model


def delete_custom_model(name: str) -> bool:
    models = load_custom_models()
    nxt = [m for m in models if m.get("name") != name]
    if len(nxt) == len(models):
        return False
    save_custom_models(nxt)
    ov = load_overrides()
    if name in ov:
        ov.pop(name, None)
        save_overrides(ov)
    return True
