"""Configuration loading: config/base.yaml + config/{env}.yaml + .env overrides.

Settings are returned as a plain nested dict wrapped in a small accessor so that
new keys can be added to YAML without touching code. Hard risk caps are NOT read
from here -- they live in strategy/sizing.py so config cannot loosen them.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_dotenv() -> None:
    """Minimal .env loader; avoids a python-dotenv dependency."""
    for name in (".env", "config/.env"):
        p = project_root() / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


class Settings:
    def __init__(self, data: dict[str, Any], env: str):
        self._d = data
        self.env = env

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested key by dotted path, e.g. settings.get('risk.stop_atr_mult')."""
        node: Any = self._d
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str) -> Path:
        """Resolve a configured relative path against the project root."""
        return project_root() / str(self.get(dotted))

    @property
    def is_live(self) -> bool:
        return self.env == "live"

    def as_dict(self) -> dict[str, Any]:
        return self._d


@lru_cache(maxsize=4)
def load_settings(env: str | None = None) -> Settings:
    _load_dotenv()
    env = env or os.environ.get("KIS_ENV", "paper")
    if env not in ("paper", "live"):
        raise ValueError(f"KIS_ENV must be 'paper' or 'live', got {env!r}")
    cfg_dir = project_root() / "config"
    base = yaml.safe_load((cfg_dir / "base.yaml").read_text(encoding="utf-8"))
    overlay = yaml.safe_load((cfg_dir / f"{env}.yaml").read_text(encoding="utf-8")) or {}
    return Settings(_deep_merge(base, overlay), env)
