"""KIS OAuth token management.

KIS rate-limits token *issuance* to roughly once per minute, so the token is
cached to disk and reused for its full 24h life. Reissuing on every process
start will get you throttled, which matters because the live system runs three
scheduled jobs a day plus any manual script.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from uat.config import load_settings, project_root
from uat.logging import get

log = get("broker.kis_auth")

TOKEN_FILE = "data/.kis_token.json"


class MissingCredentials(RuntimeError):
    pass


@dataclass(frozen=True)
class KisCredentials:
    app_key: str
    app_secret: str
    account_8: str
    account_2: str
    env: str

    @property
    def is_paper(self) -> bool:
        return self.env == "paper"


def load_credentials(env: str | None = None) -> KisCredentials:
    """Read credentials from the environment (populated from .env by config)."""
    settings = load_settings(env)
    missing = [
        k
        for k in ("KIS_APP_KEY", "KIS_APP_SECRET", "KIS_ACCOUNT_8")
        if not os.environ.get(k)
    ]
    if missing:
        raise MissingCredentials(
            "missing "
            + ", ".join(missing)
            + f" -- copy .env.example to .env and fill it in (KIS_ENV={settings.env})"
        )
    return KisCredentials(
        app_key=os.environ["KIS_APP_KEY"],
        app_secret=os.environ["KIS_APP_SECRET"],
        account_8=os.environ["KIS_ACCOUNT_8"],
        account_2=os.environ.get("KIS_ACCOUNT_2", "01"),
        env=settings.env,
    )


def _token_path() -> Path:
    p = project_root() / TOKEN_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _read_cached(env: str, margin_hours: float) -> str | None:
    p = _token_path()
    if not p.exists():
        return None
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if blob.get("env") != env:
        return None
    expires = datetime.fromisoformat(blob["expires_at"])
    if expires - datetime.now() < timedelta(hours=margin_hours):
        return None
    return blob["access_token"]


def _write_cached(env: str, token: str, expires_in: int) -> None:
    _token_path().write_text(
        json.dumps(
            {
                "env": env,
                "access_token": token,
                "expires_at": (datetime.now() + timedelta(seconds=expires_in)).isoformat(),
            }
        ),
        encoding="utf-8",
    )


def get_access_token(env: str | None = None, force: bool = False) -> str:
    settings = load_settings(env)
    creds = load_credentials(env)
    margin = float(settings.get("broker.token_refresh_margin_hours", 2))

    if not force:
        cached = _read_cached(settings.env, margin)
        if cached:
            log.info("using cached KIS token", extra={"extra_fields": {"env": settings.env}})
            return cached

    base = settings.get("broker.base_url")
    resp = httpx.post(
        f"{base}/oauth2/tokenP",
        json={
            "grant_type": "client_credentials",
            "appkey": creds.app_key,
            "appsecret": creds.app_secret,
        },
        timeout=20,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"token request failed {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    token = data["access_token"]
    _write_cached(settings.env, token, int(data.get("expires_in", 86400)))
    log.info("issued new KIS token", extra={"extra_fields": {"env": settings.env}})
    return token
