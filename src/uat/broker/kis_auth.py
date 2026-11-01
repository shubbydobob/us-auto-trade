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

TOKEN_FILE = "data/.kis_token_{env}.json"


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
    """Read credentials for the active environment.

    Paper and live are separate KIS registrations: different app keys, different
    secrets and usually a different account. Live values are read from
    KIS_LIVE_* and fall back to the unprefixed names, so a paper-only .env keeps
    working while a live run can never silently pick up the paper key.
    """
    settings = load_settings(env)
    prefix = "KIS_LIVE_" if settings.is_live else "KIS_"

    def pick(name: str, default: str | None = None) -> str | None:
        return os.environ.get(f"{prefix}{name}") or (
            os.environ.get(f"KIS_{name}") if settings.is_live else None
        ) or default

    app_key = pick("APP_KEY")
    app_secret = pick("APP_SECRET")
    account_8 = pick("ACCOUNT_8")
    account_2 = pick("ACCOUNT_2", "01")

    missing = [
        n
        for n, v in (("APP_KEY", app_key), ("APP_SECRET", app_secret),
                     ("ACCOUNT_8", account_8))
        if not v
    ]
    if missing:
        raise MissingCredentials(
            f"missing {', '.join(prefix + m for m in missing)} for KIS_ENV={settings.env}"
            " -- see .env.example"
        )
    return KisCredentials(
        app_key=app_key,
        app_secret=app_secret,
        account_8=account_8,
        account_2=account_2 or "01",
        env=settings.env,
    )


def _token_path(env: str) -> Path:
    """One cache file per environment.

    Sharing a single file means alternating paper/live invalidates the other
    token every switch, and KIS throttles issuance to about once a minute.
    """
    p = project_root() / TOKEN_FILE.format(env=env)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _read_cached(env: str, margin_hours: float) -> str | None:
    p = _token_path(env)
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
    _token_path(env).write_text(
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
