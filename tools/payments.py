"""Supabase user tracking and Stripe payment helpers for CodeLens."""

from __future__ import annotations

import os
from typing import Any

import httpx
import stripe

from tools.project_env import load_project_env

load_project_env()

_TABLE = "codelens_users"


def _cfg() -> tuple[str, str, str, str]:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
    stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
    return (
        os.getenv("SUPABASE_URL", "").rstrip("/"),
        os.getenv("SUPABASE_SERVICE_KEY", ""),
        os.getenv("STRIPE_PRICE_ID", ""),
        os.getenv("APP_URL", "http://localhost:8501").rstrip("/"),
    )


def _headers() -> dict[str, str]:
    _, key, _, _ = _cfg()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _patch(github_username: str, data: dict[str, Any]) -> None:
    url, _, _, _ = _cfg()
    r = httpx.patch(
        f"{url}/rest/v1/{_TABLE}",
        headers=_headers(),
        params={"github_username": f"eq.{github_username}"},
        json=data,
    )
    if not r.is_success:
        print(f"[payments] patch error {r.status_code}: {r.text}")
        r.raise_for_status()


def _fetch_user(github_username: str) -> dict[str, Any]:
    url, _, _, _ = _cfg()
    r = httpx.get(
        f"{url}/rest/v1/{_TABLE}",
        headers=_headers(),
        params={"github_username": f"eq.{github_username}", "limit": "1"},
    )
    if not r.is_success:
        print(f"[payments] {r.status_code} fetching user: {r.text}")
        r.raise_for_status()
    rows = r.json()
    return rows[0] if rows else {}


def get_or_create_user(github_username: str) -> dict[str, Any]:
    existing = _fetch_user(github_username)
    if existing:
        return existing
    url, _, _, _ = _cfg()
    r = httpx.post(
        f"{url}/rest/v1/{_TABLE}",
        headers=_headers(),
        json={"github_username": github_username},
    )
    if not r.is_success:
        print(f"[payments] {r.status_code} creating user: {r.text}")
        r.raise_for_status()
    rows = r.json()
    return rows[0] if rows else {}


def get_user(github_username: str) -> dict[str, Any]:
    return _fetch_user(github_username)


def can_run_analysis(github_username: str) -> tuple[bool, str]:
    """Returns (allowed, reason) where reason is 'free', 'paid', or 'blocked'."""
    user = get_user(github_username)
    if not user:
        user = get_or_create_user(github_username)
    if not user:
        return False, "blocked"
    if int(user.get("free_uses_remaining", 0)) > 0:
        return True, "free"
    if int(user.get("paid_uses_remaining", 0)) > 0:
        return True, "paid"
    return False, "blocked"


def consume_use(github_username: str, reason: str) -> None:
    user = _fetch_user(github_username)
    if reason == "free":
        current = int(user.get("free_uses_remaining", 0))
        _patch(github_username, {"free_uses_remaining": max(0, current - 1)})
    elif reason == "paid":
        current = int(user.get("paid_uses_remaining", 0))
        _patch(github_username, {"paid_uses_remaining": max(0, current - 1)})


def save_pending_analysis(
    github_username: str,
    github_url: str,
    job_description: str,
    company_url: str,
) -> None:
    """Save form inputs before redirecting to Stripe so they survive the round-trip."""
    _patch(github_username, {
        "pending_github_url": github_url,
        "pending_job_description": job_description,
        "pending_company_url": company_url,
    })


def get_pending_analysis(github_username: str) -> dict[str, str]:
    """Fetch saved form inputs after returning from Stripe."""
    user = _fetch_user(github_username)
    return {
        "github_url": user.get("pending_github_url") or "",
        "job_description": user.get("pending_job_description") or "",
        "company_url": user.get("pending_company_url") or "",
    }


def clear_pending_analysis(github_username: str) -> None:
    _patch(github_username, {
        "pending_github_url": None,
        "pending_job_description": None,
        "pending_company_url": None,
    })


def process_successful_payment(github_username: str, stripe_session_id: str) -> bool:
    """
    Increment paid_uses_remaining if this Stripe session hasn't been processed yet.
    Returns True if the credit was added, False if already processed.
    """
    user = _fetch_user(github_username)
    if not user:
        get_or_create_user(github_username)
        user = _fetch_user(github_username)
    if not user:
        raise RuntimeError(f"Could not find or create user {github_username!r}")

    # Idempotency check — skip if webhook already processed this session
    if user.get("last_stripe_session") == stripe_session_id:
        return False

    # Increment paid uses — this patch must succeed
    current = int(user.get("paid_uses_remaining", 0))
    _patch(github_username, {"paid_uses_remaining": current + 2})

    # Record the session ID to prevent double-processing (best-effort)
    try:
        _patch(github_username, {"last_stripe_session": stripe_session_id})
    except Exception:
        pass  # column may not exist yet; increment already happened

    return True


def create_checkout_session(github_username: str) -> str:
    _, _, price_id, app_url = _cfg()
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{"price": price_id, "quantity": 1}],
        mode="payment",
        success_url=f"{app_url}?payment=success&gh_user={github_username}&sid={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{app_url}?payment=cancelled",
        metadata={"github_username": github_username},
    )
    return session.url or ""


def free_uses_remaining(github_username: str) -> int:
    user = get_user(github_username)
    return int(user.get("free_uses_remaining", 0))
