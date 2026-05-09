"""Supabase user tracking and Stripe payment helpers for CodeLens."""

from __future__ import annotations

import os
from typing import Any

import stripe
from supabase import create_client, Client

from tools.project_env import load_project_env

load_project_env()

stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "")
_APP_URL = os.getenv("APP_URL", "http://localhost:8501").rstrip("/")


def _db() -> Client:
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_KEY", "")
    return create_client(url, key)


def get_or_create_user(github_username: str) -> dict[str, Any]:
    """Return the user row, creating it with 2 free uses if it doesn't exist."""
    db = _db()
    response = (
        db.table("users")
        .upsert(
            {"github_username": github_username},
            on_conflict="github_username",
            ignore_duplicates=True,
        )
        .execute()
    )
    result = db.table("users").select("*").eq("github_username", github_username).single().execute()
    return result.data or {}


def get_user(github_username: str) -> dict[str, Any]:
    """Fetch current user row from Supabase."""
    db = _db()
    result = db.table("users").select("*").eq("github_username", github_username).single().execute()
    return result.data or {}


def can_run_analysis(github_username: str) -> tuple[bool, str]:
    """
    Returns (allowed, reason).
    reason is one of: 'free', 'paid', 'blocked'
    """
    user = get_user(github_username)
    if not user:
        return False, "blocked"
    if user.get("analysis_authorized"):
        return True, "paid"
    free_remaining = user.get("free_uses_remaining", 0)
    if free_remaining > 0:
        return True, "free"
    return False, "blocked"


def consume_use(github_username: str, reason: str) -> None:
    """Decrement free uses or clear the paid authorization flag after a successful analysis."""
    db = _db()
    if reason == "free":
        db.rpc("decrement_free_uses", {"uname": github_username}).execute()
    elif reason == "paid":
        db.table("users").update({"analysis_authorized": False}).eq("github_username", github_username).execute()


def create_checkout_session(github_username: str) -> str:
    """Create a Stripe Checkout session and return the URL to redirect the user to."""
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{"price": _PRICE_ID, "quantity": 1}],
        mode="payment",
        success_url=f"{_APP_URL}?payment=success",
        cancel_url=f"{_APP_URL}?payment=cancelled",
        metadata={"github_username": github_username},
    )
    return session.url or ""


def free_uses_remaining(github_username: str) -> int:
    user = get_user(github_username)
    return int(user.get("free_uses_remaining", 0))
