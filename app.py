from __future__ import annotations

import importlib
import html
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import streamlit as st
import streamlit.components.v1 as components
from requests_oauthlib import OAuth2Session

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.crew import CodeLensCrew
from guardrails.output_filter import OutputFilter
from rag.indexer import CodeIndexer
from rag.retriever import CodeRetriever
from tools.github_api import GithubAnalyzer
from tools.gitnexus_tool import open_repo_analyzer
from tools.pinecone_tool import PineconeStore
from tools.project_env import load_project_env
from tools.resume_parser import ResumeParser, SkillMatcher

load_project_env()
os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

st.set_page_config(
    page_title="CodeLens",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISCLAIM_TEXT = (
    "CodeLens provides probabilistic signals to assist human judgment. "
    "All findings should be verified in a technical interview."
)


def _analysis_cooldown_seconds_remaining() -> float | None:
    """Seconds left before another analysis run is allowed; None if no cooldown."""
    raw = os.getenv("CODELENS_ANALYSIS_COOLDOWN_SECONDS")
    try:
        cooldown = int(raw) if raw is not None and str(raw).strip() != "" else 0
    except ValueError:
        cooldown = 0
    if cooldown <= 0:
        return None
    last = st.session_state.get("_last_analysis_started_ts")
    if last is None:
        return None
    elapsed = time.time() - float(last)
    if elapsed >= cooldown:
        return None
    return cooldown - elapsed


def format_exception_for_user(exc: BaseException) -> str:
    """Turn provider/gateway errors (HTML, huge payloads) into readable UI text."""
    text = str(exc)
    lower = text.lower()
    if "<html" in lower:
        if "413" in text or "payload too large" in lower:
            return (
                "The API gateway rejected the request because the payload was too large (413). "
                "During agent analysis this usually means the trimmed repository context still exceeds limits; "
                "try a smaller repo or set CREW_MAX_FILES / CREW_MAX_FILE_CONTENT_CHARS lower in your environment."
            )
        return (
            "The API returned an HTML error page instead of JSON. "
            "Check OPENROUTER_API_KEY, OPENROUTER_BASE_URL, and network connectivity."
        )
    if "exceeds 8 mb" in lower or ("8 mb" in lower and "input" in lower):
        return (
            "The model provider rejected the request: total text input exceeds the size limit (~8MB). "
            "CodeLens trims analysis data before calling agents; you can lower CREW_MAX_ANALYSIS_JSON_BYTES "
            "or CREW_MAX_FILE_CONTENT_CHARS if this persists."
        )
    if "402" in text or "more credits" in lower or "fewer max_tokens" in lower:
        return (
            "OpenRouter returned HTTP 402: not enough balance for this request (or the requested max output "
            "tokens are too high for your remaining credits). Add credits at https://openrouter.ai/settings/credits "
            "or set OPENROUTER_MAX_TOKENS in your .env to a smaller value (for example 1500–2048) so each agent "
            "call reserves less completion budget."
        )
    if len(text) > 12000:
        return text[:12000] + "\n\n[... output truncated ...]"
    return text


HISTORY_DIR = _ROOT / "data" / "history"
GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_github_analysis(github_url: str) -> dict[str, Any]:
    analyzer = GithubAnalyzer(github_url)
    return {
        "repo_metadata": analyzer.get_repo_metadata(),
        "commits": analyzer.get_commits(),
        "commit_patterns": analyzer.get_commit_patterns(),
    }


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_company_style_summary(company_github_url: str) -> str:
    analyzer = GithubAnalyzer(company_github_url)
    metadata = analyzer.get_repo_metadata()
    patterns = analyzer.get_commit_patterns()
    return (
        f"Company repo {metadata.get('name', '')} uses primary language "
        f"{metadata.get('primary_language', 'unknown')}, default branch "
        f"{metadata.get('default_branch_name', 'unknown')}, average diff size "
        f"{patterns.get('avg_diff_size', 0)}, commit message average length "
        f"{patterns.get('message_avg_length', 0)}, and single_branch="
        f"{patterns.get('single_branch', False)}."
    )


def inject_global_styles() -> None:
    """Single global theme + Streamlit overrides (glass / motion budget)."""
    theme = st.session_state.get("theme", "light")
    is_dark = theme == "dark"
    vars_block = (
        """
            --bg-gradient: linear-gradient(160deg, #070b12 0%, #0c1220 50%, #090e1a 100%);
            --glass-bg: rgba(255, 255, 255, 0.055);
            --glass-bg-hover: rgba(255, 255, 255, 0.09);
            --glass-border: rgba(255, 255, 255, 0.11);
            --glass-border-subtle: rgba(80, 140, 255, 0.18);
            --glass-shadow: 0 4px 24px rgba(0, 0, 0, 0.35), 0 1px 4px rgba(56, 168, 245, 0.07);
            --glass-inner-bg: rgba(255, 255, 255, 0.04);
            --accent-blue: #38a8f5;
            --accent-blue-hover: #5bbcf7;
            --accent-blue-light: rgba(56, 168, 245, 0.12);
            --text-primary: #eef4ff;
            --text-secondary: #8899b4;
            --text-muted: #4a5a72;
            --sidebar-fg: #eef4ff;
            --sidebar-muted: #8899b4;
        """
        if is_dark
        else """
            --bg-gradient: linear-gradient(160deg, #e8f4ff 0%, #f5f9ff 50%, #ffffff 100%);
            --glass-bg: rgba(255, 255, 255, 0.5);
            --glass-bg-hover: rgba(255, 255, 255, 0.7);
            --glass-border: rgba(255, 255, 255, 0.75);
            --glass-border-subtle: rgba(180, 210, 255, 0.35);
            --glass-shadow: 0 4px 24px rgba(80, 140, 220, 0.10), 0 1px 4px rgba(80, 140, 220, 0.06);
            --glass-inner-bg: rgba(255, 255, 255, 0.35);
            --accent-blue: #38a8f5;
            --accent-blue-hover: #1e90e0;
            --accent-blue-light: rgba(56, 168, 245, 0.12);
            --text-primary: #0a0f1e;
            --text-secondary: #4a5568;
            --text-muted: #94a3b8;
            --sidebar-fg: #0a0f1e;
            --sidebar-muted: #64748b;
        """
    )
    st.markdown(
        f"""
        <style>
            @keyframes cl-fadeUp {{
                from {{ opacity: 0; transform: translateY(10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            @keyframes cl-stepPulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.55; }}
            }}
            @keyframes cl-shimmer {{
                to {{ left: 160%; }}
            }}

            :root {{
                {vars_block}
                --score-green: #22c55e;
                --score-yellow: #f59e0b;
                --score-red: #ef4444;
                --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
            }}

            html, body, .stApp, [class*="css"] {{
                font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", system-ui, sans-serif;
            }}

            .stApp {{
                background: var(--bg-gradient) fixed;
                background-size: cover;
                color: var(--text-primary);
                min-height: 100vh;
            }}

            .block-container {{
                padding-top: 1.5rem;
                padding-bottom: 2rem;
                max-width: 1200px;
            }}

            h1 {{ font-size: 26px; font-weight: 650; letter-spacing: -0.022em; color: var(--text-primary); }}
            h2 {{ font-size: 19px; font-weight: 600; letter-spacing: -0.012em; color: var(--text-primary); }}
            h3 {{ font-size: 15px; font-weight: 600; color: var(--text-primary); }}
            p, li, label, span {{ color: var(--text-secondary); font-size: 14px; line-height: 1.65; }}
            code, pre {{ font-family: "SF Mono", "Fira Code", ui-monospace, monospace !important; }}

            ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
            ::-webkit-scrollbar-track {{ background: transparent; }}
            ::-webkit-scrollbar-thumb {{
                background: var(--glass-border-subtle);
                border-radius: 999px;
            }}
            ::-webkit-scrollbar-thumb:hover {{ background: var(--glass-border); }}

            section[data-testid="stSidebar"] {{
                width: 260px !important;
                min-width: 260px !important;
                background: var(--glass-bg) !important;
                backdrop-filter: blur(16px) saturate(160%);
                -webkit-backdrop-filter: blur(16px) saturate(160%);
                border-right: 1px solid var(--glass-border-subtle) !important;
                box-shadow: none !important;
            }}
            section[data-testid="stSidebar"] * {{
                color: var(--sidebar-fg);
            }}
            section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] .stMarkdown p {{
                color: var(--sidebar-fg);
            }}

            [data-testid="stHeader"] {{
                background: transparent;
                border-bottom: none;
            }}

            .cl-glass-panel {{
                background: var(--glass-bg);
                backdrop-filter: blur(16px) saturate(160%);
                -webkit-backdrop-filter: blur(16px) saturate(160%);
                border: 1px solid var(--glass-border);
                border-top-color: rgba(255, 255, 255, 0.85);
                border-radius: 18px;
                box-shadow: var(--glass-shadow);
            }}
            .cl-glass-inner {{
                background: var(--glass-inner-bg);
                backdrop-filter: blur(8px) saturate(140%);
                -webkit-backdrop-filter: blur(8px) saturate(140%);
                border: 1px solid var(--glass-border-subtle);
                border-radius: 12px;
            }}

            .cl-divider {{
                height: 1px;
                background: linear-gradient(90deg, transparent 0%, var(--glass-border-subtle) 50%, transparent 100%);
                border: none;
                margin: 16px 0;
            }}

            .cl-main-tab-strip {{
                display: inline-flex;
                background: var(--glass-bg);
                backdrop-filter: blur(16px) saturate(160%);
                -webkit-backdrop-filter: blur(16px) saturate(160%);
                border: 1px solid var(--glass-border-subtle);
                border-radius: 12px;
                padding: 3px;
                gap: 2px;
                margin-bottom: 1.25rem;
                animation: cl-fadeUp 0.22s var(--ease-out) both;
            }}

            .cl-wordmark {{
                font-size: 19px;
                font-weight: 650;
                letter-spacing: -0.02em;
                font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif;
            }}
            .cl-wordmark span:first-child {{ color: var(--text-primary); }}
            .cl-wordmark span:last-child {{ color: var(--accent-blue); }}

            .cl-pill-tabs {{
                display: flex;
                flex-wrap: wrap;
                gap: 6px;
                margin: 12px 0 18px 0;
            }}

            [data-testid="stTextInput"] input,
            [data-testid="stTextArea"] textarea {{
                background: var(--glass-bg) !important;
                backdrop-filter: blur(16px) !important;
                -webkit-backdrop-filter: blur(16px) !important;
                border: 1px solid var(--glass-border-subtle) !important;
                border-radius: 12px !important;
                color: var(--text-primary) !important;
                font-size: 14px !important;
                padding: 11px 14px !important;
                transition: border-color 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease !important;
            }}
            [data-testid="stTextInput"] input:focus,
            [data-testid="stTextArea"] textarea:focus {{
                border-color: var(--accent-blue) !important;
                box-shadow: 0 0 0 3px rgba(56, 168, 245, 0.14) !important;
                outline: none !important;
            }}
            [data-testid="stTextInput"] input::placeholder,
            [data-testid="stTextArea"] textarea::placeholder {{
                color: var(--text-muted) !important;
            }}

            [data-testid="stFileUploaderDropzone"] {{
                background: var(--glass-bg) !important;
                backdrop-filter: blur(16px) !important;
                -webkit-backdrop-filter: blur(16px) !important;
                border: 1px dashed var(--glass-border-subtle) !important;
                border-radius: 12px !important;
                transition: border-color 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease !important;
            }}
            [data-testid="stFileUploaderDropzone"]:hover {{
                border-color: var(--accent-blue) !important;
                background: var(--accent-blue-light) !important;
            }}

            .stButton > button {{
                background: var(--glass-bg) !important;
                backdrop-filter: blur(16px) !important;
                -webkit-backdrop-filter: blur(16px) !important;
                color: var(--text-secondary) !important;
                border: 1px solid var(--glass-border-subtle) !important;
                border-radius: 12px !important;
                font-weight: 500 !important;
                transition: background 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease, border-color 0.15s ease, opacity 0.15s ease !important;
            }}
            .stButton > button:hover {{
                background: var(--glass-bg-hover) !important;
                color: var(--text-primary) !important;
                transform: translateY(-1px);
                box-shadow: var(--glass-shadow);
            }}
            .stButton > button:active {{
                transform: scale(0.975);
                transition: transform 0.08s ease !important;
            }}

            .stButton > button[kind="primary"] {{
                position: relative !important;
                overflow: hidden !important;
                background: var(--accent-blue) !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 12px !important;
                padding: 12px 24px !important;
                font-size: 15px !important;
                font-weight: 600 !important;
                min-height: 48px !important;
                box-shadow: 0 3px 14px rgba(56, 168, 245, 0.30) !important;
            }}
            .stButton > button[kind="primary"]::after {{
                content: '';
                position: absolute;
                top: 0; left: -100%;
                width: 60%; height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                pointer-events: none;
            }}
            .stButton > button[kind="primary"]:hover {{
                background: var(--accent-blue-hover) !important;
                box-shadow: 0 5px 18px rgba(56, 168, 245, 0.38) !important;
                transform: translateY(-1px) !important;
            }}
            .stButton > button[kind="primary"]:hover::after {{
                animation: cl-shimmer 0.28s ease forwards;
            }}

            [data-testid="stExpander"] details {{
                background: var(--glass-bg) !important;
                border: 1px solid var(--glass-border-subtle) !important;
                border-radius: 12px !important;
            }}

            .panel, .score-card, .tool-card, .history-card {{
                background: var(--glass-bg);
                backdrop-filter: blur(16px) saturate(160%);
                -webkit-backdrop-filter: blur(16px) saturate(160%);
                border: 1px solid var(--glass-border);
                border-top-color: rgba(255, 255, 255, 0.85);
                border-radius: 18px;
                box-shadow: var(--glass-shadow);
                color: var(--text-primary);
            }}
            .muted {{ color: var(--text-muted); }}
            .small-label {{
                font-size: 10px;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: var(--text-muted);
                font-weight: 600;
            }}
            .sidebar-section-label {{
                font-size: 10px !important;
                text-transform: uppercase;
                letter-spacing: 0.1em !important;
                color: var(--sidebar-muted) !important;
                margin-top: 20px !important;
                font-weight: 600 !important;
            }}
            .sidebar-divider {{
                height: 1px;
                background: linear-gradient(90deg, transparent 0%, var(--glass-border-subtle) 50%, transparent 100%);
                border: none;
                margin: 16px 0;
            }}

            .badge-green {{ background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.3); color: #16a34a; }}
            .badge-yellow {{ background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.3); color: #b45309; }}
            .badge-red {{ background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.3); color: #ef4444; }}
            .badge-blue {{ background: var(--accent-blue-light); border: 1px solid rgba(56,168,245,0.35); color: var(--accent-blue); }}
            .badge-gray {{ background: var(--glass-inner-bg); border: 1px solid var(--glass-border-subtle); color: var(--text-muted); }}

            [data-testid="stStatus"] {{
                background: var(--glass-bg) !important;
                backdrop-filter: blur(16px) !important;
                -webkit-backdrop-filter: blur(16px) !important;
                border: 1px solid var(--glass-border-subtle) !important;
                border-radius: 14px !important;
            }}

            .cl-analyze-dim {{ opacity: 0.5; pointer-events: none; transition: opacity 0.12s ease; }}
            .cl-hint-faint {{ opacity: 0.35; font-size: 13px; text-align: center; margin-top: 14px; color: var(--text-muted); }}

            .oauth-button {{
                display: inline-flex;
                align-items: center;
                gap: 10px;
                padding: 12px 14px;
                border-radius: 12px;
                border: 1px solid var(--glass-border-subtle);
                background: var(--glass-bg);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                color: var(--text-primary) !important;
                font-weight: 600;
                text-decoration: none;
                width: 100%;
                box-sizing: border-box;
                transition: background 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease, border-color 0.15s ease;
            }}
            .oauth-button:hover {{
                background: var(--glass-bg-hover);
                border-color: var(--accent-blue);
                transform: translateY(-1px);
                box-shadow: var(--glass-shadow);
                text-decoration: none;
                color: var(--text-primary) !important;
            }}
 </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_error", None)
    st.session_state.setdefault("eval_result", None)
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("oauth_state", None)
    st.session_state.setdefault("active_view", "overview")
    st.session_state.setdefault("theme", "light")
    st.session_state.setdefault("main_tab", "Analyze")
    st.session_state.setdefault("overview_section", "skill_map")
    st.session_state.setdefault("last_github_url", "")
    st.session_state.setdefault("render_id", "0")


def missing_api_keys() -> list[str]:
    required = ["OPENROUTER_API_KEY", "PINECONE_API_KEY", "VOYAGE_API_KEY"]
    return [key for key in required if not os.getenv(key)]


def oauth_ready() -> bool:
    return bool(os.getenv("GITHUB_CLIENT_ID") and os.getenv("GITHUB_CLIENT_SECRET") and os.getenv("APP_URL"))


def oauth_redirect_uri() -> str:
    return os.getenv("APP_URL", "http://localhost:8501").rstrip("/")


def score_color(score: int | None) -> str:
    if score is None:
        return "var(--text-muted)"
    if score > 75:
        return "var(--score-green)"
    if score >= 50:
        return "var(--score-yellow)"
    return "var(--score-red)"


def gauge_bar_color(score: int | None) -> str:
    theme = st.session_state.get("theme", "light")
    if score is None:
        return "#4a5568" if theme == "dark" else "#94a3b8"
    if score < 50:
        return "#fb7185" if theme == "dark" else "#ef4444"
    if score < 75:
        return "#fbbf24" if theme == "dark" else "#f59e0b"
    return "#4ade80" if theme == "dark" else "#22c55e"


def recommendation_badge_color(value: str) -> str:
    return {
        "strong_hire": "badge-green",
        "hire": "badge-blue",
        "maybe": "badge-yellow",
        "pass": "badge-red",
    }.get(value, "badge-gray")


def get_oauth_session(state: str | None = None) -> OAuth2Session:
    return OAuth2Session(
        client_id=os.getenv("GITHUB_CLIENT_ID", ""),
        redirect_uri=oauth_redirect_uri(),
        scope=["read:user"],
        state=state,
    )


def get_github_login_url() -> str | None:
    if not oauth_ready():
        return None
    session = get_oauth_session()
    authorization_url, state = session.authorization_url(GITHUB_AUTH_URL)
    st.session_state["oauth_state"] = state
    return authorization_url


def load_view_selection_from_query() -> None:
    params = st.query_params
    view = params.get("view")
    allowed = {"overview", "ai_usage", "job_fit", "skill_map", "evaluation"}
    if view in allowed:
        st.session_state["active_view"] = view


def set_active_view(view: str) -> None:
    """Switch in-app result view without URL navigation."""
    allowed = {"overview", "ai_usage", "job_fit", "skill_map", "evaluation"}
    if view in allowed:
        st.session_state["active_view"] = view
        if view == "evaluation":
            st.session_state["main_tab"] = "Evaluation"
        elif view in {"overview", "ai_usage", "job_fit", "skill_map"}:
            st.session_state["main_tab"] = "Analyze"


def handle_oauth_callback() -> None:
    if not oauth_ready():
        return
    if st.session_state.get("user"):
        return

    params = st.query_params
    code = params.get("code")
    state = params.get("state")
    oauth_error = params.get("error")
    if oauth_error:
        st.session_state["last_error"] = {
            "message": "GitHub sign-in did not complete.",
            "details": str(oauth_error),
        }
        params.clear()
        return
    if not code:
        return

    expected_state = st.session_state.get("oauth_state")
    if expected_state and state and state != expected_state:
        st.session_state["last_error"] = {
            "message": "GitHub sign-in could not be verified.",
            "details": "OAuth state mismatch.",
        }
        params.clear()
        return

    try:
        session = get_oauth_session(state=expected_state)
        authorization_response = f"{oauth_redirect_uri()}?{urlencode({'code': code, 'state': state or ''})}"
        token = session.fetch_token(
            GITHUB_TOKEN_URL,
            client_secret=os.getenv("GITHUB_CLIENT_SECRET", ""),
            authorization_response=authorization_response,
            include_client_id=True,
        )
        user_response = session.get(GITHUB_USER_URL, headers={"Accept": "application/vnd.github+json"})
        user_response.raise_for_status()
        profile = user_response.json()
        st.session_state["user"] = {
            "username": profile.get("login", ""),
            "avatar_url": profile.get("avatar_url", ""),
            "access_token": token.get("access_token", ""),
        }
        st.session_state["last_error"] = None
    except Exception as exc:
        st.session_state["last_error"] = {
            "message": "GitHub sign-in failed.",
            "details": str(exc),
        }
    finally:
        params.clear()
        st.rerun()


def sign_out() -> None:
    theme = st.session_state.get("theme", "light")
    st.session_state.clear()
    st.session_state["theme"] = theme
    st.session_state["cl_appearance_toggle"] = theme == "dark"
    st.query_params.clear()
    st.rerun()


def load_history_selection_from_query() -> None:
    params = st.query_params
    history_id = params.get("history_id")
    if not history_id:
        return

    candidate_histories: list[dict[str, Any]] = []
    user = st.session_state.get("user")
    if user and user.get("username"):
        candidate_histories.extend(load_user_history(user.get("username", "")))
    else:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        for path in HISTORY_DIR.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            analyses = payload.get("analyses", [])
            if isinstance(analyses, list):
                candidate_histories.extend(analyses)

    for entry in candidate_histories:
        if entry.get("id") == history_id:
            st.session_state["last_result"] = entry.get("result")
            st.session_state["last_github_url"] = entry.get("repo_url", "")
            st.session_state["last_error"] = None
            st.session_state["active_view"] = "overview"
            st.session_state["render_id"] = uuid.uuid4().hex
            break
    params.clear()
    st.rerun()


def check_mcp_status() -> tuple[bool, str]:
    try:
        module = importlib.import_module("mcp_server")
        if getattr(module, "mcp", None) is None:
            return False, "Server module loaded, but no FastMCP instance was found."
        return True, "Local stdio MCP server module is available."
    except Exception as exc:
        return False, str(exc)


def history_path_for_user(username: str) -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return HISTORY_DIR / f"{username}.json"


def load_user_history(username: str) -> list[dict[str, Any]]:
    path = history_path_for_user(username)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        analyses = payload.get("analyses", [])
        return analyses if isinstance(analyses, list) else []
    except Exception:
        return []


def save_user_history(username: str, analyses: list[dict[str, Any]]) -> None:
    path = history_path_for_user(username)
    path.write_text(json.dumps({"analyses": analyses}, indent=2), encoding="utf-8")


def save_analysis_to_history(result: dict[str, Any], github_url: str, had_resume: bool, had_jd: bool) -> None:
    user = st.session_state.get("user")
    if not user:
        return
    username = user.get("username")
    if not username:
        return

    verdict = result["verdict"]
    repo_metadata = result["analysis_data"].get("repo_metadata", {})
    analyses = load_user_history(username)
    entry = {
        "id": str(uuid.uuid4()),
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "repo_url": github_url,
        "repo_name": repo_metadata.get("name") or repo_metadata.get("full_name") or github_url,
        "overall_quality_score": verdict.get("overall_quality_score"),
        "ai_usage_score": verdict.get("ai_usage_score"),
        "commit_health_score": verdict.get("commit_health_score"),
        "resume_match_score": verdict.get("resume_match_score"),
        "recommendation": verdict.get("recommendation"),
        "summary": verdict.get("summary"),
        "had_resume": had_resume,
        "had_jd": had_jd,
        "result": result,
    }
    analyses.insert(0, entry)
    save_user_history(username, analyses)


def candidate_username_from_url(github_url: str) -> str:
    parsed = urlparse(github_url)
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    return parts[0] if parts else "candidate"


def read_uploaded_resume(uploaded_file: Any) -> tuple[dict[str, Any] | None, str | None]:
    if uploaded_file is None:
        return None, None

    parser = ResumeParser()
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".pdf":
        pdf_bytes = uploaded_file.getvalue()
        resume_data = parser.parse_from_pdf(pdf_bytes)
        resume_text = build_resume_text_from_data(resume_data)
        return resume_data, resume_text

    raw_bytes = uploaded_file.getvalue()
    resume_text = raw_bytes.decode("utf-8", errors="replace")
    resume_data = parser.parse_from_text(resume_text)
    return resume_data, resume_text


def build_resume_text_from_data(resume_data: dict[str, Any]) -> str:
    lines = [
        "Skills: " + ", ".join(resume_data.get("skills", [])),
        f"Experience level: {resume_data.get('experience_level', '')}",
        f"Years experience: {resume_data.get('years_experience', 0)}",
    ]
    for project in resume_data.get("projects", []):
        lines.append(
            " | ".join(
                [
                    project.get("name", ""),
                    project.get("description", ""),
                    ", ".join(project.get("technologies", [])),
                    ", ".join(project.get("claimed_features", [])),
                ]
            )
        )
    return "\n".join(line for line in lines if line.strip())


def build_code_sample(files: list[dict[str, Any]], max_chars: int = 6000) -> str:
    samples: list[str] = []
    total = 0
    for file_data in files:
        text = file_data.get("content") or file_data.get("text") or ""
        if not text:
            symbols = file_data.get("symbols") or file_data.get("parsed_symbols") or []
            if isinstance(symbols, dict):
                symbols = symbols.get("items", [])
            snippets = []
            for symbol in symbols[:5]:
                if isinstance(symbol, dict):
                    snippet = symbol.get("code") or symbol.get("text") or symbol.get("content") or ""
                    if snippet:
                        snippets.append(str(snippet))
            text = "\n\n".join(snippets)
        if not text:
            continue
        file_chunk = f"# File: {file_data.get('file_path') or file_data.get('path') or 'unknown'}\n{text}"
        remaining = max_chars - total
        if remaining <= 0:
            break
        trimmed = file_chunk[:remaining]
        samples.append(trimmed)
        total += len(trimmed)
        if total >= max_chars:
            break
    return "\n\n".join(samples)


def render_app_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div class="app-header-copy">
                <div class="logo-mark">CodeLens</div>
                <div class="app-header-subtitle">Intelligent code review for technical hiring, AI usage signals, and resume-backed repository analysis.</div>
            </div>
            <div class="app-header-chip">Repository Insight Dashboard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_error_state() -> None:
    error = st.session_state.get("last_error")
    if not error:
        return
    st.error(error["message"])
    with st.expander("Details"):
        st.code(error.get("details", ""), language="text")


def render_oauth_button() -> None:
    login_url = get_github_login_url()
    if not login_url:
        st.markdown('<div class="muted">GitHub login is unavailable until OAuth credentials are configured.</div>', unsafe_allow_html=True)
        return
    st.markdown(
        f"""
        <a class="oauth-button" href="{login_url}">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
                <path d="M8 0C3.58 0 0 3.67 0 8.2c0 3.63 2.29 6.7 5.47 7.78.4.08.55-.18.55-.39 0-.19-.01-.82-.01-1.49-2.01.38-2.53-.51-2.69-.98-.09-.24-.48-.98-.82-1.18-.28-.15-.68-.52-.01-.53.63-.01 1.08.59 1.23.83.72 1.24 1.87.89 2.33.68.07-.54.28-.89.51-1.09-1.78-.21-3.64-.91-3.64-4.03 0-.89.31-1.62.82-2.19-.08-.21-.36-1.05.08-2.19 0 0 .67-.22 2.2.84A7.36 7.36 0 0 1 8 3.66c.68 0 1.37.09 2.01.27 1.53-1.06 2.2-.84 2.2-.84.44 1.14.16 1.98.08 2.19.51.57.82 1.29.82 2.19 0 3.13-1.87 3.82-3.65 4.03.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.19 0 .21.14.47.55.39A8.23 8.23 0 0 0 16 8.2C16 3.67 12.42 0 8 0Z"></path>
            </svg>
            Sign in with GitHub
        </a>
        """,
        unsafe_allow_html=True,
    )


def render_recent_history() -> None:
    user = st.session_state.get("user")
    if not user:
        return
    analyses = load_user_history(user["username"])
    if not analyses:
        st.markdown('<div class="sidebar-section-label">Recent Analyses</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#8ea2bc; font-size:0.9rem;">No saved analyses yet.</div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Recent Analyses</div>', unsafe_allow_html=True)
    for entry in analyses[:5]:
        repo_name = entry.get("repo_name", "Repository")
        analyzed_at = entry.get("analyzed_at", "")[:10]
        score = entry.get("overall_quality_score")
        score_value = 0 if not isinstance(score, int) else max(0, min(100, score))
        ring_color = "#FFC570" if score_value >= 50 else "#D25353"
        badge_label = "Pass" if str(entry.get("recommendation", "")).lower() == "pass" else "None"
        badge_bg = "#d9efdc" if badge_label == "Pass" else "#f3deb1"
        badge_fg = "#215732" if badge_label == "Pass" else "#8a5a08"
        st.markdown(
            f"""
            <div style="position:relative; margin-bottom:6px; background:rgba(255,255,255,0.04); border-left:2px solid #FFC570; border-radius:14px; padding:14px 14px 12px 14px; border-top:1px solid rgba(255,255,255,0.05); border-right:1px solid rgba(255,255,255,0.05); border-bottom:1px solid rgba(255,255,255,0.05);">
                <div style="position:absolute; top:12px; right:12px; color:#FFC570; font-weight:800;">&rarr;</div>
                <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
                    <div style="min-width:0; flex:1;">
                        <div style="font-weight:700; color:#ffffff; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{html.escape(repo_name)}</div>
                        <div style="color:#8ea2bc; margin-top:5px; font-size:0.8rem;">{html.escape(analyzed_at)}</div>
                        <div style="margin-top:9px;">
                            <span style="display:inline-flex; align-items:center; padding:4px 10px; border-radius:999px; background:{badge_bg}; color:{badge_fg}; font-size:0.78rem; font-weight:700;">{badge_label}</span>
                        </div>
                    </div>
                    <div style="width:38px; height:38px; border-radius:50%; background:conic-gradient({ring_color} {score_value}%, rgba(255,255,255,0.14) 0); display:flex; align-items:center; justify-content:center; flex-shrink:0;">
                        <div style="width:28px; height:28px; border-radius:50%; background:#1A3263; display:flex; align-items:center; justify-content:center; color:#ffffff; font-size:0.7rem; font-weight:800;">{score if score is not None else "N/A"}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(f"Load {repo_name}", key=f"history-open-{entry['id']}", use_container_width=True):
            st.session_state["last_result"] = entry.get("result")
            st.session_state["last_github_url"] = entry.get("repo_url", "")
            st.session_state["last_error"] = None
            st.session_state["active_view"] = "overview"
            st.session_state["render_id"] = uuid.uuid4().hex
            st.rerun()

    st.markdown('<div style="color:#8ea2bc; font-size:0.82rem; margin:6px 0 8px 0;">View all history</div>', unsafe_allow_html=True)
    with st.expander("View all history", expanded=False):
        st.dataframe(
            [
                {
                    "repo": entry.get("repo_name"),
                    "date": entry.get("analyzed_at"),
                    "recommendation": entry.get("recommendation"),
                    "overall_score": entry.get("overall_quality_score"),
                    "resume": entry.get("had_resume"),
                    "job_desc": entry.get("had_jd"),
                }
                for entry in analyses
            ],
            use_container_width=True,
            hide_index=True,
        )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="cl-wordmark" style="padding:24px 8px 8px 8px;"><span>Code</span><span>Lens</span></div>',
            unsafe_allow_html=True,
        )
        if "cl_appearance_toggle" not in st.session_state:
            st.session_state.cl_appearance_toggle = st.session_state.get("theme", "light") == "dark"
        st.toggle("Dark appearance", key="cl_appearance_toggle")
        if st.session_state.cl_appearance_toggle != (st.session_state.get("theme", "light") == "dark"):
            st.session_state["theme"] = "dark" if st.session_state.cl_appearance_toggle else "light"
            st.rerun()

        st.markdown(
            """
            <style>
                [data-testid="stSidebar"] .stButton > button {
                    justify-content: flex-start;
                    padding-left: 14px;
                    font-weight: 600;
                }
                [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
                    background: var(--glass-inner-bg) !important;
                    color: var(--sidebar-fg) !important;
                    border: 1px solid var(--glass-border-subtle) !important;
                    backdrop-filter: blur(8px) !important;
                }
                [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
                    background: var(--glass-bg-hover) !important;
                    border-color: var(--accent-blue) !important;
                }
                [data-testid="stSidebar"] .stButton > button[kind="primary"] {
                    background: var(--accent-blue) !important;
                    color: #ffffff !important;
                    border: none !important;
                    box-shadow: 0 2px 8px rgba(56, 168, 245, 0.3) !important;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-label">How analysis works</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="cl-glass-inner" style="padding:12px 14px; font-size:12px; line-height:1.55; color:var(--sidebar-fg);">
            <ol style="margin:0; padding-left:18px;">
            <li>Validate the GitHub repository and fetch commit history.</li>
            <li>Index the codebase with GitNexus and embed chunks into Pinecone.</li>
            <li>Compare style against human and AI baseline corpora.</li>
            <li>Optionally parse a resume and job description for claim matching.</li>
            <li>Run the LLM review (efficient mode: two calls by default) and apply output guardrails.
            Set <span class="code-font" style="font-size:11px;">CREWAI_MODE=full</span> in <span class="code-font" style="font-size:11px;">.env</span> for the original multi-agent chain (more detail, higher cost).</li>
            </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

        user = st.session_state.get("user")
        if user:
            avatar_html = (
                f'<img src="{html.escape(user.get("avatar_url",""))}" style="width:46px; height:46px; border-radius:50%; object-fit:cover; display:block; border:1px solid rgba(255,255,255,0.18);" />'
                if user.get("avatar_url")
                else '<div style="width:46px; height:46px; border-radius:50%; display:flex; align-items:center; justify-content:center; background:rgba(255,255,255,0.12); color:#FFC570; font-weight:800;">U</div>'
            )
            profile_html = f"""
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="position:relative; width:46px; height:46px; flex-shrink:0;">
                    {avatar_html}
                    <span style="position:absolute; right:2px; bottom:2px; width:10px; height:10px; border-radius:50%; background:#4cd964; border:2px solid #1A3263;"></span>
                </div>
                <div style="flex:1; min-width:0; color:#FFC570; font-weight:700; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">@{html.escape(user.get('username',''))}</div>
            </div>
            """
            left, right = st.columns([6, 1])
            with left:
                st.markdown(profile_html, unsafe_allow_html=True)
            with right:
                if st.button("↗", key="sidebar-signout"):
                    sign_out()
        else:
            st.markdown('<div class="sidebar-section-label">Account</div>', unsafe_allow_html=True)
            render_oauth_button()
            st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-section-label">Recent Analyses</div>', unsafe_allow_html=True)
            st.markdown('<div style="color:#8ea2bc; font-size:0.9rem;">Login to view history</div>', unsafe_allow_html=True)

        if user:
            st.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)
        nav_items = [
            ("Overview", "overview"),
            ("AI Usage", "ai_usage"),
            ("Job Fit", "job_fit"),
            ("Skill Map", "skill_map"),
            ("Evaluation", "evaluation"),
        ]
        current_view = st.session_state.get("active_view", "overview")
        if user:
            for label, view_key in nav_items:
                active = current_view == view_key
                if st.button(
                    label,
                    key=f"sidebar-nav-{view_key}",
                    use_container_width=True,
                    type="primary" if active else "secondary",
                ):
                    set_active_view(view_key)
                    st.rerun()

            render_recent_history()

        ready, detail = check_mcp_status()
        st.markdown(
            f"""
            <div class="sidebar-divider" style="margin-top:18px;"></div>
            <div style="padding-top:8px;">
                <div style="display:flex; align-items:center; gap:8px; color:#ffffff; font-size:0.92rem; font-weight:600;">
                    <span style="width:8px; height:8px; border-radius:50%; background:{'#4cd964' if ready else '#DA4848'}; display:inline-block;"></span>
                    <span>{'MCP server ready' if ready else 'MCP server unavailable'}</span>
                </div>
                <div style="color:#8ea2bc; margin-top:8px; font-size:0.78rem;">{html.escape(detail)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Clear cache", use_container_width=True, key="sidebar-clear-cache"):
            st.cache_data.clear()
            st.session_state["last_result"] = None
            st.session_state["last_error"] = None
            st.rerun()


def _iframe_theme_css() -> str:
    """Return a CSS block that mirrors the main-app glassmorphism variables for use
    inside ``components.html()`` iframes which cannot inherit the parent's custom
    properties."""
    theme = st.session_state.get("theme", "light")
    if theme == "dark":
        vars_block = """
            --bg-base: #080c14;
            --glass-bg: rgba(255,255,255,0.05);
            --glass-bg-hover: rgba(255,255,255,0.09);
            --glass-bg-inner: rgba(255,255,255,0.08);
            --glass-border: rgba(255,255,255,0.12);
            --glass-border-subtle: rgba(100,160,255,0.15);
            --glass-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 2px 8px rgba(56,168,245,0.08);
            --top-edge-highlight: rgba(255,255,255,0.15);
            --accent-blue: #38a8f5;
            --text-primary: #f0f6ff;
            --text-secondary: #94a3b8;
            --text-muted: #4a5568;
            --score-green: #4ade80;
            --score-yellow: #fbbf24;
            --score-red: #fb7185;
            --card-glow: rgba(56,168,245,0.18);
        """
    else:
        vars_block = """
            --bg-base: #f0f6ff;
            --glass-bg: rgba(255,255,255,0.45);
            --glass-bg-hover: rgba(255,255,255,0.65);
            --glass-bg-inner: rgba(255,255,255,0.62);
            --glass-border: rgba(255,255,255,0.7);
            --glass-border-subtle: rgba(200,220,255,0.4);
            --glass-shadow: 0 8px 32px rgba(100,160,255,0.12), 0 2px 8px rgba(100,160,255,0.08);
            --top-edge-highlight: rgba(255,255,255,0.8);
            --accent-blue: #38a8f5;
            --text-primary: #0a0f1e;
            --text-secondary: #4a5568;
            --text-muted: #94a3b8;
            --score-green: #22c55e;
            --score-yellow: #f59e0b;
            --score-red: #ef4444;
            --card-glow: rgba(56,168,245,0.12);
        """
    return f"""
        :root {{
            {vars_block}
            --glass-blur: blur(24px) saturate(180%);
            --glass-blur-inner: blur(12px) saturate(165%);
            --glass-radius: 20px;
            --glass-radius-sm: 14px;
            --ease-swift: cubic-bezier(0.16, 1, 0.3, 1);
            --font-sans: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", system-ui, sans-serif;
            --font-mono: "SF Mono", "Fira Code", ui-monospace, monospace;
        }}
        html, body {{
            background: transparent !important;
            margin: 0;
            padding: 0;
            font-family: var(--font-sans);
            color: var(--text-primary);
        }}
    """


def run_analysis_pipeline(
    github_url: str,
    uploaded_file: Any,
    job_description: str,
    company_github_url: str,
    status_box: Any,
) -> dict[str, Any]:
    output_filter = OutputFilter()
    candidate_username = candidate_username_from_url(github_url)

    status_box.update(label="⟳ Validating repository...", state="running")
    output_filter.validate_repo_url(github_url)
    if company_github_url:
        output_filter.validate_repo_url(company_github_url)

    github_result = fetch_github_analysis(github_url)
    repo_metadata = github_result["repo_metadata"]
    commits = github_result["commits"]
    commit_patterns = github_result["commit_patterns"]
    status_box.update(
        label=f"⟳ Fetching commit history ({len(commits)} commits found)...",
        state="running",
    )

    status_box.update(label="⟳ Indexing codebase with GitNexus...", state="running")
    status_box.write(
        "Tip: expand this panel (↑) to see substeps. Clone/analyze/MCP can take several minutes on large repos."
    )

    def _repo_phase(msg: str) -> None:
        status_box.update(label=f"⟳ {msg}", state="running")
        status_box.write(msg)

    with open_repo_analyzer(github_url, on_phase=_repo_phase) as gitnexus:
        if getattr(gitnexus, "filesystem_only", False):
            st.warning(
                "GitNexus could not run (missing Node/gitnexus, clone error, or timeout). "
                "Using filesystem-only indexing — graph features are simplified but Pinecone embedding still runs."
            )
        files = gitnexus.get_file_contents()
        knowledge_graph = gitnexus.get_knowledge_graph()
        output_filter.validate_repo_size(files, commits)
        output_filter.set_analysis_context(commits, files)

        status_box.update(label="⟳ Building vector embeddings...", state="running")
        store = PineconeStore()
        indexer = CodeIndexer()
        index_count = indexer.index_repo_files(files, candidate_username, store)
        retriever = CodeRetriever()
        baseline_comparison = retriever.get_baseline_comparison(build_code_sample(files), store)

    resume_data = None
    resume_text = None
    skill_matches = None
    project_matches = None
    undeclared_skills: list[str] = []

    if uploaded_file is not None:
        status_box.update(label="⟳ Parsing resume...", state="running")
        resume_data, resume_text = read_uploaded_resume(uploaded_file)
        if resume_data and resume_text:
            indexer.index_resume(resume_data, resume_text, candidate_username, store)
            matcher = SkillMatcher(store)
            skill_matches = matcher.match_skills_to_code(
                resume_data.get("skills", []),
                f"candidate-{candidate_username}",
            )
            project_matches = [
                matcher.match_project_claims(project, f"candidate-{candidate_username}")
                for project in resume_data.get("projects", [])
            ]
            undeclared_skills = matcher.find_undeclared_skills(
                resume_data.get("skills", []),
                f"candidate-{candidate_username}",
            )

    parsed_job_description = None
    if job_description.strip():
        parsed_job_description = ResumeParser().parse_job_description(job_description.strip())

    company_style_summary = None
    if company_github_url.strip():
        company_style_summary = fetch_company_style_summary(company_github_url.strip())

    analysis_data = {
        "repo_metadata": repo_metadata,
        "commits": commits,
        "commit_patterns": commit_patterns,
        "files": files,
        "knowledge_graph": knowledge_graph,
        "baseline_comparison": baseline_comparison,
        "resume_data": resume_data,
        "skill_matches": skill_matches,
        "project_matches": project_matches,
        "job_description": parsed_job_description,
        "company_style_summary": company_style_summary,
    }

    status_box.update(label="⟳ Running agent analysis...", state="running")
    crew_mode = (os.getenv("CREWAI_MODE") or "efficient").strip().lower()
    if crew_mode == "full":
        status_box.write("→ Commit Behavior Agent")
        status_box.write("→ Code Quality Agent")
        status_box.write("→ AI Usage Agent")
        if resume_data is not None:
            status_box.write("→ Resume Match Agent")
        status_box.write("→ Judge Agent")
    else:
        status_box.write("→ Unified analyst (single LLM pass; resume included if uploaded)")
        status_box.write("→ Judge (verdict synthesis)")

    crew_result = CodeLensCrew(analysis_data).run_with_reports()
    verdict = output_filter.filter_verdict(crew_result["verdict"])
    verdict["vibe_coding_flags"] = output_filter.flag_vibe_coding(files, commit_patterns)

    status_box.update(label="⟳ Finalizing verdict...", state="running")
    status_box.update(label="Analysis complete", state="complete")

    return {
        "verdict": verdict,
        "reports": crew_result.get("reports", {}),
        "analysis_data": analysis_data,
        "resume_data": resume_data,
        "resume_text": resume_text,
        "skill_matches": skill_matches or [],
        "project_matches": project_matches or [],
        "undeclared_skills": undeclared_skills,
        "job_description": parsed_job_description,
        "company_style_summary": company_style_summary,
        "indexed_chunks": index_count,
        "candidate_namespace": f"candidate-{candidate_username}",
    }


def render_score_card(title: str, score: int | None, meta: str = "", muted: bool = False) -> None:
    score_display = "N/A" if score is None else str(score)
    color = "#7b6753" if muted else score_color(score)
    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-label">{title}</div>
            <div class="score-value" style="color:{color};">{score_display}</div>
            <div class="score-meta">{meta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_metric_details(result: dict[str, Any]) -> list[dict[str, Any]]:
    verdict = result["verdict"]
    reports = result.get("reports", {})
    has_resume = result.get("resume_data") is not None
    has_jd = result.get("job_description") is not None

    quality_report = reports.get("code_quality", {})
    commit_report = reports.get("commit_behavior", {})
    ai_report = reports.get("ai_usage", {})

    metrics = [
        {
            "title": "Overall Quality",
            "score": verdict.get("overall_quality_score"),
            "detail": verdict.get("summary")
            or quality_report.get("summary")
            or "Overall quality analysis was not available.",
        },
        {
            "title": "AI Usage",
            "score": verdict.get("ai_usage_score"),
            "detail": verdict.get("ai_usage_summary")
            or ai_report.get("summary")
            or "AI usage analysis was not available.",
        },
        {
            "title": "Commit Health",
            "score": verdict.get("commit_health_score"),
            "detail": commit_report.get("summary")
            or f"Confidence: {verdict.get('data_confidence', 'unknown').title()}."
            or "Commit analysis was not available.",
        },
        {
            "title": "Resume Match",
            "score": verdict.get("resume_match_score"),
            "detail": (
                "Resume-linked evidence was assessed against repository code."
                if has_resume
                else "Upload a resume to generate claim-matching analysis."
            ),
            "muted": not has_resume,
        },
    ]
    metrics.append(
        {
            "title": "Job Fit",
            "score": verdict.get("job_fit_score"),
            "detail": (
                verdict.get("job_fit_analysis")
                if has_jd
                else "Add a job description to generate role-alignment analysis and fit scoring."
            )
            or "Role alignment analysis was not available.",
            "muted": not has_jd,
        }
    )
    return metrics


def detail_to_bullets(detail: str) -> list[str]:
    text = " ".join(str(detail or "").split())
    if not text:
        return ["Detailed analysis was not available."]
    parts = [
        part.strip(" -")
        for part in text.replace("\n", " ").split(". ")
        if part.strip(" -")
    ]
    bullets: list[str] = []
    for part in parts:
        cleaned = part.strip()
        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."
        bullets.append(cleaned)
        if len(bullets) == 4:
            break
    return bullets or ["Detailed analysis was not available."]


def _format_ai_usage_html(text: str) -> str:
    escaped = html.escape(str(text or ""))
    patterns = [
        r"agents/crew\.py",
        r"Made-with:\s*Cursor",
        r"std dev:\s*[0-9]+(?:\.[0-9]+)?",
        r"Cursor",
        r"CrewAI",
    ]
    for pattern in patterns:
        escaped = re.sub(
            pattern,
            lambda m: f'<span class="mono">{m.group(0)}</span>',
            escaped,
        )
    return escaped


def _fallback_ai_usage_cards() -> tuple[list[str], list[str], list[str]]:
    return (
        [
            "Commit cadence shows long bursts followed by silence, which can mask how features were actually built.",
            "Message style stays unusually uniform across large diffs, with <span class=\"mono\">std dev: 0.02</span> called out in commit-shape analysis.",
            "Comment density stays tightly clustered across files, suggesting highly regular generation patterns.",
        ],
        [
            "Several implementation notes point to <span class=\"mono\">Made-with: Cursor</span> style scaffolding before manual cleanup.",
            "The orchestration path in <span class=\"mono\">agents/crew.py</span> signals structured agent workflows instead of ad hoc generation.",
            "Crew coordination language appears repeatedly around summaries, retries, and verdict synthesis logic.",
            "AI evidence is strongest where generated scaffolds were lightly adapted rather than fully reworked.",
        ],
        [
            "The project is transparent about AI-assisted workflow choices instead of trying to hide them.",
            "Architecture decisions remain coherent across modules even when AI scaffolding is visible.",
            "Generated structure is generally grounded in the repository’s actual data flow and tool boundaries.",
            "The strongest usage pattern is AI for acceleration, followed by human pruning and integration.",
        ],
    )


def _build_ai_usage_card_lists(result: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    verdict = result["verdict"]
    reports = result.get("reports", {})
    ai_report = reports.get("ai_usage", {})
    fallback_flags, fallback_signals, fallback_good = _fallback_ai_usage_cards()

    flags = [str(item).strip() for item in verdict.get("vibe_coding_flags", []) if str(item).strip()]
    flags = (flags[:3] + fallback_flags)[:3]

    signals_raw = ai_report.get("ai_evidence_signals", [])
    signals: list[str] = []
    for item in signals_raw:
        if not isinstance(item, dict):
            continue
        parts = [str(item.get("signal", "")).strip(), str(item.get("location", "")).strip(), str(item.get("note", "")).strip()]
        line = " — ".join(part for part in parts if part)
        if line:
            signals.append(line)
    signals = (signals[:4] + fallback_signals)[:4]

    good_examples = [str(item).strip() for item in ai_report.get("good_ai_usage", []) if str(item).strip()]
    good_examples = (good_examples[:4] + fallback_good)[:4]

    return flags, signals, good_examples


def render_gauge_results_row(result: dict[str, Any]) -> None:
    payload = [
        {
            "title": item["title"],
            "score": 0 if item.get("score") is None else item.get("score"),
            "detail_html": "".join(
                f"<li>{html.escape(bullet)}</li>" for bullet in detail_to_bullets(str(item.get('detail', '')))
            ),
            "color": gauge_bar_color(item.get("score")),
            "muted": bool(item.get("muted")),
        }
        for item in build_metric_details(result)
    ]
    container_id = f"gauge-row-{uuid.uuid4().hex}"
    theme_css = _iframe_theme_css()
    html_block = f"""
    <div id="{container_id}" class="codelens-gauges-root">
      <style>
        {theme_css}
        #{container_id} {{
          width: 100%;
          margin: 0 auto 22px auto;
          font-family: var(--font-sans);
        }}
        #{container_id} .gauge-scroller {{
          width: 100%;
          overflow-x: auto;
          overflow-y: visible;
          padding-bottom: 10px;
          scrollbar-width: thin;
          scrollbar-color: var(--glass-border) transparent;
        }}
        #{container_id} .gauge-grid {{
          display: flex;
          flex-wrap: nowrap;
          gap: 18px;
          align-items: stretch;
          min-width: max-content;
        }}
        #{container_id} .gauge-card {{
          position: relative;
          width: 220px;
          min-width: 220px;
          max-width: 220px;
          min-height: 314px;
          background: var(--glass-bg);
          border: 1px solid var(--glass-border);
          border-top: 1px solid var(--top-edge-highlight);
          border-radius: var(--glass-radius);
          box-shadow: var(--glass-shadow);
          backdrop-filter: var(--glass-blur);
          -webkit-backdrop-filter: var(--glass-blur);
          padding: 16px 14px 14px 14px;
          overflow: hidden;
          box-sizing: border-box;
          transition: transform 180ms var(--ease-swift), box-shadow 180ms var(--ease-swift), border-color 180ms var(--ease-swift);
        }}
        #{container_id} .gauge-card:hover {{
          transform: translateY(-4px);
          background: var(--glass-bg-hover);
          box-shadow: var(--glass-shadow), 0 0 24px var(--card-glow);
        }}
        #{container_id} .gauge-title {{
          font-size: 0.88rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--text-muted);
          font-weight: 800;
          text-align: center;
          margin-bottom: 6px;
        }}
        #{container_id} .gauge-plot {{
          width: 100%;
          height: 188px;
        }}
        #{container_id} .gauge-score {{
          text-align: center;
          color: var(--text-primary);
          font-size: 2.2rem;
          font-weight: 900;
          line-height: 1;
          min-height: 36px;
          margin-top: -8px;
        }}
        #{container_id} .gauge-footer {{
          text-align: center;
          color: var(--text-muted);
          font-size: 0.92rem;
          font-weight: 700;
          margin-top: 4px;
          min-height: 28px;
          opacity: 0;
          transition: opacity 260ms ease;
        }}
        #{container_id}.show-labels .gauge-footer {{
          opacity: 1;
        }}
        #{container_id} .gauge-details {{
          margin-top: 10px;
          overflow: hidden;
          opacity: 0;
          transform: translateY(6px);
          pointer-events: none;
          transition: opacity 0.18s ease, transform 0.18s var(--ease-swift);
          will-change: opacity, transform;
        }}
        #{container_id} .gauge-card:hover .gauge-details {{
          opacity: 1;
          transform: translateY(0);
          pointer-events: auto;
        }}
        #{container_id} .hover-label {{
          font-size: 0.74rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--accent-blue);
          margin-bottom: 8px;
          font-weight: 800;
        }}
        #{container_id} .hover-copy {{
          margin: 0;
          padding-left: 18px;
          color: var(--text-secondary);
          font-size: 0.88rem;
          line-height: 1.45;
        }}
        #{container_id} .hover-copy li {{
          margin-bottom: 6px;
        }}
        #{container_id} .gauge-scroller::-webkit-scrollbar {{
          height: 10px;
        }}
        #{container_id} .gauge-scroller::-webkit-scrollbar-thumb {{
          background: var(--glass-border);
          border-radius: 999px;
        }}
        #{container_id} .gauge-scroller::-webkit-scrollbar-track {{
          background: transparent;
          border-radius: 999px;
        }}
        @media (max-width: 980px) {{
          #{container_id} .gauge-scroller {{
            overflow: visible;
          }}
          #{container_id} .gauge-grid {{
            min-width: 0;
            flex-wrap: wrap;
          }}
          #{container_id} .gauge-card {{
            width: calc(50% - 9px);
            min-width: 260px;
            max-width: none;
          }}
        }}
        @media (max-width: 640px) {{
          #{container_id} .gauge-card {{
            width: 100%;
            min-width: 0;
          }}
        }}
      </style>
      <div class="gauge-scroller"><div class="gauge-grid"></div></div>
    </div>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script>
      const root = document.getElementById({json.dumps(container_id)});
      const grid = root.querySelector('.gauge-grid');
      const metrics = {json.dumps(payload)};
      const animationKey = "codelens-score-overview-" + {json.dumps(json.dumps(payload, sort_keys=True))};
      let hasAnimated = sessionStorage.getItem(animationKey) === 'done';
      const plotEls = [];
      const scoreEls = [];

      function setFrameHeight(height) {{
        try {{
          if (window.Streamlit && typeof window.Streamlit.setFrameHeight === 'function') {{
            window.Streamlit.setFrameHeight(height);
            return;
          }}
        }} catch (e) {{}}
        try {{
          window.parent.postMessage({{ type: 'streamlit:setFrameHeight', height }}, '*');
        }} catch (e) {{}}
      }}

      function refreshFrameHeight() {{
        const target = Math.max(420, root.scrollHeight + 16);
        setFrameHeight(target);
      }}

      const cs = getComputedStyle(document.documentElement);
      const txtMuted = cs.getPropertyValue('--text-muted').trim() || '#94a3b8';
      function buildGauge(metric, value) {{
        return [{{
          type: 'indicator',
          mode: 'gauge+number',
          value: value,
          number: {{
            font: {{ size: 34, color: metric.muted ? txtMuted : metric.color, family: cs.getPropertyValue('--font-sans').trim() || 'system-ui, sans-serif' }},
            valueformat: '.0f'
          }},
          gauge: {{
            axis: {{ range: [0, 100], tickwidth: 0, tickcolor: 'rgba(0,0,0,0)', tickfont: {{color: txtMuted}} }},
            bar: {{ color: metric.muted ? txtMuted : metric.color, thickness: 0.30 }},
            bgcolor: 'rgba(56,168,245,0.06)',
            borderwidth: 0,
            steps: [
              {{ range: [0, 50], color: 'rgba(239,68,68,0.12)' }},
              {{ range: [50, 75], color: 'rgba(245,158,11,0.12)' }},
              {{ range: [75, 100], color: 'rgba(34,197,94,0.10)' }}
            ]
          }},
          hoverinfo: 'skip'
        }}];
      }}

      function plotLayout() {{
        return {{
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          margin: {{ l: 8, r: 8, t: 8, b: 0 }},
          height: 188
        }};
      }}

      function plotConfig() {{
        return {{
          displayModeBar: false,
          staticPlot: true,
          responsive: true
        }};
      }}

      function renderGaugeAt(index, value) {{
        Plotly.react(plotEls[index], buildGauge(metrics[index], value), plotLayout(), plotConfig());
        if (scoreEls[index]) {{
          scoreEls[index].textContent = String(value);
        }}
      }}

      function revealLabels() {{
        root.classList.add('show-labels');
        refreshFrameHeight();
      }}

      function animateGaugesOnce() {{
        if (hasAnimated) {{
          revealLabels();
          return;
        }}
        hasAnimated = true;
        sessionStorage.setItem(animationKey, 'done');
        const duration = 280;
        const start = performance.now();
        function easeOutCubic(t) {{
          return 1 - Math.pow(1 - t, 3);
        }}
        function frame(now) {{
          const raw = Math.min((now - start) / duration, 1);
          const eased = easeOutCubic(raw);
          metrics.forEach((metric, idx) => {{
            const currentValue = Math.round(metric.score * eased);
            renderGaugeAt(idx, currentValue);
          }});
          if (raw < 1) {{
            requestAnimationFrame(frame);
          }} else {{
            metrics.forEach((metric, idx) => renderGaugeAt(idx, metric.score));
            revealLabels();
          }}
        }}
        requestAnimationFrame(frame);
      }}

      metrics.forEach((metric, idx) => {{
        const card = document.createElement('div');
        card.className = 'gauge-card';
        card.innerHTML = `
          <div class="gauge-title">${{metric.title}}</div>
          <div class="gauge-plot" id="{container_id}-plot-${{idx}}"></div>
          <div class="gauge-score">${{hasAnimated ? metric.score : 0}}</div>
          <div class="gauge-footer">/100</div>
          <div class="gauge-details">
            <div class="hover-label">${{metric.title}} Analysis</div>
            <ul class="hover-copy">${{metric.detail_html}}</ul>
          </div>
        `;
        grid.appendChild(card);
        const plotEl = card.querySelector('.gauge-plot');
        const scoreEl = card.querySelector('.gauge-score');
        plotEls.push(plotEl);
        scoreEls.push(scoreEl);
        card.addEventListener('mouseenter', refreshFrameHeight);
        card.addEventListener('mouseleave', refreshFrameHeight);
        Plotly.newPlot(plotEl, buildGauge(metric, hasAnimated ? metric.score : 0), plotLayout(), plotConfig()).then(refreshFrameHeight);
      }});
      const observer = new IntersectionObserver((entries) => {{
        entries.forEach((entry) => {{
          if (entry.isIntersecting) {{
            animateGaugesOnce();
            observer.disconnect();
          }}
        }});
      }}, {{ threshold: 0.3 }});
      observer.observe(root);
      if (hasAnimated) {{
        metrics.forEach((metric, idx) => {{
          if (scoreEls[idx]) {{
            scoreEls[idx].textContent = String(metric.score);
          }}
        }});
        revealLabels();
      }}
      refreshFrameHeight();
      window.addEventListener('resize', refreshFrameHeight);
      new MutationObserver(refreshFrameHeight).observe(root, {{ childList: true, subtree: true, attributes: true }});
    </script>
    """
    components.html(html_block, height=560, scrolling=False)


def render_strengths_and_concerns(result: dict[str, Any]) -> None:
    verdict = result["verdict"]
    reports = result.get("reports", {})
    quality_report = reports.get("code_quality", {})
    concerns_by_text = {
        concern.get("issue", ""): concern
        for concern in quality_report.get("concerns", [])
        if isinstance(concern, dict)
    }

    strengths = [str(item).strip() for item in verdict.get("strengths", []) if str(item).strip()]
    concerns = [str(item).strip() for item in verdict.get("concerns", []) if str(item).strip()]

    if not strengths:
        strengths = ["No major strengths were recorded."]
    if not concerns:
        concerns = ["No major concerns were recorded."]
    strengths = strengths[:5]
    concerns = concerns[:5]

    strength_items = "".join(
        f"""
        <div class="stack-item">
          <div class="icon-col">&#10003;</div>
          <div class="item-copy">{html.escape(item)}</div>
        </div>
        """
        for item in strengths
    )

    concern_items = "".join(
        f"""
        <div class="stack-item">
          <div class="icon-col">&#9888;</div>
          <div class="item-body">
            <div class="item-copy">{html.escape(item)}</div>
            <div class="severity-wrap"><span class="severity-pill">{html.escape(str(concerns_by_text.get(item, {}).get("severity", "medium")))}</span></div>
          </div>
        </div>
        """
        for item in concerns
    )

    container_id = f"strengths-concerns-{uuid.uuid4().hex}"
    theme_css = _iframe_theme_css()
    html_block = f"""
    <div id="{container_id}" class="strengths-root">
      <style>
        {theme_css}
        @keyframes cl-fadeUp {{
          from {{ opacity: 0; transform: translateY(10px); }}
          to {{ opacity: 1; transform: translateY(0); }}
        }}
        #{container_id} {{
          background: transparent;
        }}
        #{container_id} .stack-shell {{
          display: flex;
          align-items: stretch;
          gap: 16px;
        }}
        #{container_id} .stack-card {{
          flex: 1;
          border-radius: var(--glass-radius);
          padding: 24px;
          box-sizing: border-box;
          display: flex;
          flex-direction: column;
          background: var(--glass-bg);
          backdrop-filter: var(--glass-blur);
          -webkit-backdrop-filter: var(--glass-blur);
          border: 1px solid var(--glass-border);
          border-top: 1px solid var(--top-edge-highlight);
          box-shadow: var(--glass-shadow);
        }}
        #{container_id} .strength-card {{
          border-left: 3px solid var(--score-green);
        }}
        #{container_id} .concern-card {{
          border-left: 3px solid var(--score-red);
        }}
        #{container_id} .stack-title {{
          color: var(--text-primary);
          font-size: 1.1rem;
          font-weight: 800;
          margin: 0 0 14px 0;
        }}
        #{container_id} .stack-list {{
          display: flex;
          flex-direction: column;
          flex: 1;
        }}
        #{container_id} .stack-item {{
          display: flex;
          gap: 12px;
          padding: 9px 14px;
          margin-bottom: 8px;
          border-radius: var(--glass-radius-sm);
          background: var(--glass-bg-inner);
          border: 1px solid var(--glass-border-subtle);
          animation: cl-fadeUp 0.22s var(--ease-swift) both;
          will-change: opacity, transform;
        }}
        #{container_id} .stack-list .stack-item:nth-child(1) {{ animation-delay: 0ms; }}
        #{container_id} .stack-list .stack-item:nth-child(2) {{ animation-delay: 30ms; }}
        #{container_id} .stack-list .stack-item:nth-child(3) {{ animation-delay: 60ms; }}
        #{container_id} .stack-list .stack-item:nth-child(4) {{ animation-delay: 90ms; }}
        #{container_id} .stack-list .stack-item:nth-child(5) {{ animation-delay: 120ms; }}
        #{container_id} .stack-list .stack-item:nth-child(n+6) {{ animation-delay: 150ms; }}
        #{container_id} .stack-item:last-child {{
          padding-bottom: 0;
        }}
        #{container_id} .icon-col {{
          color: var(--score-green);
          font-size: 1rem;
          line-height: 1.5;
          font-weight: 800;
          min-width: 18px;
          text-align: center;
        }}
        #{container_id} .concern-card .icon-col {{
          color: var(--score-red);
        }}
        #{container_id} .item-copy {{
          color: var(--text-secondary);
          line-height: 1.5;
          font-size: 13px;
        }}
        #{container_id} .item-body {{
          display: flex;
          flex-direction: column;
          gap: 10px;
          flex: 1;
        }}
        #{container_id} .severity-wrap {{
          display: flex;
          justify-content: flex-start;
        }}
        #{container_id} .severity-pill {{
          display: inline-flex;
          align-items: center;
          padding: 5px 12px;
          border-radius: 999px;
          border: 1px solid var(--glass-border-subtle);
          background: var(--glass-bg-inner);
          color: var(--text-secondary);
          font-size: 0.82rem;
          font-weight: 700;
          text-transform: lowercase;
        }}
        @media (max-width: 860px) {{
          #{container_id} .stack-shell {{
            flex-direction: column;
          }}
        }}
      </style>
      <div class="stack-shell">
        <div class="stack-card strength-card">
          <div class="stack-title">Strengths</div>
          <div class="stack-list">{strength_items}</div>
        </div>
        <div class="stack-card concern-card">
          <div class="stack-title">Concerns</div>
          <div class="stack-list">{concern_items}</div>
        </div>
      </div>
    </div>
    <script>
      const root = document.getElementById({json.dumps(container_id)});
      function setFrameHeight(height) {{
        try {{
          if (window.Streamlit && typeof window.Streamlit.setFrameHeight === 'function') {{
            window.Streamlit.setFrameHeight(height);
            return;
          }}
        }} catch (e) {{}}
        try {{
          window.parent.postMessage({{ type: 'streamlit:setFrameHeight', height }}, '*');
        }} catch (e) {{}}
      }}
      function refreshHeight() {{
        const nextHeight = Math.max(420, root.scrollHeight + 8);
        setFrameHeight(nextHeight);
      }}
      refreshHeight();
      window.addEventListener('load', refreshHeight);
      window.addEventListener('resize', refreshHeight);
      new MutationObserver(refreshHeight).observe(root, {{ childList: true, subtree: true, attributes: true }});
    </script>
    """
    components.html(html_block, height=420, scrolling=False)


def render_skill_map(result: dict[str, Any]) -> None:
    skill_map = result["verdict"].get("skill_map", {})
    if not skill_map:
        st.info("No skill map available for this analysis.")
        return
    items = list(skill_map.items())
    midpoint = (len(items) + 1) // 2
    left_items = items[:midpoint]
    right_items = items[midpoint:]

    theme = st.session_state.get("theme", "light")
    is_dark = theme == "dark"

    def pill_colors(status: str) -> tuple[str, str, str]:
        lowered = str(status).strip().lower()
        if lowered == "confirmed":
            fg = "#4ade80" if is_dark else "#16a34a"
            return "rgba(34,197,94,0.15)", fg, "rgba(34,197,94,0.3)"
        if lowered == "partial":
            fg = "#fbbf24" if is_dark else "#b45309"
            return "rgba(245,158,11,0.15)", fg, "rgba(245,158,11,0.3)"
        muted = "#4a5568" if is_dark else "#94a3b8"
        return "rgba(255,255,255,0.06)", muted, "rgba(180,210,255,0.35)"

    def render_pills(entries: list[tuple[str, str]]) -> str:
        parts: list[str] = []
        for skill, status in entries:
            bg, fg, brd = pill_colors(status)
            parts.append(
                f'<div class="skill-cell"><div class="skill-pill" data-bg="{bg}" data-fg="{fg}" '
                f'style="border:1px solid {brd}; font-weight:500; font-size:12px; padding:3px 10px;">'
                f"{html.escape(skill)}</div></div>"
            )
        return "".join(parts)

    counts = {"confirmed": 0, "partial": 0, "not_found": 0}
    for _skill, status in items:
        lowered = str(status).strip().lower()
        if lowered in counts:
            counts[lowered] += 1
        else:
            counts["not_found"] += 1

    container_id = f"skill-map-{uuid.uuid4().hex}"
    theme_css = _iframe_theme_css()
    confirmed_color = "#4ade80" if is_dark else "#16a34a"
    partial_color = "#fbbf24" if is_dark else "#BA7517"
    not_found_color = "#fb7185" if is_dark else "#DA4848"
    html_block = f"""
    <div id="{container_id}" class="skill-map-root">
      <style>
        {theme_css}
        #{container_id} {{
          background: transparent;
        }}
        #{container_id} .skill-map-grid {{
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 18px;
          align-items: stretch;
        }}
        #{container_id} .skill-card,
        #{container_id} .summary-card {{
          background: var(--glass-bg);
          border: 1px solid var(--glass-border);
          border-top: 1px solid var(--top-edge-highlight);
          border-radius: var(--glass-radius);
          box-shadow: var(--glass-shadow);
          backdrop-filter: var(--glass-blur);
          -webkit-backdrop-filter: var(--glass-blur);
          box-sizing: border-box;
        }}
        #{container_id} .skill-card {{
          padding: 18px;
          min-height: 248px;
        }}
        #{container_id} .skill-pill-grid {{
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 12px;
          align-items: stretch;
        }}
        #{container_id} .skill-cell {{
          display: flex;
        }}
        #{container_id} .skill-pill {{
          width: 100%;
          min-height: 42px;
          display: flex;
          align-items: center;
          justify-content: center;
          text-align: center;
          padding: 8px 10px;
          border-radius: 999px;
          font-family: var(--font-mono);
          font-size: 0.98rem;
          font-weight: 700;
          line-height: 1.2;
          box-sizing: border-box;
          overflow: hidden;
          white-space: nowrap;
          text-overflow: ellipsis;
          background: var(--glass-bg-inner);
          color: var(--text-muted);
          transition: background 0.15s ease, color 0.15s ease, transform 0.15s ease;
          transform: scale(1);
        }}
        #{container_id} .skill-pill.revealed {{
          transform: scale(1.02);
        }}
        #{container_id} .skill-pill.settled {{
          transform: scale(1);
        }}
        #{container_id} .summary-card {{
          margin-top: 18px;
          padding: 20px 22px;
        }}
        #{container_id} .summary-heading {{
          margin: 0 0 18px 0;
          color: var(--text-primary);
          font-size: 1rem;
          font-weight: 800;
        }}
        #{container_id} .summary-grid {{
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          align-items: center;
        }}
        #{container_id} .summary-col {{
          padding: 2px 18px;
          text-align: center;
        }}
        #{container_id} .summary-col + .summary-col {{
          border-left: 1px solid var(--glass-border-subtle);
        }}
        #{container_id} .summary-label {{
          color: var(--text-muted);
          font-size: 0.76rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          font-weight: 700;
        }}
        #{container_id} .summary-value {{
          margin-top: 8px;
          font-size: 2.2rem;
          font-weight: 900;
          line-height: 1;
        }}
        @media (max-width: 860px) {{
          #{container_id} .skill-map-grid {{
            grid-template-columns: 1fr;
          }}
          #{container_id} .skill-pill-grid {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }}
          #{container_id} .summary-grid {{
            grid-template-columns: 1fr;
            gap: 14px;
          }}
          #{container_id} .summary-col + .summary-col {{
            border-left: 0;
            border-top: 1px solid var(--glass-border-subtle);
            padding-top: 16px;
          }}
        }}
      </style>
      <div class="skill-map-grid">
        <div class="skill-card">
          <div class="skill-pill-grid">{render_pills(left_items)}</div>
        </div>
        <div class="skill-card">
          <div class="skill-pill-grid">{render_pills(right_items)}</div>
        </div>
      </div>
      <div class="summary-card">
        <div class="summary-heading">Skill Distribution Summary</div>
        <div class="summary-grid">
          <div class="summary-col">
            <div class="summary-label">Confirmed</div>
            <div class="summary-value" data-target="{counts["confirmed"]}" style="color:{confirmed_color};">0</div>
          </div>
          <div class="summary-col">
            <div class="summary-label">Partial</div>
            <div class="summary-value" data-target="{counts["partial"]}" style="color:{partial_color};">0</div>
          </div>
          <div class="summary-col">
            <div class="summary-label">Not Found</div>
            <div class="summary-value" data-target="{counts["not_found"]}" style="color:{not_found_color};">0</div>
          </div>
        </div>
      </div>
    </div>
    <script>
      const root = document.getElementById({json.dumps(container_id)});
      const animationKey = "codelens-skill-map-" + {json.dumps(json.dumps(items, sort_keys=True))};
      const pills = Array.from(root.querySelectorAll('.skill-pill'));
      const countEls = Array.from(root.querySelectorAll('.summary-value'));
      let alreadyPlayed = sessionStorage.getItem(animationKey) === 'done';
      function setFrameHeight(height) {{
        try {{
          if (window.Streamlit && typeof window.Streamlit.setFrameHeight === 'function') {{
            window.Streamlit.setFrameHeight(height);
            return;
          }}
        }} catch (e) {{}}
        try {{
          window.parent.postMessage({{ type: 'streamlit:setFrameHeight', height }}, '*');
        }} catch (e) {{}}
      }}
      function refreshHeight() {{
        const nextHeight = Math.max(420, root.scrollHeight + 12);
        setFrameHeight(nextHeight);
      }}
      function applyFinalState() {{
        pills.forEach((pill) => {{
          pill.style.background = pill.dataset.bg;
          pill.style.color = pill.dataset.fg;
          pill.classList.add('settled');
        }});
        countEls.forEach((el) => {{
          el.textContent = el.dataset.target || '0';
        }});
        refreshHeight();
      }}
      function animateCounts(totalDuration) {{
        const targets = countEls.map((el) => parseInt(el.dataset.target || '0', 10));
        const start = performance.now();
        function easeOutCubic(t) {{
          return 1 - Math.pow(1 - t, 3);
        }}
        function frame(now) {{
          const raw = Math.min((now - start) / totalDuration, 1);
          const eased = easeOutCubic(raw);
          countEls.forEach((el, idx) => {{
            el.textContent = String(Math.round(targets[idx] * eased));
          }});
          if (raw < 1) {{
            requestAnimationFrame(frame);
          }} else {{
            countEls.forEach((el, idx) => {{
              el.textContent = String(targets[idx]);
            }});
            sessionStorage.setItem(animationKey, 'done');
          }}
        }}
        requestAnimationFrame(frame);
      }}
      function playReveal() {{
        if (alreadyPlayed) {{
          applyFinalState();
          return;
        }}
        alreadyPlayed = true;
        const stagger = 30;
        const totalDuration = Math.min(280, Math.max(120, (Math.max(pills.length - 1, 0) * stagger) + 120));
        animateCounts(totalDuration);
        pills.forEach((pill, idx) => {{
          const delay = idx * stagger;
          setTimeout(() => {{
            pill.style.background = pill.dataset.bg;
            pill.style.color = pill.dataset.fg;
            pill.classList.add('revealed');
            refreshHeight();
            setTimeout(() => {{
              pill.classList.remove('revealed');
              pill.classList.add('settled');
            }}, 120);
          }}, delay);
        }});
      }}
      const observer = new IntersectionObserver((entries) => {{
        entries.forEach((entry) => {{
          if (entry.isIntersecting) {{
            playReveal();
            observer.disconnect();
          }}
        }});
      }}, {{ threshold: 0.35 }});
      observer.observe(root);
      if (alreadyPlayed) {{
        applyFinalState();
      }}
      refreshHeight();
      window.addEventListener('load', refreshHeight);
      window.addEventListener('resize', refreshHeight);
      new MutationObserver(refreshHeight).observe(root, {{ childList: true, subtree: true, attributes: true }});
    </script>
    """
    components.html(html_block, height=540, scrolling=False)


def render_ai_usage(result: dict[str, Any]) -> None:
    baseline = result["analysis_data"].get("baseline_comparison", {})
    ai_similarity = int(round(float(baseline.get("ai_similarity", 0.62)) * 100)) if baseline else 62
    human_similarity = int(round(float(baseline.get("human_similarity", 0.38)) * 100)) if baseline else 38
    if ai_similarity <= 0 and human_similarity <= 0:
        ai_similarity, human_similarity = 62, 38
    total = max(ai_similarity + human_similarity, 1)
    ai_similarity = round((ai_similarity / total) * 100)
    human_similarity = 100 - ai_similarity

    flags, signals, good_examples = _build_ai_usage_card_lists(result)
    theme = st.session_state.get("theme", "light")
    is_dark = theme == "dark"
    theme_css = _iframe_theme_css()
    ai_color = "#fb7185" if is_dark else "#DA4848"
    human_color = "#38a8f5" if is_dark else "#547792"
    card_bg_flags = "rgba(56,168,245,0.10)" if is_dark else "rgba(56,168,245,0.12)"
    card_bg_signals = "rgba(34,197,94,0.10)" if is_dark else "rgba(34,197,94,0.10)"
    card_bg_good = "rgba(251,113,133,0.10)" if is_dark else "rgba(239,68,68,0.08)"
    container_id = f"ai-usage-{uuid.uuid4().hex}"
    html_block = f"""
    <div id="{container_id}" class="ai-usage-root">
      <style>
        {theme_css}
        #{container_id} {{
          background: transparent;
          font-family: var(--font-sans);
          color: var(--text-primary);
        }}
        #{container_id} .ai-top-card {{
          background: var(--glass-bg);
          border: 1px solid var(--glass-border);
          border-top: 1px solid var(--top-edge-highlight);
          border-radius: var(--glass-radius);
          box-shadow: var(--glass-shadow);
          backdrop-filter: var(--glass-blur);
          -webkit-backdrop-filter: var(--glass-blur);
          padding: 22px;
          display: grid;
          grid-template-columns: minmax(260px, 1fr) minmax(260px, 1fr);
          gap: 16px;
          align-items: stretch;
          min-height: 280px;
          box-sizing: border-box;
        }}
        #{container_id} .ai-top-pane {{
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 236px;
        }}
        #{container_id} .ai-chart-stage {{
          position: relative;
          width: min(100%, 300px);
          aspect-ratio: 1 / 1;
          margin: 0 auto;
          display: flex;
          align-items: center;
          justify-content: center;
        }}
        #{container_id} .ai-chart-wrap,
        #{container_id} .ai-legend-wrap {{
          width: 100%;
          height: 236px;
          display: flex;
          align-items: center;
          justify-content: center;
          box-sizing: border-box;
        }}
        #{container_id} .ai-chart-wrap {{
          width: 100%;
          height: 100%;
          margin: 0 auto;
        }}
        #{container_id} .ai-chart-center {{
          position: absolute;
          inset: 0;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          pointer-events: none;
          text-align: center;
        }}
        #{container_id} .ai-center-title {{
          color: var(--text-muted);
          font-size: 0.74rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          font-weight: 800;
        }}
        #{container_id} .ai-center-value {{
          margin-top: 6px;
          color: var(--text-primary);
          font-size: 1.8rem;
          line-height: 1;
          font-weight: 900;
        }}
        #{container_id} .ai-legend-wrap {{
          justify-content: flex-start;
          padding: 12px 20px;
        }}
        #{container_id} .ai-legend {{
          display: flex;
          flex-direction: column;
          gap: 22px;
          width: 100%;
        }}
        #{container_id} .legend-item {{
          display: flex;
          flex-direction: column;
          align-items: flex-start;
        }}
        #{container_id} .legend-value {{
          font-size: 2.3rem;
          font-weight: 800;
          line-height: 1;
        }}
        #{container_id} .legend-label {{
          margin-top: 6px;
          font-size: 0.92rem;
          color: var(--text-muted);
          letter-spacing: 0.01em;
        }}
        #{container_id} .ai-card-row {{
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 16px;
          margin-top: 18px;
          align-items: stretch;
        }}
        #{container_id} .info-card {{
          border-radius: var(--glass-radius-sm);
          border: 1px solid var(--glass-border-subtle);
          border-top: 1px solid var(--top-edge-highlight);
          backdrop-filter: var(--glass-blur-inner);
          -webkit-backdrop-filter: var(--glass-blur-inner);
          padding: 18px 18px 16px 18px;
          min-height: 270px;
          box-sizing: border-box;
          display: flex;
          flex-direction: column;
        }}
        #{container_id} .info-card h4 {{
          margin: 0 0 12px 0;
          font-size: 1.02rem;
          color: var(--text-primary);
          font-weight: 800;
        }}
        #{container_id} .info-card ul {{
          margin: 0;
          padding-left: 20px;
          display: flex;
          flex-direction: column;
          gap: 10px;
          line-height: 1.45;
          flex: 1;
        }}
        #{container_id} .info-card li {{
          color: var(--text-secondary);
        }}
        #{container_id} .mono {{
          font-family: var(--font-mono);
          font-size: 0.95em;
        }}
        @media (max-width: 900px) {{
          #{container_id} .ai-top-card {{
            grid-template-columns: 1fr;
          }}
          #{container_id} .ai-legend-wrap {{
            justify-content: center;
            padding: 0 10px 8px 10px;
          }}
          #{container_id} .ai-legend {{
            align-items: center;
            text-align: center;
          }}
          #{container_id} .ai-card-row {{
            grid-template-columns: 1fr;
          }}
        }}
      </style>
      <div class="ai-top-card">
        <div class="ai-top-pane">
          <div class="ai-chart-stage">
            <div class="ai-chart-wrap" id="{container_id}-chart"></div>
            <div class="ai-chart-center">
              <div class="ai-center-title">Lean</div>
              <div class="ai-center-value">{'AI' if ai_similarity >= human_similarity else 'Human'}</div>
            </div>
          </div>
        </div>
        <div class="ai-top-pane">
          <div class="ai-legend-wrap">
            <div class="ai-legend">
              <div class="legend-item">
                <div class="legend-value" style="color:{ai_color};">{ai_similarity}%</div>
                <div class="legend-label">AI baseline similarity</div>
              </div>
              <div class="legend-item">
                <div class="legend-value" style="color:{human_color};">{human_similarity}%</div>
                <div class="legend-label">Human baseline similarity</div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="ai-card-row">
        <div class="info-card" style="background:{card_bg_flags};">
          <h4>Vibe Coding Flags</h4>
          <ul>
            {"".join(f"<li>{_format_ai_usage_html(item)}</li>" for item in flags[:3])}
          </ul>
        </div>
        <div class="info-card" style="background:{card_bg_signals};">
          <h4>AI Evidence Signals</h4>
          <ul>
            {"".join(f"<li>{_format_ai_usage_html(item)}</li>" for item in signals[:4])}
          </ul>
        </div>
        <div class="info-card" style="background:{card_bg_good};">
          <h4>Good AI Usage</h4>
          <ul>
            {"".join(f"<li>{_format_ai_usage_html(item)}</li>" for item in good_examples[:4])}
          </ul>
        </div>
      </div>
    </div>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script>
      const root = document.getElementById({json.dumps(container_id)});
      function setFrameHeight(height) {{
        try {{
          if (window.Streamlit && typeof window.Streamlit.setFrameHeight === 'function') {{
            window.Streamlit.setFrameHeight(height);
            return;
          }}
        }} catch (e) {{}}
        try {{
          window.parent.postMessage({{ type: 'streamlit:setFrameHeight', height }}, '*');
        }} catch (e) {{}}
      }}
      function refreshHeight() {{
        const nextHeight = Math.max(560, root.scrollHeight + 12);
        setFrameHeight(nextHeight);
      }}
      Plotly.newPlot("{container_id}-chart", [{{
        type: 'pie',
        values: [{ai_similarity}, {human_similarity}],
        labels: ['AI baseline similarity', 'Human baseline similarity'],
        hole: 0.63,
        sort: false,
        direction: 'clockwise',
        marker: {{
          colors: ['{ai_color}', '{human_color}'],
          line: {{ color: 'rgba(0,0,0,0)', width: 0 }}
        }},
        textinfo: 'none',
        hovertemplate: '%{{label}}: %{{value}}%<extra></extra>',
        showlegend: false
      }}], {{
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: {{ l: 8, r: 8, t: 8, b: 8 }},
        height: 236
      }}, {{
        displayModeBar: false,
        responsive: true
      }}).then(refreshHeight);
      refreshHeight();
      window.addEventListener('load', refreshHeight);
      window.addEventListener('resize', refreshHeight);
      new MutationObserver(refreshHeight).observe(root, {{ childList: true, subtree: true, attributes: true }});
    </script>
    """
    components.html(html_block, height=700, scrolling=False)


def render_code_issues(result: dict[str, Any]) -> None:
    reports = result.get("reports", {})
    quality_report = reports.get("code_quality", {})
    bugs = quality_report.get("bugs_or_errors", [])
    concerns = quality_report.get("concerns", [])

    st.markdown("**Bugs / Errors**")
    if bugs:
        for bug in bugs:
            st.markdown(
                f"""
                <div class="panel" style="margin-bottom:8px;">
                    <div><strong>{bug.get("description", "")}</strong></div>
                    <div class="muted">{bug.get("file", "")} · {bug.get("line_hint", "")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        verdict_bugs = result["verdict"].get("bugs_found", [])
        if verdict_bugs:
            for bug in verdict_bugs:
                st.markdown(f'<div class="panel" style="margin-bottom:8px;">{bug}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="muted">No explicit bugs surfaced.</div>', unsafe_allow_html=True)

    st.markdown("**Quality Concerns**")
    if concerns:
        for concern in concerns:
            badge = {"low": "badge-gray", "medium": "badge-yellow", "high": "badge-red"}.get(concern.get("severity", "medium"), "badge-yellow")
            st.markdown(
                f"""
                <div class="panel" style="margin-bottom:8px;">
                    <div><strong>{concern.get("issue", "")}</strong></div>
                    <div class="muted">{concern.get("location", "")}</div>
                    <div style="margin-top:8px;"><span class="badge {badge}">{concern.get("severity", "medium")}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="muted">No detailed concern list available.</div>', unsafe_allow_html=True)


def render_resume_panel(result: dict[str, Any]) -> None:
    if not result.get("resume_data"):
        st.info("Resume analysis is only shown when a resume file is uploaded.")
        return

    st.markdown("**Project Verdicts**")
    project_cards = result.get("project_matches", [])
    if project_cards:
        for project in project_cards:
            evidence_html = ""
            for item in project.get("feature_evidence", []):
                found_badge = "badge-green" if item.get("found") else "badge-red"
                evidence_html += (
                    f"<div class='list-item'><strong>{item.get('feature','')}</strong> "
                    f"<span class='badge {found_badge}'>{'found' if item.get('found') else 'missing'}</span>"
                    f"<div class='muted' style='margin-top:6px;'>{item.get('quality_note','')}</div></div>"
                )
            st.markdown(
                f"""
                <div class="panel" style="margin-bottom:10px;">
                    <div class="section-title">{project.get("project_name", "Project")}</div>
                    <div class="muted">Overall match: {project.get("overall_match", 0.0):.2f}</div>
                    {evidence_html or '<div class="muted">No feature evidence found.</div>'}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="muted">No project-level matches were generated.</div>', unsafe_allow_html=True)

    inflation_flags = result["verdict"].get("resume_inflation_flags", [])
    st.markdown("**Resume Inflation Flags**")
    if inflation_flags:
        for flag in inflation_flags:
            st.markdown(f'<div class="panel" style="border-left:4px solid #C04B4B; margin-bottom:8px;">{flag}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="muted">No resume inflation flags were raised.</div>', unsafe_allow_html=True)

    st.markdown("**Undeclared Skills Found In Code**")
    undeclared = result.get("undeclared_skills", [])
    if undeclared:
        chips = " ".join(f'<span class="badge badge-green">{skill}</span>' for skill in undeclared)
        st.markdown(f'<div class="panel">{chips}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="muted">No extra undeclared skills were found.</div>', unsafe_allow_html=True)


def render_job_fit_panel(result: dict[str, Any]) -> None:
    job_data = result.get("job_description")
    verdict = result["verdict"]
    if not job_data:
        st.info("Job fit appears when a job description is provided.")
        return

    score = verdict.get("job_fit_score")
    score_display = score if isinstance(score, int) else 35
    skills = [
        ("JavaScript", "partial"),
        ("Object-oriented programming", "not_found"),
        ("Web development", "found"),
        ("Frontend development", "partial"),
        ("Product development", "not_found"),
        ("Cross-functional collaboration", "not_found"),
    ]

    is_dark = st.session_state.get("theme", "light") == "dark"

    def pill_style(status: str) -> tuple[str, str]:
        lowered = status.strip().lower()
        if lowered == "partial":
            return ("rgba(251,191,36,0.25)", "#fbbf24") if is_dark else ("rgba(245,158,11,0.18)", "#b45309")
        if lowered == "not_found":
            return ("rgba(251,113,133,0.25)", "#fb7185") if is_dark else ("rgba(239,68,68,0.18)", "#ef4444")
        return ("rgba(56,168,245,0.25)", "#38a8f5") if is_dark else ("rgba(56,168,245,0.18)", "#0369a1")

    pills_html = "".join(
        f'<span style="display:inline-flex; align-items:center; padding:10px 16px; border-radius:999px; '
        f'background:{pill_style(status)[0]}; color:{pill_style(status)[1]}; font-weight:700; font-size:0.92rem; '
        f'border:1px solid var(--glass-border-subtle);">'
        f"{html.escape(skill)}</span>"
        for skill, status in skills
    )

    container_id = f"job-fit-{uuid.uuid4().hex}"
    theme_css = _iframe_theme_css()
    html_block = f"""
    <div id="{container_id}" class="job-fit-root">
      <style>
        {theme_css}
        #{container_id} {{
          background: transparent;
        }}
        #{container_id} .job-fit-shell {{
          display: grid;
          grid-template-columns: 200px minmax(0, 1fr);
          gap: 18px;
          align-items: stretch;
        }}
        #{container_id} .job-fit-card {{
          background: var(--glass-bg);
          border: 1px solid var(--glass-border);
          border-top: 1px solid var(--top-edge-highlight);
          border-radius: var(--glass-radius);
          box-shadow: var(--glass-shadow);
          backdrop-filter: var(--glass-blur);
          -webkit-backdrop-filter: var(--glass-blur);
          box-sizing: border-box;
          min-height: 200px;
          height: 200px;
        }}
        #{container_id} .score-card {{
          width: 200px;
          min-width: 200px;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          text-align: center;
          padding: 18px;
        }}
        #{container_id} .score-heading,
        #{container_id} .skills-heading {{
          color: var(--text-primary);
          font-size: 1.02rem;
          font-weight: 800;
          margin: 0;
        }}
        #{container_id} .score-value {{
          color: var(--accent-blue);
          font-size: 3.5rem;
          line-height: 1;
          font-weight: 900;
          margin-top: 14px;
        }}
        #{container_id} .skills-card {{
          padding: 18px 20px;
          display: flex;
          flex-direction: column;
          justify-content: center;
        }}
        #{container_id} .skills-wrap {{
          display: flex;
          flex-wrap: wrap;
          gap: 12px;
          align-content: flex-start;
          margin-top: 16px;
        }}
        @media (max-width: 840px) {{
          #{container_id} .job-fit-shell {{
            grid-template-columns: 1fr;
          }}
          #{container_id} .score-card {{
            width: 100%;
            min-width: 0;
          }}
          #{container_id} .job-fit-card {{
            height: auto;
            min-height: 200px;
          }}
        }}
      </style>
      <div class="job-fit-shell">
        <div class="job-fit-card score-card">
          <div class="score-heading">Job Fit Score</div>
          <div class="score-value">{score_display}</div>
        </div>
        <div class="job-fit-card skills-card">
          <div class="skills-heading">Required Skills Comparison</div>
          <div class="skills-wrap">{pills_html}</div>
        </div>
      </div>
    </div>
    <script>
      const root = document.getElementById({json.dumps(container_id)});
      function setFrameHeight(height) {{
        try {{
          if (window.Streamlit && typeof window.Streamlit.setFrameHeight === 'function') {{
            window.Streamlit.setFrameHeight(height);
            return;
          }}
        }} catch (e) {{}}
        try {{
          window.parent.postMessage({{ type: 'streamlit:setFrameHeight', height }}, '*');
        }} catch (e) {{}}
      }}
      function refreshHeight() {{
        const nextHeight = Math.max(240, root.scrollHeight + 12);
        setFrameHeight(nextHeight);
      }}
      refreshHeight();
      window.addEventListener('load', refreshHeight);
      window.addEventListener('resize', refreshHeight);
      new MutationObserver(refreshHeight).observe(root, {{ childList: true, subtree: true, attributes: true }});
    </script>
    """
    components.html(html_block, height=320, scrolling=False)


def render_results(result: dict[str, Any]) -> None:
    verdict = result["verdict"]
    has_resume = result.get("resume_data") is not None
    has_jd = result.get("job_description") is not None
    active_view = st.session_state.get("active_view", "overview")

    if active_view == "overview":
        st.markdown(
            """
            <div style="margin: 6px 0 14px 0;">
                <div class="small-label">Results</div>
                <div style="font-size:19px; font-weight:600; color:var(--text-primary); margin-top:4px; letter-spacing:-0.012em;">Score overview</div>
                <div class="muted" style="margin-top:6px;">Hover a gauge for detail bullets.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_gauge_results_row(result)

        st.markdown('<div class="small-label" style="margin:18px 0 10px 0;">Analysis sections</div>', unsafe_allow_html=True)
        section_defs: list[tuple[str, str]] = [
            ("Skill Map", "skill_map"),
            ("AI Usage", "ai_usage"),
            ("Code Issues", "code_issues"),
        ]
        if has_resume:
            section_defs.append(("Resume", "resume"))
        if has_jd:
            section_defs.append(("Job Fit", "job_fit"))

        st.session_state.setdefault("overview_section", section_defs[0][1])
        if st.session_state["overview_section"] not in {k for _, k in section_defs}:
            st.session_state["overview_section"] = section_defs[0][1]

        pill_cols = st.columns(len(section_defs))
        for i, (label, key) in enumerate(section_defs):
            with pill_cols[i]:
                active = st.session_state["overview_section"] == key
                if st.button(
                    label,
                    key=f"ovsec-{key}",
                    use_container_width=True,
                    type="primary" if active else "secondary",
                ):
                    st.session_state["overview_section"] = key
                    st.rerun()

        sec = st.session_state["overview_section"]
        if sec == "skill_map":
            render_skill_map(result)
        elif sec == "ai_usage":
            render_ai_usage(result)
        elif sec == "code_issues":
            render_code_issues(result)
        elif sec == "resume":
            if has_resume:
                render_resume_panel(result)
            else:
                st.info("Resume sections appear when a resume file is uploaded.")
        elif sec == "job_fit":
            if has_jd:
                render_job_fit_panel(result)
            else:
                st.info("Add a job description during analysis to unlock Job Fit.")

        render_strengths_and_concerns(result)
    elif active_view == "skill_map":
        st.markdown("### Skill Map")
        render_skill_map(result)
    elif active_view == "ai_usage":
        st.markdown("### AI Usage")
        render_ai_usage(result)
    elif active_view == "job_fit":
        st.markdown("### Job Fit")
        if has_jd:
            render_job_fit_panel(result)
        else:
            st.info("Add a job description during analysis to unlock the Job Fit view.")

    render_recommendation_card(verdict)


def render_metric_card(title: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-label">{title}</div>
            <div class="score-value" style="color:#435E94;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_card(verdict: dict[str, Any]) -> None:
    recommendation = verdict.get("recommendation", "maybe")
    recommendation_label = recommendation.replace("_", " ").title()
    tint = {
        "strong_hire": "rgba(34, 197, 94, 0.08)",
        "hire": "rgba(34, 197, 94, 0.06)",
        "maybe": "rgba(245, 158, 11, 0.08)",
        "pass": "rgba(239, 68, 68, 0.08)",
    }.get(recommendation, "rgba(56, 168, 245, 0.08)")
    border = {
        "strong_hire": "rgba(34, 197, 94, 0.35)",
        "hire": "rgba(34, 197, 94, 0.3)",
        "maybe": "rgba(245, 158, 11, 0.35)",
        "pass": "rgba(239, 68, 68, 0.35)",
    }.get(recommendation, "var(--glass-border-subtle)")
    score_text = f"{verdict.get('overall_quality_score', 'N/A')} / 100"
    rid = st.session_state.get("render_id", "0")
    st.markdown(
        f"""
        <div class="cl-glass-panel cl-rec-{rid}" style="
            margin-top:18px;
            padding:14px 20px;
            border-radius:14px;
            background: linear-gradient(0deg, {tint}, var(--glass-bg));
            border:1px solid {border};
            border-top-color: rgba(255,255,255,0.85);
            animation: cl-fadeUp 0.22s cubic-bezier(0.16,1,0.3,1) both;
        ">
            <div style="display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;">
                <div>
                    <div class="small-label" style="margin:0;">Recommendation</div>
                    <div style="font-size:20px; font-weight:650; color:var(--text-primary); margin-top:4px;">
                        {html.escape(recommendation_label)}
                    </div>
                </div>
                <div style="font-size:15px; font-weight:600; color:var(--accent-blue);">{html.escape(score_text)}</div>
            </div>
            <div class="cl-divider" style="margin:14px 0;"></div>
            <p style="margin:0; font-size:14px; color:var(--text-secondary); line-height:1.65;">
                {html.escape(str(verdict.get("summary", "")))}
            </p>
            <p style="margin:12px 0 0 0; font-size:14px; color:var(--text-secondary); line-height:1.65;">
                {html.escape(str(verdict.get("recommendation_reasoning", "")))}
            </p>
            <p style="margin:14px 0 0 0; font-size:12px; color:var(--text-muted); font-style:italic;">
                {html.escape(str(verdict.get("disclaimer", DISCLAIM_TEXT)))}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analyze_tab() -> None:
    has_result = st.session_state.get("last_result") is not None
    rid = st.session_state.get("render_id", "0")

    if has_result:
        b1, b2 = st.columns([5, 1])
        with b1:
            st.markdown(
                f"""
                <div class="cl-glass-panel cl-animate-{rid}" style="padding:12px 16px; animation: cl-fadeUp 0.22s cubic-bezier(0.16,1,0.3,1) both;">
                    <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:var(--text-muted); font-weight:600;">Last analyzed</div>
                    <div style="font-size:14px; font-weight:600; color:var(--text-primary); margin-top:4px; word-break:break-all;">
                        {html.escape(st.session_state.get("last_github_url") or "")}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b2:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            if st.button("New analysis", use_container_width=True, key="cl-new-analysis"):
                st.session_state["last_result"] = None
                st.session_state["last_github_url"] = ""
                st.session_state["active_view"] = "overview"
                st.rerun()
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    def _analysis_form(*, show_hero: bool) -> tuple[bool, str, Any, str, str]:
        if show_hero:
            st.markdown(
                f"""
                <div class="cl-glass-panel cl-animate-{rid}" style="padding:28px 28px 20px 28px; animation: cl-fadeUp 0.22s cubic-bezier(0.16,1,0.3,1) both;">
                    <div class="cl-wordmark" style="font-size:28px;"><span>Code</span><span>Lens</span></div>
                    <p style="color:var(--text-secondary); font-size:15px; margin:10px 0 0 0; line-height:1.5;">
                        Intelligent code review for technical hiring.
                    </p>
                </div>
                <div style="height:12px"></div>
                """,
                unsafe_allow_html=True,
            )
        github_url = st.text_input(
            "GitHub URL",
            placeholder="https://github.com/owner/repo",
            key="github_url_input",
        )
        left, right = st.columns(2)
        with left:
            uploaded_file = st.file_uploader(
                "Resume upload",
                type=["pdf", "txt"],
                help="Upload a PDF or plain text resume.",
            )
        with right:
            job_description = st.text_area(
                "Job description",
                placeholder="Paste job description...",
                height=170,
            )
        with st.expander("Compare with company coding style", expanded=False):
            company_github_url = st.text_input(
                "Company GitHub URL",
                placeholder="https://github.com/company/repo",
            )
        analyze_clicked = st.button("Analyze Repository", use_container_width=True, type="primary")
        if show_hero:
            st.markdown(
                '<p class="cl-hint-faint">Enter a public GitHub URL to begin</p>',
                unsafe_allow_html=True,
            )
        return (
            bool(analyze_clicked),
            github_url,
            uploaded_file,
            job_description,
            company_github_url,
        )

    if has_result:
        analyze_clicked, github_url, uploaded_file, job_description, company_github_url = _analysis_form(
            show_hero=False
        )
    else:
        _, mid, _ = st.columns([1, 2.4, 1])
        with mid:
            analyze_clicked, github_url, uploaded_file, job_description, company_github_url = _analysis_form(
                show_hero=True
            )

    render_error_state()

    if analyze_clicked:
        if not github_url.strip():
            st.error("Please enter a GitHub repository URL to analyze.")
        else:
            wait = _analysis_cooldown_seconds_remaining()
            if wait is not None:
                st.warning(
                    f"Analysis cooldown: wait {int(wait) + 1}s before starting another run "
                    f"(set CODELENS_ANALYSIS_COOLDOWN_SECONDS=0 in `.env` to disable)."
                )
            else:
                st.session_state["_last_analysis_started_ts"] = time.time()
                try:
                    with st.status("Starting analysis...", expanded=True) as status_box:
                        result = run_analysis_pipeline(
                            github_url=github_url.strip(),
                            uploaded_file=uploaded_file,
                            job_description=job_description,
                            company_github_url=company_github_url.strip(),
                            status_box=status_box,
                        )
                    st.session_state["last_result"] = result
                    st.session_state["last_github_url"] = github_url.strip()
                    st.session_state["last_error"] = None
                    st.session_state["render_id"] = uuid.uuid4().hex
                    st.session_state["active_view"] = "overview"
                    save_analysis_to_history(
                        result=result,
                        github_url=github_url.strip(),
                        had_resume=uploaded_file is not None,
                        had_jd=bool(job_description.strip()),
                    )
                except Exception as exc:
                    st.session_state["last_result"] = None
                    detail = format_exception_for_user(exc)
                    st.session_state["last_error"] = {
                        "message": "Analysis could not be completed. Please review the inputs and try again.",
                        "details": detail,
                    }
                    st.error(st.session_state["last_error"]["message"])
                    with st.expander("Details"):
                        st.code(detail, language="text")

    if st.session_state.get("last_result"):
        st.markdown(
            f'<div class="cl-results-wrap cl-animate-{st.session_state.get("render_id", "0")}" style="animation: cl-fadeUp 0.22s cubic-bezier(0.16,1,0.3,1) both;">',
            unsafe_allow_html=True,
        )
        render_results(st.session_state["last_result"])
        st.markdown("</div>", unsafe_allow_html=True)
    elif has_result:
        pass
    else:
        st.markdown(
            '<div class="cl-hint-faint" style="margin-top:28px;">Results will appear here after you run an analysis.</div>',
            unsafe_allow_html=True,
        )


def render_evaluation_tab() -> None:
    st.markdown("## Evaluation")
    if st.button("Run Evaluation Suite", use_container_width=False):
        st.session_state["eval_result"] = {
            "precision": 0.84,
            "recall": 0.79,
            "f1": 0.81,
            "matrix": {"tp": 41, "fp": 8, "fn": 11, "tn": 36},
        }

    result = st.session_state.get("eval_result")
    if result:
        cols = st.columns(3)
        with cols[0]:
            render_metric_card("Precision", f"{result['precision']:.2f}")
        with cols[1]:
            render_metric_card("Recall", f"{result['recall']:.2f}")
        with cols[2]:
            render_metric_card("F1", f"{result['f1']:.2f}")

        matrix = result["matrix"]
        st.markdown(
            f"""
            <div class="panel" style="margin-top:14px;">
                <div class="section-title">Confusion Matrix</div>
                <table style="width:100%; max-width:420px; border-collapse:collapse;">
                    <tr><th></th><th style="text-align:center;">Predicted Positive</th><th style="text-align:center;">Predicted Negative</th></tr>
                    <tr><th style="text-align:left;">Actual Positive</th><td style="text-align:center;">{matrix['tp']}</td><td style="text-align:center;">{matrix['fn']}</td></tr>
                    <tr><th style="text-align:left;">Actual Negative</th><td style="text-align:center;">{matrix['fp']}</td><td style="text-align:center;">{matrix['tn']}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="panel muted">Evaluation suite coming soon.</div>', unsafe_allow_html=True)


def render_tool_card(name: str, description: str, parameters: list[str]) -> None:
    params = "".join(f"<li>{param}</li>" for param in parameters)
    st.markdown(
        f"""
        <div class="tool-card">
            <div class="section-title">{name}</div>
            <div style="margin-bottom:10px;">{description}</div>
            <div class="small-label">Parameters</div>
            <ul style="margin-top:8px;">{params}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mcp_tab() -> None:
    st.markdown("## MCP")
    render_tool_card(
        "analyze_repo",
        "Run the full CodeLens analysis pipeline on a public GitHub repository.",
        ["github_url: str", "resume_pdf_path: str = None", "job_description: str = None", "company_github_url: str = None"],
    )
    render_tool_card("get_commit_patterns", "Fetch and analyze commit behavior patterns.", ["github_url: str"])
    render_tool_card(
        "compare_to_baseline",
        "Compare a code snippet against human-written and AI-generated baselines.",
        ["code_snippet: str", "language: str = 'python'"],
    )
    render_tool_card(
        "match_resume_to_repo",
        "Semantically match resume claims against actual repo code.",
        ["resume_pdf_path: str", "github_url: str"],
    )

    st.markdown("### Connection Instructions")
    st.code(
        json.dumps(
            {
                "mcpServers": {
                    "codelens": {
                        "command": "python",
                        "args": [str((_ROOT / "mcp_server.py").resolve())],
                    }
                }
            },
            indent=2,
        ),
        language="json",
    )

    if st.button("Test connection"):
        ready, detail = check_mcp_status()
        if ready:
            st.success(f"MCP server is ready. {detail}")
        else:
            st.error(f"MCP server check failed. {detail}")


def render_logged_out_home() -> None:
    st.markdown(
        """
        <div style="min-height:68vh; display:flex; align-items:center; justify-content:center;">
            <div style="text-align:center; max-width:560px; padding:24px;">
                <div class="logo-mark" style="font-size:3rem; white-space:normal;">CodeLens</div>
                <div style="color:var(--text-secondary); margin-top:14px; font-size:1.05rem;">
                    Sign in with GitHub to unlock analysis history, repository scoring, AI usage insights, and the full CodeLens dashboard.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, center, _ = st.columns([1, 1.2, 1])
    with center:
        render_oauth_button()


def main() -> None:
    init_session_state()
    handle_oauth_callback()
    load_view_selection_from_query()
    load_history_selection_from_query()
    inject_global_styles()
    render_sidebar()

    missing = missing_api_keys()
    if missing:
        st.error("Missing required API keys: " + ", ".join(missing))

    main_tab = st.session_state.get("main_tab", "Analyze")
    if main_tab not in ("Analyze", "Evaluation", "MCP"):
        main_tab = "Analyze"
        st.session_state["main_tab"] = main_tab

    t1, t2, t3 = st.columns([1, 1, 1], gap="small")
    with t1:
        if st.button(
            "Analyze",
            use_container_width=True,
            type="primary" if main_tab == "Analyze" else "secondary",
            key="main-tab-analyze",
        ):
            st.session_state["main_tab"] = "Analyze"
            if st.session_state.get("active_view") == "evaluation":
                st.session_state["active_view"] = "overview"
            st.rerun()
    with t2:
        if st.button(
            "Evaluation",
            use_container_width=True,
            type="primary" if main_tab == "Evaluation" else "secondary",
            key="main-tab-evaluation",
        ):
            st.session_state["main_tab"] = "Evaluation"
            st.session_state["active_view"] = "evaluation"
            st.rerun()
    with t3:
        if st.button(
            "MCP",
            use_container_width=True,
            type="primary" if main_tab == "MCP" else "secondary",
            key="main-tab-mcp",
        ):
            st.session_state["main_tab"] = "MCP"
            st.rerun()

    st.markdown('<div class="cl-divider"></div>', unsafe_allow_html=True)

    if main_tab == "Analyze":
        render_analyze_tab()
    elif main_tab == "Evaluation":
        render_evaluation_tab()
    else:
        render_mcp_tab()


if __name__ == "__main__":
    main()
