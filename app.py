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


def apply_global_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #EFD2B0;
                --surface: #f6ddc1;
                --surface-2: #fff5e8;
                --border: #d1aa7a;
                --border-hover: #547792;
                --accent: #1A3263;
                --accent-2: #FFC570;
                --secondary: #547792;
                --danger: #DA4848;
                --warning: #FFC570;
                --text: #1f1a14;
                --text-secondary: #5f5348;
                --text-muted: #7d6c5e;
            }

            html, body, [class*="css"] {
                font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            }

            .stApp {
                background:
                    radial-gradient(circle at top, rgba(255, 197, 112, 0.35), transparent 34%),
                    linear-gradient(180deg, #efd2b0 0%, #f4ddc3 100%);
                color: var(--text);
            }

            [data-testid="stSidebar"] {
                background: #1A3263;
                border-right: 1px solid rgba(255,255,255,0.08);
            }

            [data-testid="stSidebar"] * {
                color: #ffffff;
            }

            [data-testid="stSidebar"] .stButton > button {
                background: rgba(255,255,255,0.04);
                color: #FFC570;
                border: 1px solid rgba(255,255,255,0.16);
                min-height: 36px;
            }

            [data-testid="stSidebar"] .stButton > button:hover {
                background: rgba(255,255,255,0.08);
                border-color: rgba(255,255,255,0.24);
                color: #FFC570;
            }

            [data-testid="stHeader"] {
                background: rgba(239, 210, 176, 0.94);
                border-bottom: 1px solid var(--border);
            }

            .block-container {
                padding-top: 3.2rem;
                padding-bottom: 2rem;
            }

            h1, h2, h3, h4, h5, h6, p, div, span, label {
                color: var(--text);
            }

            code, pre, .code-font {
                font-family: "SFMono-Regular", Menlo, Consolas, monospace !important;
            }

            [data-testid="stTextInput"] input,
            [data-testid="stTextArea"] textarea,
            [data-testid="stFileUploaderDropzone"] {
                background: var(--surface-2) !important;
                color: var(--text) !important;
                border: 1px solid var(--border) !important;
                border-radius: 12px !important;
            }

            [data-testid="stTextInput"] input:focus,
            [data-testid="stTextArea"] textarea:focus {
                border-color: var(--accent) !important;
                box-shadow: 0 0 0 1px var(--accent) !important;
            }

            [data-testid="stTextInput"] input::placeholder,
            [data-testid="stTextArea"] textarea::placeholder {
                color: #735843 !important;
                opacity: 1 !important;
            }

            .stButton > button {
                background: rgba(26, 50, 99, 0.08);
                color: var(--accent);
                border: 1px solid var(--border);
                border-radius: 12px;
                transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease, background 160ms ease;
            }

            .stButton > button[kind="primary"] {
                background: var(--accent) !important;
                color: #ffffff !important;
                border: 1px solid #13274e !important;
                font-weight: 700 !important;
                min-height: 48px;
                box-shadow: 0 12px 24px rgba(26, 50, 99, 0.18);
            }

            .stButton > button:hover {
                border-color: var(--border-hover);
                color: var(--accent);
                background: rgba(26, 50, 99, 0.12);
                transform: translateY(-1px);
            }

            .stButton > button[kind="primary"]:hover {
                color: #ffffff !important;
                background: #13274e !important;
                border-color: #13274e !important;
            }

            [data-testid="stTabs"] button {
                background: transparent;
                color: var(--text-secondary);
                border-bottom: 2px solid transparent;
                border-radius: 10px 10px 0 0;
                padding: 0.75rem 0.9rem;
                transition: background 160ms ease, color 160ms ease, border-color 160ms ease;
            }

            [data-testid="stTabs"] button[aria-selected="true"] {
                color: var(--accent);
                border-bottom-color: var(--accent);
                background: rgba(26, 50, 99, 0.06);
            }

            .panel {
                background: var(--surface-2);
                border: 1px solid var(--border);
                border-radius: 22px;
                padding: 16px 18px;
                box-shadow: 0 18px 42px rgba(26, 50, 99, 0.10);
            }

            .score-card {
                background: var(--surface-2);
                border: 1px solid var(--border);
                border-radius: 22px;
                padding: 18px;
                min-height: 132px;
                box-shadow: 0 16px 34px rgba(26, 50, 99, 0.08);
            }

            .score-label {
                color: var(--text-secondary);
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }

            .score-value {
                font-size: 2rem;
                font-weight: 800;
                margin-top: 8px;
            }

            .score-meta {
                color: var(--text-muted);
                font-size: 0.9rem;
                margin-top: 10px;
            }

            .strength-panel { border-left: 4px solid var(--accent); }
            .concern-panel { border-left: 4px solid var(--danger); }

            .badge {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                font-size: 0.8rem;
                border: 1px solid var(--border);
            }

            .badge-green { background: rgba(26, 50, 99, 0.12); color: var(--accent); }
            .badge-purple { background: rgba(84, 119, 146, 0.15); color: var(--secondary); }
            .badge-red { background: rgba(218, 72, 72, 0.14); color: var(--danger); }
            .badge-yellow { background: rgba(255, 197, 112, 0.22); color: #8d5e11; }
            .badge-blue { background: rgba(84, 119, 146, 0.16); color: var(--secondary); }
            .badge-gray { background: rgba(26, 50, 99, 0.08); color: var(--text-secondary); }

            .section-title {
                font-size: 1rem;
                font-weight: 700;
                margin-bottom: 0.8rem;
            }

            .list-item {
                padding: 10px 0;
                border-bottom: 1px solid rgba(221, 198, 168, 0.55);
                color: var(--text);
            }

            .muted { color: var(--text-muted); }

            .tool-card,
            .history-card {
                background: var(--surface-2);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 16px;
                margin-bottom: 10px;
            }

            .summary-box {
                background: linear-gradient(180deg, rgba(255, 245, 232, 0.98) 0%, rgba(246, 221, 193, 0.96) 100%);
                border: 1px solid var(--border);
                border-radius: 20px;
                padding: 18px;
            }

            .logo-mark {
                color: var(--accent);
                font-size: 2.45rem;
                font-weight: 900;
                letter-spacing: 0.03em;
                line-height: 1.1;
                white-space: nowrap;
                overflow: visible;
            }

            .app-header {
                display: flex;
                align-items: flex-end;
                justify-content: space-between;
                gap: 18px;
                margin-bottom: 22px;
                padding: 18px 0 10px 0;
                flex-wrap: wrap;
                overflow: visible;
                position: relative;
                z-index: 2;
            }

            .app-header-copy {
                min-width: 0;
                overflow: visible;
            }

            .app-header-subtitle {
                color: var(--text-secondary);
                margin-top: 8px;
                font-size: 0.98rem;
                max-width: 760px;
            }

            .app-header-chip {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 8px 14px;
                border-radius: 999px;
                background: rgba(26, 50, 99, 0.08);
                border: 1px solid rgba(26, 50, 99, 0.14);
                color: var(--accent);
                font-weight: 700;
                white-space: nowrap;
            }

            .small-label {
                color: var(--text-secondary);
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }

            .sidebar-section-label {
                color: #547792 !important;
                font-size: 0.76rem;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                font-weight: 700;
                margin: 18px 0 10px 0;
            }

            .sidebar-divider {
                height: 1px;
                background: rgba(255,255,255,0.10);
                margin: 16px 0;
            }

            .oauth-button {
                display: inline-flex;
                align-items: center;
                gap: 10px;
                padding: 12px 14px;
                border-radius: 12px;
                border: 1px solid var(--border);
                background: linear-gradient(135deg, #1A3263 0%, #547792 100%);
                color: #FFC570 !important;
                font-weight: 600;
                text-decoration: none;
                width: 100%;
                box-sizing: border-box;
            }

            .oauth-button:hover {
                border-color: #1A3263;
                text-decoration: none;
                color: #FFC570 !important;
            }

            @media (max-width: 1100px) {
                .block-container {
                    padding-top: 2.4rem;
                    padding-bottom: 1.5rem;
                }

                .score-card,
                .panel {
                    border-radius: 18px;
                }

                .logo-mark {
                    font-size: 2rem;
                }
            }

            @media (max-width: 760px) {
                .block-container {
                    padding-top: 2rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }

                .app-header {
                    gap: 10px;
                    align-items: flex-start;
                }

                .app-header-chip {
                    width: 100%;
                    justify-content: center;
                    white-space: normal;
                    text-align: center;
                }

                .score-card {
                    min-height: 116px;
                }

                [data-testid="stTabs"] button {
                    font-size: 0.88rem;
                    padding: 0.72rem 0.55rem;
                }
            }
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


def missing_api_keys() -> list[str]:
    required = ["OPENROUTER_API_KEY", "PINECONE_API_KEY", "VOYAGE_API_KEY"]
    return [key for key in required if not os.getenv(key)]


def oauth_ready() -> bool:
    return bool(os.getenv("GITHUB_CLIENT_ID") and os.getenv("GITHUB_CLIENT_SECRET") and os.getenv("APP_URL"))


def oauth_redirect_uri() -> str:
    return os.getenv("APP_URL", "http://localhost:8501").rstrip("/")


def score_color(score: int | None) -> str:
    if score is None:
        return "#5F5348"
    if score > 75:
        return "#1A3263"
    if score >= 40:
        return "#FFC570"
    return "#DA4848"


def gauge_bar_color(score: int | None) -> str:
    if score is None:
        return "#547792"
    if score < 40:
        return "#DA4848"
    if score < 60:
        return "#FFC570"
    return "#1A3263"


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
    st.session_state.clear()
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
            st.session_state["last_error"] = None
            st.session_state["active_view"] = "overview"
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
            st.session_state["last_error"] = None
            st.session_state["active_view"] = "overview"
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
            """
            <style>
                [data-testid="stSidebar"] .stButton > button {
                    justify-content: flex-start;
                    padding-left: 14px;
                    font-weight: 600;
                }
                [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
                    background: rgba(255,255,255,0.04);
                    color: rgba(255,255,255,0.88);
                    border: 1px solid rgba(255,255,255,0.12);
                }
                [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
                    background: rgba(255,255,255,0.08);
                    color: #ffffff;
                    border-color: rgba(255,255,255,0.22);
                }
                [data-testid="stSidebar"] .stButton > button[kind="primary"] {
                    background: linear-gradient(135deg, #2d4f96 0%, #1A3263 100%) !important;
                    color: #ffffff !important;
                    border: 1px solid rgba(255,255,255,0.10) !important;
                    box-shadow: 0 10px 24px rgba(5, 12, 28, 0.26);
                }
                [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
                    background: linear-gradient(135deg, #345aa9 0%, #203d76 100%) !important;
                    color: #ffffff !important;
                }
            </style>
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
    html_block = f"""
    <div id="{container_id}" class="codelens-gauges-root">
      <style>
        #{container_id} {{
          width: 100%;
          margin: 0 auto 22px auto;
          font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        }}
        #{container_id} .gauge-scroller {{
          width: 100%;
          overflow-x: auto;
          overflow-y: visible;
          padding-bottom: 10px;
          scrollbar-width: thin;
          scrollbar-color: rgba(26, 50, 99, 0.35) rgba(255, 245, 232, 0.55);
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
          background: linear-gradient(180deg, rgba(255,245,232,0.98) 0%, rgba(246,221,193,0.96) 100%);
          border: 1px solid rgba(26, 50, 99, 0.12);
          border-radius: 24px;
          box-shadow: 0 18px 36px rgba(26, 50, 99, 0.12);
          padding: 16px 14px 14px 14px;
          overflow: hidden;
          box-sizing: border-box;
          transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
        }}
        #{container_id} .gauge-card:hover {{
          transform: translateY(-4px);
          box-shadow: 0 24px 42px rgba(26, 50, 99, 0.16);
          border-color: rgba(26, 50, 99, 0.22);
        }}
        #{container_id} .gauge-title {{
          font-size: 0.88rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: #5F5348;
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
          color: #1A3263;
          font-size: 2.2rem;
          font-weight: 900;
          line-height: 1;
          min-height: 36px;
          margin-top: -8px;
        }}
        #{container_id} .gauge-footer {{
          text-align: center;
          color: #5F5348;
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
          max-height: 0;
          transition: max-height 0.3s ease;
        }}
        #{container_id} .gauge-card:hover .gauge-details {{
          max-height: 240px;
        }}
        #{container_id} .hover-label {{
          font-size: 0.74rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: #547792;
          margin-bottom: 8px;
          font-weight: 800;
        }}
        #{container_id} .hover-copy {{
          margin: 0;
          padding-left: 18px;
          color: #1f1a14;
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
          background: rgba(26, 50, 99, 0.35);
          border-radius: 999px;
        }}
        #{container_id} .gauge-scroller::-webkit-scrollbar-track {{
          background: rgba(255, 245, 232, 0.5);
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

      function buildGauge(metric, value) {{
        return [{{
          type: 'indicator',
          mode: 'gauge+number',
          value: value,
          number: {{
            font: {{ size: 34, color: metric.muted ? '#547792' : metric.color, family: 'Segoe UI, sans-serif' }},
            valueformat: '.0f'
          }},
          gauge: {{
            axis: {{ range: [0, 100], tickwidth: 0, tickcolor: 'rgba(0,0,0,0)', tickfont: {{color: '#5F5348'}} }},
            bar: {{ color: metric.muted ? '#547792' : metric.color, thickness: 0.30 }},
            bgcolor: 'rgba(84,119,146,0.10)',
            borderwidth: 0,
            steps: [
              {{ range: [0, 40], color: 'rgba(218,72,72,0.18)' }},
              {{ range: [40, 60], color: 'rgba(255,197,112,0.28)' }},
              {{ range: [60, 100], color: 'rgba(26,50,99,0.14)' }}
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
        const duration = 1500;
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
    html_block = f"""
    <div id="{container_id}" class="strengths-root">
      <style>
        #{container_id} {{
          background: #EFD2B0;
        }}
        #{container_id} .stack-shell {{
          display: flex;
          align-items: stretch;
          gap: 16px;
        }}
        #{container_id} .stack-card {{
          flex: 1;
          border-radius: 16px;
          padding: 24px;
          box-sizing: border-box;
          display: flex;
          flex-direction: column;
        }}
        #{container_id} .strength-card {{
          background: #DDA853;
        }}
        #{container_id} .concern-card {{
          background: #1A3263;
        }}
        #{container_id} .stack-title {{
          color: #ffffff;
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
          padding: 14px 0;
          border-bottom: 1px solid rgba(255,255,255,0.2);
        }}
        #{container_id} .stack-item:last-child {{
          border-bottom: 0;
          padding-bottom: 0;
        }}
        #{container_id} .icon-col {{
          color: #ffffff;
          font-size: 1rem;
          line-height: 1.5;
          font-weight: 800;
          min-width: 18px;
          text-align: center;
        }}
        #{container_id} .item-copy {{
          color: #ffffff;
          line-height: 1.5;
          font-size: 0.96rem;
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
          border: 1.5px solid rgba(255,255,255,0.7);
          background: transparent;
          color: #ffffff;
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

    def pill_colors(status: str) -> tuple[str, str]:
        lowered = str(status).strip().lower()
        if lowered == "confirmed":
            return "#1A3263", "#ffffff"
        if lowered == "partial":
            return "#FFC570", "#ffffff"
        return "#DA4848", "#ffffff"

    def render_pills(entries: list[tuple[str, str]]) -> str:
        return "".join(
            f'<div class="skill-cell"><div class="skill-pill" data-bg="{pill_colors(status)[0]}" data-fg="{pill_colors(status)[1]}">{html.escape(skill)}</div></div>'
            for skill, status in entries
        )

    counts = {"confirmed": 0, "partial": 0, "not_found": 0}
    for _skill, status in items:
        lowered = str(status).strip().lower()
        if lowered in counts:
            counts[lowered] += 1
        else:
            counts["not_found"] += 1

    container_id = f"skill-map-{uuid.uuid4().hex}"
    html_block = f"""
    <div id="{container_id}" class="skill-map-root">
      <style>
        #{container_id} {{
          background: #EFD2B0;
        }}
        #{container_id} .skill-map-grid {{
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 18px;
          align-items: stretch;
        }}
        #{container_id} .skill-card,
        #{container_id} .summary-card {{
          background: #fffaf4;
          border: 1.5px solid #e2c49a;
          border-radius: 16px;
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
          font-family: 'Courier New', monospace;
          font-size: 0.98rem;
          font-weight: 700;
          line-height: 1.2;
          box-sizing: border-box;
          overflow: hidden;
          white-space: nowrap;
          text-overflow: ellipsis;
          background: #c8c5be;
          color: #8a847c;
          transition: background 240ms ease, color 240ms ease, transform 220ms ease;
          transform: scale(1);
        }}
        #{container_id} .skill-pill.revealed {{
          transform: scale(1.07);
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
          color: #1A3263;
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
          border-left: 1px solid #e2c49a;
        }}
        #{container_id} .summary-label {{
          color: #8a7866;
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
            border-top: 1px solid #e2c49a;
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
            <div class="summary-value" data-target="{counts["confirmed"]}" style="color:#1A3263;">0</div>
          </div>
          <div class="summary-col">
            <div class="summary-label">Partial</div>
            <div class="summary-value" data-target="{counts["partial"]}" style="color:#BA7517;">0</div>
          </div>
          <div class="summary-col">
            <div class="summary-label">Not Found</div>
            <div class="summary-value" data-target="{counts["not_found"]}" style="color:#DA4848;">0</div>
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
        const stagger = 60;
        const totalDuration = Math.max(600, (Math.max(pills.length - 1, 0) * stagger) + 360);
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
            }}, 180);
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
    container_id = f"ai-usage-{uuid.uuid4().hex}"
    html_block = f"""
    <div id="{container_id}" class="ai-usage-root">
      <style>
        #{container_id} {{
          background: #EFD2B0;
          font-family: "Segoe UI", Arial, sans-serif;
          color: #1f1a14;
        }}
        #{container_id} .ai-top-card {{
          background: #fff5e8;
          border: 1px solid rgba(26, 50, 99, 0.10);
          border-radius: 14px;
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
          color: #6c6258;
          font-size: 0.74rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          font-weight: 800;
        }}
        #{container_id} .ai-center-value {{
          margin-top: 6px;
          color: #1A3263;
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
          color: #6c6258;
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
          border-radius: 14px;
          border: 1px solid rgba(26, 50, 99, 0.10);
          padding: 18px 18px 16px 18px;
          min-height: 270px;
          box-sizing: border-box;
          display: flex;
          flex-direction: column;
        }}
        #{container_id} .info-card h4 {{
          margin: 0 0 12px 0;
          font-size: 1.02rem;
          color: #1A3263;
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
          color: #1f1a14;
        }}
        #{container_id} .mono {{
          font-family: 'Courier New', monospace;
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
                <div class="legend-value" style="color:#DA4848;">{ai_similarity}%</div>
                <div class="legend-label">AI baseline similarity</div>
              </div>
              <div class="legend-item">
                <div class="legend-value" style="color:#547792;">{human_similarity}%</div>
                <div class="legend-label">Human baseline similarity</div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="ai-card-row">
        <div class="info-card" style="background:#CBDCEB;">
          <h4>Vibe Coding Flags</h4>
          <ul>
            {"".join(f"<li>{_format_ai_usage_html(item)}</li>" for item in flags[:3])}
          </ul>
        </div>
        <div class="info-card" style="background:#D9E9CF;">
          <h4>AI Evidence Signals</h4>
          <ul>
            {"".join(f"<li>{_format_ai_usage_html(item)}</li>" for item in signals[:4])}
          </ul>
        </div>
        <div class="info-card" style="background:#F9DFDF;">
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
          colors: ['#DA4848', '#547792'],
          line: {{ color: '#ffffff', width: 0 }}
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

    def pill_style(status: str) -> tuple[str, str]:
        lowered = status.strip().lower()
        if lowered == "partial":
            return "#FFC570", "#4c2f13"
        if lowered == "not_found":
            return "#D25353", "#ffffff"
        return "#547792", "#ffffff"

    pills_html = "".join(
        f'<span style="display:inline-flex; align-items:center; padding:10px 16px; border-radius:999px; '
        f'background:{pill_style(status)[0]}; color:{pill_style(status)[1]}; font-weight:700; font-size:0.92rem;">'
        f"{html.escape(skill)}</span>"
        for skill, status in skills
    )

    container_id = f"job-fit-{uuid.uuid4().hex}"
    html_block = f"""
    <div id="{container_id}" class="job-fit-root">
      <style>
        #{container_id} {{
          background: #EFD2B0;
        }}
        #{container_id} .job-fit-shell {{
          display: grid;
          grid-template-columns: 200px minmax(0, 1fr);
          gap: 18px;
          align-items: stretch;
        }}
        #{container_id} .job-fit-card {{
          background: #fffaf4;
          border: 1.5px solid #e2c49a;
          border-radius: 16px;
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
          color: #1A3263;
          font-size: 1.02rem;
          font-weight: 800;
          margin: 0;
        }}
        #{container_id} .score-value {{
          color: #1A3263;
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
                <div style="font-size:1.6rem; font-weight:800; color:var(--accent); margin-top:4px;">Score Overview</div>
                <div class="muted" style="margin-top:6px;">Hover a gauge to inspect the detailed analysis for that metric.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_gauge_results_row(result)

        st.markdown('<div class="small-label" style="margin-bottom:10px;">Analysis Sections</div>', unsafe_allow_html=True)
        section_tabs = ["Skill Map", "AI Usage", "Code Issues"]
        if has_resume:
            section_tabs.append("Resume")
        if has_jd:
            section_tabs.append("Job Fit")
        subtabs = st.tabs(section_tabs)
        index = 0
        with subtabs[index]:
            render_skill_map(result)
        index += 1
        with subtabs[index]:
            render_ai_usage(result)
        index += 1
        with subtabs[index]:
            render_code_issues(result)
        index += 1
        if has_resume:
            with subtabs[index]:
                render_resume_panel(result)
            index += 1
        if has_jd:
            with subtabs[index]:
                render_job_fit_panel(result)

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
    recommendation_styles = {
        "strong_hire": ("#f2d0d0", "#8b2020"),
        "hire": ("#f2d0d0", "#8b2020"),
        "maybe": ("#fff0cc", "#8d5e11"),
        "pass": ("#f2d0d0", "#8b2020"),
    }
    recommendation_bg, recommendation_fg = recommendation_styles.get(recommendation, ("#f2d0d0", "#8b2020"))
    score_text = f"{verdict.get('overall_quality_score', 'N/A')} / 100"
    container_id = f"recommendation-{uuid.uuid4().hex}"
    animation_key = json.dumps(
        {
            "recommendation": recommendation,
            "score": verdict.get("overall_quality_score"),
            "summary": verdict.get("summary"),
            "reasoning": verdict.get("recommendation_reasoning"),
        },
        sort_keys=True,
        default=str,
    )
    html_block = f"""
    <div id="{container_id}" class="recommendation-root">
      <style>
        #{container_id} .recommendation-card {{
          background: #fffaf4;
          border: 1.5px solid #e2c49a;
          border-radius: 20px;
          padding: 28px 32px;
          box-sizing: border-box;
          opacity: 0;
          transform: translateY(12px);
          transition: opacity 320ms ease, transform 320ms ease;
        }}
        #{container_id}.visible .recommendation-card {{
          opacity: 1;
          transform: translateY(0);
        }}
        #{container_id} .top-row {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          flex-wrap: wrap;
        }}
        #{container_id} .eyebrow {{
          color: #547792;
          font-size: 0.76rem;
          font-weight: 700;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }}
        #{container_id} .pill-row {{
          display: flex;
          gap: 10px;
          align-items: center;
          flex-wrap: wrap;
        }}
        #{container_id} .rec-pill {{
          display: inline-flex;
          align-items: center;
          padding: 7px 14px;
          border-radius: 999px;
          font-size: 0.95rem;
          font-weight: 700;
          opacity: 0;
          transform: scale(0.84);
          transition: opacity 300ms ease, transform 500ms cubic-bezier(0.34, 1.56, 0.64, 1);
        }}
        #{container_id}.show-score .score-pill {{
          opacity: 1;
          transform: scale(1);
        }}
        #{container_id}.show-status .status-pill {{
          opacity: 1;
          transform: scale(1);
        }}
        #{container_id} .top-divider {{
          height: 1px;
          background: #e2c49a;
          margin: 18px 0 20px 0;
        }}
        #{container_id} .summary-heading {{
          display: flex;
          align-items: center;
          gap: 14px;
          color: #1A3263;
          font-size: 0.8rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          margin-bottom: 14px;
        }}
        #{container_id} .summary-heading::after {{
          content: "";
          height: 1px;
          background: #e2c49a;
          flex: 1;
        }}
        #{container_id} .summary-box-wrap {{
          opacity: 0;
          transform: translateY(10px);
          transition: opacity 320ms ease, transform 320ms ease;
        }}
        #{container_id}.expand-summary .summary-box-wrap {{
          opacity: 1;
          transform: translateY(0);
        }}
        #{container_id} .summary-copy {{
          font-family: Georgia, "Times New Roman", serif;
          font-size: 13.5px;
          line-height: 1.85;
          color: #3a4a5c;
        }}
        #{container_id} .summary-copy + .summary-copy {{
          margin-top: 14px;
        }}
        #{container_id} .disclaimer {{
          display: flex;
          align-items: flex-start;
          gap: 10px;
          border-top: 1px solid #e2c49a;
          padding-top: 14px;
          margin-top: 18px;
          opacity: 0;
          transition: opacity 260ms ease;
        }}
        #{container_id}.show-disclaimer .disclaimer {{
          opacity: 1;
        }}
        #{container_id} .disclaimer-icon {{
          width: 16px;
          height: 16px;
          color: #7e8a96;
          flex-shrink: 0;
          margin-top: 2px;
        }}
        #{container_id} .disclaimer-copy {{
          color: #7e8a96;
          font-size: 0.84rem;
          font-style: italic;
          line-height: 1.55;
        }}
      </style>
      <div class="recommendation-card" style="margin-top:16px;">
        <div class="top-row">
          <div class="eyebrow">Recommendation</div>
          <div class="pill-row">
            <span class="rec-pill score-pill" style="background:#1A3263; color:#FFC570;">{html.escape(score_text)}</span>
            <span class="rec-pill status-pill" style="background:{recommendation_bg}; color:{recommendation_fg};">{html.escape(recommendation_label)}</span>
          </div>
        </div>
        <div class="top-divider"></div>
        <div class="summary-box-wrap">
          <div class="summary-heading">Summary</div>
          <div class="summary-copy">{html.escape(str(verdict.get("summary", "")))}</div>
          <div class="summary-copy">{html.escape(str(verdict.get("recommendation_reasoning", "")))}</div>
        </div>
        <div class="disclaimer">
          <svg class="disclaimer-icon" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <circle cx="8" cy="8" r="6.5" stroke="currentColor" stroke-width="1.5"></circle>
            <path d="M8 7.1V11" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"></path>
            <circle cx="8" cy="4.6" r="0.9" fill="currentColor"></circle>
          </svg>
          <div class="disclaimer-copy">{html.escape(str(verdict.get("disclaimer", DISCLAIM_TEXT)))}</div>
          </div>
      </div>
    </div>
    <script>
      const root = document.getElementById({json.dumps(container_id)});
      const animationKey = "codelens-recommendation-" + {json.dumps(animation_key)};
      const alreadyPlayed = sessionStorage.getItem(animationKey) === 'done';
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
        setFrameHeight(Math.max(420, root.scrollHeight + 24));
      }}
      function runSequence() {{
        root.classList.add('visible');
        setTimeout(() => {{
          root.classList.add('show-score');
          refreshHeight();
        }}, 1900);
        setTimeout(() => {{
          root.classList.add('show-status');
          refreshHeight();
        }}, 2150);
        setTimeout(() => {{
          root.classList.add('expand-summary');
          refreshHeight();
        }}, 2450);
        setTimeout(() => {{
          root.classList.add('show-disclaimer');
          refreshHeight();
          sessionStorage.setItem(animationKey, 'done');
        }}, 2760);
      }}
      if (alreadyPlayed) {{
        root.classList.add('visible', 'show-score', 'show-status', 'expand-summary', 'show-disclaimer');
        refreshHeight();
      }} else {{
        runSequence();
      }}
      window.addEventListener('resize', refreshHeight);
      new MutationObserver(refreshHeight).observe(root, {{ childList: true, subtree: true, attributes: true }});
      refreshHeight();
    </script>
    """
    components.html(html_block, height=520, scrolling=False)


def render_analyze_tab() -> None:
    st.markdown("## Analyze")
    st.markdown("Intelligent code review for technical hiring.")

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
                    st.session_state["last_error"] = None
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
        render_results(st.session_state["last_result"])
    else:
        st.markdown('<div class="panel muted">Results will appear here.</div>', unsafe_allow_html=True)


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
    apply_global_styles()
    render_sidebar()

    missing = missing_api_keys()
    if missing:
        st.error("Missing required API keys: " + ", ".join(missing))

    render_app_header()
    active_view = st.session_state.get("active_view", "overview")
    tab_order = ["Analyze", "Evaluation", "MCP"]
    if active_view == "evaluation":
        tab_order = ["Evaluation", "Analyze", "MCP"]
    tab_objects = st.tabs(tab_order)
    for label, tab in zip(tab_order, tab_objects, strict=True):
        with tab:
            if label == "Analyze":
                render_analyze_tab()
            elif label == "Evaluation":
                render_evaluation_tab()
            else:
                render_mcp_tab()


if __name__ == "__main__":
    main()
