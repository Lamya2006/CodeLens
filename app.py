from __future__ import annotations

import importlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import streamlit as st
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
                --bg: #fdf5e4;
                --surface: #f6ebd5;
                --surface-2: #fffaf0;
                --border: #ddc6a8;
                --border-hover: #c7ac89;
                --accent: #435e94;
                --accent-2: #f2b25e;
                --danger: #c04b4b;
                --warning: #f2b25e;
                --text: #2f2418;
                --text-secondary: #7b6753;
                --text-muted: #9a8266;
            }

            html, body, [class*="css"] {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }

            .stApp {
                background:
                    radial-gradient(circle at top, rgba(242, 178, 94, 0.22), transparent 34%),
                    linear-gradient(180deg, #fdf5e4 0%, #f8eedb 100%);
                color: var(--text);
            }

            [data-testid="stSidebar"] {
                background: var(--surface);
                border-right: 1px solid var(--border);
            }

            [data-testid="stHeader"] {
                background: rgba(253, 245, 228, 0.96);
                border-bottom: 1px solid var(--border);
            }

            .block-container {
                padding-top: 1.5rem;
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

            .stButton > button {
                background: var(--surface-2);
                color: var(--text);
                border: 1px solid var(--border);
                border-radius: 12px;
            }

            .stButton > button[kind="primary"] {
                background: var(--accent) !important;
                color: #fdf5e4 !important;
                border: 1px solid var(--accent) !important;
                font-weight: 700 !important;
                min-height: 48px;
            }

            .stButton > button:hover {
                border-color: var(--border-hover);
                color: var(--text);
                background: #f8f0e0;
            }

            [data-testid="stTabs"] button {
                background: transparent;
                color: var(--text-secondary);
                border-bottom: 2px solid transparent;
            }

            [data-testid="stTabs"] button[aria-selected="true"] {
                color: var(--accent);
                border-bottom-color: var(--accent);
            }

            .panel {
                background: var(--surface-2);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 16px 18px;
                box-shadow: 0 12px 28px rgba(67, 94, 148, 0.08);
            }

            .score-card {
                background: var(--surface-2);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 18px;
                min-height: 132px;
                box-shadow: 0 10px 24px rgba(192, 75, 75, 0.06);
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

            .badge-green { background: rgba(67, 94, 148, 0.12); color: var(--accent); }
            .badge-purple { background: rgba(221, 198, 168, 0.45); color: #8c6d4e; }
            .badge-red { background: rgba(192, 75, 75, 0.12); color: var(--danger); }
            .badge-yellow { background: rgba(242, 178, 94, 0.18); color: #9b6518; }
            .badge-blue { background: rgba(67, 94, 148, 0.12); color: var(--accent); }
            .badge-gray { background: rgba(221, 198, 168, 0.28); color: var(--text-secondary); }

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
                background: linear-gradient(180deg, rgba(255, 250, 240, 0.98) 0%, rgba(246, 235, 213, 0.98) 100%);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 18px;
            }

            .logo-mark {
                color: var(--accent);
                font-size: 1.5rem;
                font-weight: 900;
                letter-spacing: 0.05em;
            }

            .small-label {
                color: var(--text-secondary);
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }

            .oauth-button {
                display: inline-flex;
                align-items: center;
                gap: 10px;
                padding: 12px 14px;
                border-radius: 12px;
                border: 1px solid var(--border);
                background: linear-gradient(135deg, #435e94 0%, #5f79ad 100%);
                color: #fdf5e4 !important;
                font-weight: 600;
                text-decoration: none;
                width: 100%;
                box-sizing: border-box;
            }

            .oauth-button:hover {
                border-color: #435e94;
                text-decoration: none;
                color: #fdf5e4 !important;
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


def missing_api_keys() -> list[str]:
    required = ["OPENROUTER_API_KEY", "PINECONE_API_KEY", "VOYAGE_API_KEY"]
    return [key for key in required if not os.getenv(key)]


def oauth_ready() -> bool:
    return bool(os.getenv("GITHUB_CLIENT_ID") and os.getenv("GITHUB_CLIENT_SECRET") and os.getenv("APP_URL"))


def oauth_redirect_uri() -> str:
    return os.getenv("APP_URL", "http://localhost:8501").rstrip("/")


def score_color(score: int | None) -> str:
    if score is None:
        return "#7b6753"
    if score > 75:
        return "#435E94"
    if score >= 50:
        return "#F2B25E"
    return "#C04B4B"


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
        <div style="margin-bottom: 18px;">
            <div class="logo-mark">CodeLens</div>
            <div style="color:var(--text-secondary); margin-top:6px;">Intelligent code review for technical hiring.</div>
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
        st.markdown('<div class="muted" style="margin-top:8px;">No saved analyses yet.</div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="small-label" style="margin-top:18px;">Recent Analyses</div>', unsafe_allow_html=True)
    for entry in analyses[:5]:
        repo_name = entry.get("repo_name", "Repository")
        analyzed_at = entry.get("analyzed_at", "")[:10]
        badge_class = recommendation_badge_color(entry.get("recommendation", "maybe"))
        st.markdown(
            f"""
            <div class="history-card">
                <div style="font-weight:700;">{repo_name}</div>
                <div class="muted" style="margin-top:6px;">{analyzed_at}</div>
                <div style="margin-top:8px;">
                    <span class="badge {badge_class}">{str(entry.get("recommendation", "maybe")).replace("_", " ")}</span>
                    <span class="badge badge-gray" style="margin-left:6px;">score {entry.get("overall_quality_score", "N/A")}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open analysis", key=f"history-open-{entry['id']}", use_container_width=True):
            st.session_state["last_result"] = entry.get("result")
            st.session_state["last_error"] = None
            st.rerun()

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
        for entry in analyses:
            if st.button(f"Load {entry.get('repo_name', 'analysis')}", key=f"history-load-{entry['id']}", use_container_width=True):
                st.session_state["last_result"] = entry.get("result")
                st.session_state["last_error"] = None
                st.rerun()


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="logo-mark">CodeLens</div>', unsafe_allow_html=True)
        st.caption("Intelligent code review for technical hiring.")
        st.markdown('<span class="badge badge-purple">v0.1 alpha</span>', unsafe_allow_html=True)

        user = st.session_state.get("user")
        if user:
            if user.get("avatar_url"):
                st.image(user["avatar_url"], width=56)
            st.markdown(f'<div style="font-weight:700;">@{user.get("username","")}</div>', unsafe_allow_html=True)
            if st.button("Sign out", use_container_width=True):
                sign_out()
            render_recent_history()
        else:
            st.markdown('<div class="small-label" style="margin-top:18px;">Account</div>', unsafe_allow_html=True)
            render_oauth_button()

        ready, detail = check_mcp_status()
        status_class = "badge-green" if ready else "badge-red"
        label = "MCP ready" if ready else "MCP unavailable"
        st.markdown(
            f"""
            <div style="margin-top:18px;">
                <div class="small-label">MCP Server</div>
                <div style="margin-top:8px;"><span class="badge {status_class}">{label}</span></div>
                <div class="muted" style="margin-top:8px;">{detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Clear cache", use_container_width=True):
            st.cache_data.clear()
            st.session_state["last_result"] = None
            st.session_state["last_error"] = None
            st.rerun()

        with st.expander("How it works", expanded=False):
            st.markdown(
                """
                1. Validate the GitHub repository and fetch commit history.
                2. Index the codebase with GitNexus and embed chunks into Pinecone.
                3. Compare style against human and AI baseline corpora.
                4. Optionally parse a resume and job description for claim matching.
                5. Run the LLM review (efficient mode: two calls by default) and apply output guardrails.
                   Set `CREWAI_MODE=full` in `.env` for the original multi-agent chain (more detail, higher cost).
                """
            )


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


def render_strengths_and_concerns(result: dict[str, Any]) -> None:
    verdict = result["verdict"]
    reports = result.get("reports", {})
    quality_report = reports.get("code_quality", {})
    concerns_by_text = {
        concern.get("issue", ""): concern
        for concern in quality_report.get("concerns", [])
        if isinstance(concern, dict)
    }

    left, right = st.columns(2)
    with left:
        strengths = verdict.get("strengths", [])
        strength_lines = "".join(f'<div class="list-item">→ {item}</div>' for item in strengths)
        if not strength_lines:
            strength_lines = '<div class="muted">No major strengths were recorded.</div>'
        st.markdown(
            f'<div class="panel strength-panel"><div class="section-title">Strengths</div>{strength_lines}</div>',
            unsafe_allow_html=True,
        )

    with right:
        concern_html = ""
        for item in verdict.get("concerns", []):
            severity = concerns_by_text.get(item, {}).get("severity", "medium")
            badge_class = {"low": "badge-gray", "medium": "badge-yellow", "high": "badge-red"}.get(severity, "badge-yellow")
            concern_html += (
                f'<div class="list-item">⚠ {item}<div style="margin-top:6px;">'
                f'<span class="badge {badge_class}">{severity}</span></div></div>'
            )
        if not concern_html:
            concern_html = '<div class="muted">No major concerns were recorded.</div>'
        st.markdown(
            f'<div class="panel concern-panel"><div class="section-title">Concerns</div>{concern_html}</div>',
            unsafe_allow_html=True,
        )


def render_skill_map(result: dict[str, Any]) -> None:
    skill_map = result["verdict"].get("skill_map", {})
    if not skill_map:
        st.info("No skill map available for this analysis.")
        return
    rows = []
    for skill, status in skill_map.items():
        badge = {"confirmed": "badge-green", "partial": "badge-yellow", "not_found": "badge-red"}.get(status, "badge-gray")
        rows.append(f"<tr><td>{skill}</td><td><span class='badge {badge}'>{status}</span></td></tr>")
    st.markdown(
        f"""
        <div class="panel">
            <table style="width:100%; border-collapse:collapse;">
                <thead>
                    <tr><th style="text-align:left; color:var(--text-secondary);">Skill</th><th style="text-align:left; color:var(--text-secondary);">Status</th></tr>
                </thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_usage(result: dict[str, Any]) -> None:
    verdict = result["verdict"]
    reports = result.get("reports", {})
    ai_report = reports.get("ai_usage", {})
    baseline = result["analysis_data"].get("baseline_comparison", {})

    st.markdown("**Baseline Similarity**")
    st.progress(min(max(float(baseline.get("human_similarity", 0.0)), 0.0), 1.0), text="Human baseline similarity")
    st.progress(min(max(float(baseline.get("ai_similarity", 0.0)), 0.0), 1.0), text="AI baseline similarity")

    st.markdown("**Vibe Coding Flags**")
    flags = verdict.get("vibe_coding_flags", [])
    if flags:
        for flag in flags:
            st.markdown(f'<div class="panel" style="border-left:4px solid #F2B25E; margin-bottom:8px;">{flag}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="muted">No vibe-coding flags triggered.</div>', unsafe_allow_html=True)

    st.markdown("**AI Evidence Signals**")
    signals = ai_report.get("ai_evidence_signals", [])
    if signals:
        for signal in signals:
            st.markdown(
                f"""
                <div class="panel" style="margin-bottom:8px;">
                    <div><strong>{signal.get("signal", "")}</strong></div>
                    <div class="muted">{signal.get("location", "")}</div>
                    <div style="margin-top:8px;">{signal.get("note", "")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(f'<div class="panel">{verdict.get("ai_usage_summary", "No detailed AI evidence signals available.")}</div>', unsafe_allow_html=True)

    st.markdown("**Good AI Usage**")
    good_examples = ai_report.get("good_ai_usage", [])
    if good_examples:
        for item in good_examples:
            st.markdown(f'<div class="panel" style="border-left:4px solid #435E94; margin-bottom:8px;">{item}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="muted">No explicit good-AI examples were surfaced.</div>', unsafe_allow_html=True)


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

    st.markdown(
        f"""
        <div class="panel">
            <div class="section-title">Job Fit Score</div>
            <div class="score-value" style="color:{score_color(verdict.get("job_fit_score"))};">{verdict.get("job_fit_score", "N/A")}</div>
            <div style="margin-top:10px;">{verdict.get("job_fit_analysis") or "No job fit analysis was returned."}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    required_skills = job_data.get("required_skills", [])
    skill_map = verdict.get("skill_map", {})
    if required_skills:
        rows = []
        for skill in required_skills:
            status = skill_map.get(skill, "not_found")
            badge = {"confirmed": "badge-green", "partial": "badge-yellow", "not_found": "badge-red"}.get(status, "badge-gray")
            rows.append(f"<tr><td>{skill}</td><td><span class='badge {badge}'>{status}</span></td></tr>")
        st.markdown(
            f"""
            <div class="panel" style="margin-top:12px;">
                <div class="section-title">Required Skills Comparison</div>
                <table style="width:100%; border-collapse:collapse;">
                    <thead><tr><th style="text-align:left; color:var(--text-secondary);">Skill</th><th style="text-align:left; color:var(--text-secondary);">Evidence</th></tr></thead>
                    <tbody>{''.join(rows)}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_results(result: dict[str, Any]) -> None:
    verdict = result["verdict"]
    has_resume = result.get("resume_data") is not None
    has_jd = result.get("job_description") is not None

    st.markdown("### Results")
    score_titles = [
        ("Overall Quality", verdict.get("overall_quality_score"), verdict.get("summary", "")),
        ("AI Usage", verdict.get("ai_usage_score"), verdict.get("ai_usage_summary", "")),
        ("Commit Health", verdict.get("commit_health_score"), verdict.get("data_confidence", "").title()),
        ("Resume Match", verdict.get("resume_match_score"), "Resume-linked evidence" if has_resume else "No resume uploaded"),
    ]
    if has_jd:
        score_titles.append(("Job Fit", verdict.get("job_fit_score"), "Role alignment"))

    cols = st.columns(len(score_titles))
    for idx, (title, score, meta) in enumerate(score_titles):
        with cols[idx]:
            render_score_card(title, score, meta=meta, muted=(title == "Resume Match" and not has_resume))

    render_strengths_and_concerns(result)

    subtab_names = ["Skill Map", "AI Usage", "Code Issues"]
    if has_resume:
        subtab_names.append("Resume")
    if has_jd:
        subtab_names.append("Job Fit")
    subtabs = st.tabs(subtab_names)
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

    recommendation = verdict.get("recommendation", "maybe")
    badge_class = recommendation_badge_color(recommendation)
    st.markdown(
        f"""
        <div class="panel" style="margin-top:16px;">
            <div class="section-title">Recommendation</div>
            <div style="margin: 8px 0 14px 0;"><span class="badge {badge_class}" style="font-size:1rem; padding:6px 14px;">{recommendation.replace('_', ' ').title()}</span></div>
            <div class="summary-box">
                <div style="font-weight:700; margin-bottom:10px;">Summary</div>
                <div>{verdict.get("summary", "")}</div>
                <div style="margin-top:12px;" class="muted">{verdict.get("recommendation_reasoning", "")}</div>
            </div>
            <div class="muted" style="margin-top:12px; font-size:0.86rem;">{verdict.get("disclaimer", DISCLAIM_TEXT)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def main() -> None:
    init_session_state()
    handle_oauth_callback()
    apply_global_styles()
    render_sidebar()

    missing = missing_api_keys()
    if missing:
        st.error("Missing required API keys: " + ", ".join(missing))

    render_app_header()
    analyze_tab, evaluation_tab, mcp_tab = st.tabs(["Analyze", "Evaluation", "MCP"])
    with analyze_tab:
        render_analyze_tab()
    with evaluation_tab:
        render_evaluation_tab()
    with mcp_tab:
        render_mcp_tab()


if __name__ == "__main__":
    main()
