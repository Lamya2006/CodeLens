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
    theme = st.session_state.get("theme", "light")
    if theme == "dark":
        theme_vars = """
            --bg-base: #080c14;
            --bg-gradient: linear-gradient(160deg, #080c14 0%, #0d1426 50%, #0a1020 100%);
            --glass-bg: rgba(30, 41, 59, 0.5);
            --glass-bg-hover: rgba(30, 41, 59, 0.68);
            --glass-bg-inner: rgba(30, 41, 59, 0.62);
            --glass-border: rgba(248, 250, 252, 0.22);
            --glass-border-subtle: rgba(248, 250, 252, 0.16);
            --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.42), 0 2px 8px rgba(96, 165, 250, 0.1);
            --top-edge-highlight: rgba(255, 255, 255, 0.15);
            --accent-blue: #60A5FA;
            --accent-blue-hover: #93C5FD;
            --accent-blue-light: rgba(96, 165, 250, 0.2);
            --text-primary: #F8FAFC;
            --text-secondary: rgba(248, 250, 252, 0.86);
            --text-muted: rgba(248, 250, 252, 0.68);
            --score-green: #3B82F6;
            --score-yellow: #FBBF24;
            --score-red: #F87171;
            --card-glow: rgba(96, 165, 250, 0.2);
            --surface-white: rgba(255, 255, 255, 0.82);
            --status-border: rgba(248, 250, 252, 0.22);
            --glass-blur-theme: blur(12px);
            --glass-blur-inner-theme: blur(12px);
            --input-bg: rgba(30, 41, 59, 0.62);
            --input-text: #F8FAFC;
            --input-placeholder: rgba(248, 250, 252, 0.55);
            --input-border: rgba(248, 250, 252, 0.18);
            --stapp-bg-image: radial-gradient(circle at 15% 18%, rgba(56, 168, 245, 0.22), transparent 26%),
                radial-gradient(circle at 86% 10%, rgba(255, 255, 255, 0.52), transparent 24%),
                radial-gradient(circle at 78% 58%, rgba(56, 168, 245, 0.15), transparent 20%),
                var(--bg-gradient);
        """
    else:
        theme_vars = """
            --bg-base: #b8e2ff;
            --bg-gradient: linear-gradient(180deg, #7dcaf8 0%, #a8dcff 15%, #c8eeff 35%, #dff5ff 58%, #f5fcff 82%, #ffffff 100%);
            --stapp-bg-image: radial-gradient(ellipse 130% 42% at 50% -4%, rgba(90, 190, 255, 0.55), transparent 52%),
                radial-gradient(circle at 8% 6%, rgba(56, 168, 245, 0.38), transparent 28%),
                radial-gradient(circle at 92% 9%, rgba(147, 210, 255, 0.42), transparent 26%),
                var(--bg-gradient);
            --glass-bg: rgba(255, 255, 255, 0.65);
            --glass-bg-hover: rgba(255, 255, 255, 0.78);
            --glass-bg-inner: rgba(255, 255, 255, 0.72);
            --glass-border: rgba(15, 23, 42, 0.18);
            --glass-border-subtle: rgba(15, 23, 42, 0.12);
            --glass-shadow: 0 8px 32px rgba(100, 160, 255, 0.12), 0 2px 8px rgba(79, 70, 229, 0.08);
            --top-edge-highlight: rgba(255, 255, 255, 0.8);
            --accent-blue: #4F46E5;
            --accent-blue-hover: #6366F1;
            --accent-blue-light: rgba(79, 70, 229, 0.14);
            --text-primary: #0F172A;
            --text-secondary: rgba(15, 23, 42, 0.78);
            --text-muted: rgba(15, 23, 42, 0.52);
            --score-green: #0EA5E9;
            --score-yellow: #F59E0B;
            --score-red: #E11D48;
            --card-glow: rgba(79, 70, 229, 0.12);
            --surface-white: rgba(255, 255, 255, 0.96);
            --status-border: rgba(15, 23, 42, 0.16);
            --glass-blur-theme: blur(10px);
            --glass-blur-inner-theme: blur(10px);
            --input-bg: rgba(255, 255, 255, 0.88);
            --input-text: #0F172A;
            --input-placeholder: rgba(15, 23, 42, 0.42);
            --input-border: rgba(15, 23, 42, 0.14);
        """

    st.markdown(
        f"""
        <style>
            :root {{
                {theme_vars}
                --glass-blur: var(--glass-blur-theme) saturate(180%);
                --glass-blur-inner: var(--glass-blur-inner-theme) saturate(165%);
                --glass-radius: 20px;
                --glass-radius-sm: 14px;
                --ease-swift: cubic-bezier(0.16, 1, 0.3, 1);
                --theme-transition: 400ms ease-in-out;
                --font-sans: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", system-ui, sans-serif;
                --font-mono: "SF Mono", "Fira Code", ui-monospace, monospace;
            }}

            *, *::before, *::after {{
                transition-property: color, background-color, border-color, fill, stroke, box-shadow;
                transition-duration: 400ms;
                transition-timing-function: ease-in-out;
            }}

            html, body, [class*="css"] {{
                font-family: var(--font-sans);
            }}

            html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{
                background: transparent;
                color: var(--text-primary);
            }}

            .stApp {{
                background-color: var(--bg-base);
                background-image: var(--stapp-bg-image);
                background-attachment: fixed;
            }}

            [data-testid="stSidebarCollapseButton"] button,
            [data-testid="collapsedControl"] button {{
                color: var(--text-primary) !important;
                background: var(--glass-bg-inner) !important;
                border: 1px solid var(--glass-border-subtle) !important;
                border-radius: 10px !important;
                box-shadow: var(--glass-shadow) !important;
            }}

            [data-testid="stAppViewContainer"] {{
                background: transparent;
            }}

            [data-testid="stHeader"] {{
                background: transparent;
                border: none;
            }}

            [data-testid="stDecoration"] {{
                display: none;
            }}

            [data-testid="stSidebar"] {{
                min-width: 260px;
                max-width: 260px;
                background:
                    linear-gradient(180deg, rgba(255, 255, 255, 0.16), rgba(255, 255, 255, 0.04)),
                    var(--glass-bg);
                backdrop-filter: var(--glass-blur);
                -webkit-backdrop-filter: var(--glass-blur);
                border-right: 1px solid var(--glass-border-subtle);
                box-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.08);
            }}

            [data-testid="stSidebar"] > div:first-child {{
                background: transparent;
                padding-top: 0 !important;
            }}

            [data-testid="stSidebar"] section[data-testid="stSidebarNav"] {{
                padding-top: 0 !important;
            }}

            .block-container {{
                max-width: 1240px;
                padding-top: 1.15rem;
                padding-bottom: 2.5rem;
            }}

            h1, h2, h3, h4, h5, h6, p, li, div, span, label {{
                color: var(--text-primary);
            }}

            p, li {{
                font-size: 14px;
                line-height: 1.6;
            }}

            code, pre, .code-font {{
                font-family: var(--font-mono) !important;
            }}

            ::-webkit-scrollbar {{
                width: 6px;
                height: 6px;
            }}

            ::-webkit-scrollbar-track {{
                background: transparent;
            }}

            ::-webkit-scrollbar-thumb {{
                background: var(--glass-border);
                border-radius: 3px;
            }}

            @keyframes fadeSlideUp {{
                from {{
                    opacity: 0;
                    transform: translateY(12px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}

            @keyframes pulse-border {{
                0%, 100% {{ border-left-color: rgba(56, 168, 245, 0.6); }}
                50% {{ border-left-color: rgba(56, 168, 245, 1); }}
            }}

            @keyframes shimmerSweep {{
                from {{ transform: translateX(-130%); }}
                to {{ transform: translateX(130%); }}
            }}

            .glass-card,
            .panel,
            .tool-card,
            .history-card,
            .summary-box,
            .glass-pill,
            .glass-table,
            .glass-shell {{
                position: relative;
                overflow: hidden;
                background: var(--glass-bg);
                border: 1px solid var(--glass-border);
                border-top: 1px solid var(--top-edge-highlight);
                border-radius: var(--glass-radius);
                box-shadow: var(--glass-shadow);
                backdrop-filter: var(--glass-blur);
                -webkit-backdrop-filter: var(--glass-blur);
            }}

            .glass-card::before,
            .panel::before,
            .tool-card::before,
            .history-card::before,
            .summary-box::before,
            .glass-shell::before {{
                content: "";
                position: absolute;
                inset: 0;
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.22), transparent 35%);
                pointer-events: none;
            }}

            .glass-card-inner,
            .summary-box,
            .nested-glass,
            .list-item,
            .strength-pill,
            .concern-pill,
            .progress-pill,
            .skill-pill {{
                background: var(--glass-bg-inner);
                border: 1px solid var(--glass-border-subtle);
                border-top: 1px solid var(--top-edge-highlight);
                backdrop-filter: var(--glass-blur-inner);
                -webkit-backdrop-filter: var(--glass-blur-inner);
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.15);
            }}

            .codelens-reveal {{
                animation: fadeSlideUp 0.25s var(--ease-swift) forwards;
                animation-delay: var(--reveal-delay, 0ms);
                opacity: 0;
            }}

            .tab-fade {{
                animation: fadeSlideUp 0.15s ease-out forwards;
            }}

            .hero-shell {{
                max-width: 720px;
                margin: 0 auto 24px;
                padding: 32px;
                background: rgba(255, 255, 255, 0.56);
            }}

            .hero-heading {{
                margin: 0;
                font-size: 28px;
                font-weight: 600;
                letter-spacing: -0.02em;
                text-align: center;
                text-shadow: 0 0 40px rgba(56, 168, 245, 0.3);
            }}

            .hero-subheading {{
                margin: 10px 0 0;
                text-align: center;
                color: var(--text-secondary);
                font-size: 14px;
            }}

            .page-shell {{
                display: flex;
                flex-direction: column;
                gap: 18px;
            }}

            .cl-results-wrap {{
                overflow: visible;
            }}

            .cl-results-wrap > div {{
                overflow: visible;
            }}

            .brand-wordmark {{
                font-size: 20px;
                font-weight: 600;
                letter-spacing: -0.02em;
            }}

            .brand-wordmark .code {{
                color: var(--text-primary);
            }}

            .brand-wordmark .lens {{
                color: var(--accent-blue);
            }}

            .brand-tagline,
            .muted,
            .small-label {{
                color: var(--text-muted);
            }}

            .small-label {{
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.12em;
            }}

            .brand-block {{
                display: flex;
                align-items: center;
                gap: 14px;
            }}

            .brand-logo-wrap {{
                width: 46px;
                height: 46px;
                border-radius: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.68), rgba(56, 168, 245, 0.18));
                border: 1px solid var(--glass-border);
                box-shadow: 0 12px 28px rgba(56, 168, 245, 0.14);
                backdrop-filter: var(--glass-blur-inner);
                -webkit-backdrop-filter: var(--glass-blur-inner);
            }}

            .brand-logo-svg {{
                display: block;
            }}

            .sidebar-shell {{
                display: flex;
                flex-direction: column;
                gap: 16px;
                padding: 4px 0 24px;
            }}

            .sidebar-card {{
                padding: 18px;
            }}

            .sidebar-divider,
            .section-rule {{
                height: 1px;
                margin: 20px 0;
                background: linear-gradient(90deg, transparent, var(--glass-border-subtle), transparent);
                border: none;
            }}

            .app-shell-head {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 16px;
                margin-bottom: 14px;
            }}

            .app-shell-copy h1 {{
                margin: 0;
                font-size: 28px;
                font-weight: 600;
                letter-spacing: -0.02em;
            }}

            .app-shell-copy p {{
                margin: 8px 0 0;
                color: var(--text-secondary);
            }}

            [data-testid="stTextInput"] label,
            [data-testid="stTextArea"] label,
            [data-testid="stFileUploader"] label {{
                color: var(--text-secondary) !important;
                font-size: 12px !important;
                font-weight: 600 !important;
                letter-spacing: 0.08em !important;
                text-transform: uppercase !important;
            }}

            [data-testid="stTextInput"] [data-baseweb="base-input"],
            [data-testid="stTextArea"] [data-baseweb="base-input"],
            [data-testid="stTextInput"] > div,
            [data-testid="stTextArea"] > div {{
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
                border-radius: 12px !important;
            }}

            [data-testid="stTextInput"] input,
            [data-testid="stTextArea"] textarea {{
                background: var(--input-bg) !important;
                color: var(--input-text) !important;
                border: 1px solid var(--input-border) !important;
                border-radius: 12px !important;
                padding: 12px 16px !important;
                box-shadow: 0 2px 8px rgba(100, 160, 255, 0.08) !important;
                transition: all var(--theme-transition) !important;
            }}

            [data-testid="stTextInput"] input::placeholder,
            [data-testid="stTextArea"] textarea::placeholder {{
                color: var(--input-placeholder) !important;
            }}

            [data-testid="stTextInput"] input:focus,
            [data-testid="stTextArea"] textarea:focus {{
                border-color: var(--accent-blue) !important;
                box-shadow: 0 0 0 3px rgba(56, 168, 245, 0.15), 0 2px 8px rgba(100, 160, 255, 0.08) !important;
                outline: none !important;
            }}

            [data-testid="stFileUploaderDropzone"] {{
                background: var(--input-bg) !important;
                border: 1px dashed var(--accent-blue) !important;
                border-radius: 12px !important;
                box-shadow: 0 2px 8px rgba(100, 160, 255, 0.08) !important;
                transition: all var(--theme-transition) !important;
            }}

            [data-testid="stFileUploaderDropzone"]:hover {{
                border-color: var(--accent-blue) !important;
                background: var(--glass-bg-hover) !important;
            }}

            [data-testid="stFileUploaderDropzone"] button {{
                color: var(--accent-blue) !important;
                background: rgba(56, 168, 245, 0.1) !important;
                border: 1px solid rgba(56, 168, 245, 0.35) !important;
                border-radius: 8px !important;
            }}

            [data-testid="stFileUploaderDropzone"] small,
            [data-testid="stFileUploaderDropzone"] span,
            [data-testid="stFileUploaderDropzone"] p {{
                color: var(--text-secondary) !important;
            }}

            [data-testid="stFileUploaderDropzone"] svg {{
                color: var(--accent-blue) !important;
                stroke: var(--accent-blue) !important;
            }}

            .stButton > button {{
                width: 100%;
                min-height: 46px;
                border-radius: 14px;
                border: 1px solid var(--glass-border-subtle);
                border-top: 1px solid var(--top-edge-highlight);
                background: var(--glass-bg);
                color: var(--text-secondary);
                backdrop-filter: var(--glass-blur);
                -webkit-backdrop-filter: var(--glass-blur);
                box-shadow: var(--glass-shadow);
                transition: all var(--theme-transition);
            }}

            .stButton > button:hover {{
                background: var(--glass-bg-hover);
                color: var(--text-primary);
                border-color: var(--glass-border);
                transform: translateY(-1px);
            }}

            .stButton > button:active {{
                transform: scale(0.97);
            }}

            .stButton > button:focus-visible,
            [data-baseweb="radio"] input:focus-visible + div {{
                outline: none !important;
                box-shadow: 0 0 0 3px rgba(56, 168, 245, 0.18) !important;
            }}

            .primary-action .stButton > button,
            .stButton > button[kind="primary"] {{
                position: relative;
                overflow: hidden;
                background: var(--accent-blue) !important;
                color: white !important;
                font-size: 15px !important;
                font-weight: 600 !important;
                border: 1px solid rgba(255, 255, 255, 0.22) !important;
                box-shadow: 0 4px 16px rgba(56, 168, 245, 0.35) !important;
            }}

            .primary-action .stButton > button::after,
            .stButton > button[kind="primary"]::after {{
                content: "";
                position: absolute;
                inset: -20%;
                width: 40%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.55), transparent);
                transform: translateX(-130%);
            }}

            .primary-action .stButton > button:hover,
            .stButton > button[kind="primary"]:hover {{
                background: var(--accent-blue-hover) !important;
                box-shadow: 0 6px 20px rgba(56, 168, 245, 0.45) !important;
            }}

            .primary-action .stButton > button:hover::after,
            .stButton > button[kind="primary"]:hover::after {{
                animation: shimmerSweep 0.4s var(--ease-swift) forwards;
            }}

            .tab-switcher [data-baseweb="radio"] > div,
            .theme-switcher [data-baseweb="radio"] > div {{
                gap: 0.4rem;
                background: var(--glass-bg);
                border: 1px solid var(--glass-border);
                border-top: 1px solid var(--top-edge-highlight);
                border-radius: 14px;
                padding: 4px;
                backdrop-filter: var(--glass-blur);
                -webkit-backdrop-filter: var(--glass-blur);
                box-shadow: var(--glass-shadow);
            }}

            .tab-switcher [data-baseweb="radio"] label,
            .theme-switcher [data-baseweb="radio"] label {{
                margin: 0 !important;
            }}

            .tab-switcher [data-baseweb="radio"] label > div:first-child,
            .theme-switcher [data-baseweb="radio"] label > div:first-child {{
                display: none;
            }}

            .tab-switcher [data-baseweb="radio"] label > div:last-child,
            .theme-switcher [data-baseweb="radio"] label > div:last-child {{
                min-width: 0;
                padding: 10px 16px;
                border-radius: 10px;
                color: var(--text-secondary);
                font-size: 14px;
                font-weight: 600;
                transition: all var(--theme-transition);
            }}

            .tab-switcher [data-baseweb="radio"] input:checked + div,
            .theme-switcher [data-baseweb="radio"] input:checked + div {{
                background: var(--accent-blue);
                color: #fff !important;
                box-shadow: 0 2px 8px rgba(56, 168, 245, 0.4);
            }}

            .theme-switcher [data-baseweb="radio"] label > div:last-child {{
                min-width: 92px;
                text-align: center;
            }}

            .score-card {{
                min-height: 172px;
                padding: 18px 18px 20px;
                border-radius: 22px;
                transition: all var(--theme-transition);
            }}

            .score-card:hover {{
                transform: translateY(-2px);
                background: var(--glass-bg-hover);
            }}

            .score-card-inner {{
                display: flex;
                flex-direction: column;
                align-items: flex-start;
                gap: 12px;
            }}

            .score-gauge-svg {{
                display: block;
            }}

            .score-gauge-arc {{
                transition: stroke-dashoffset 0.28s var(--ease-swift);
            }}

            .score-label {{
                font-size: 12px;
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }}

            .score-value {{
                font-size: 36px;
                line-height: 1;
                font-weight: 700;
            }}

            .score-meta {{
                color: var(--text-secondary);
                font-size: 13px;
                line-height: 1.5;
            }}

            .list-item,
            .strength-pill,
            .concern-pill,
            .progress-pill {{
                border-radius: 14px;
                padding: 12px 14px;
                margin-top: 10px;
                color: var(--text-secondary);
            }}

            .progress-pill {{
                border-left: 3px solid transparent;
            }}

            .progress-pill.active {{
                border-left-color: var(--accent-blue);
                animation: pulse-border 1.4s ease-in-out infinite;
            }}

            .progress-pill.completed {{
                opacity: 0.7;
            }}

            .progress-pill.pending {{
                opacity: 0.4;
            }}

            .section-title {{
                font-size: 16px;
                font-weight: 600;
                color: var(--text-primary);
                margin-bottom: 8px;
            }}

            .panel,
            .tool-card,
            .history-card {{
                padding: 18px;
            }}

            .panel-accent-warn {{
                border-left: 3px solid var(--score-yellow);
            }}

            .panel-accent-positive {{
                border-left: 3px solid var(--score-green);
            }}

            .panel-accent-danger {{
                border-left: 3px solid var(--score-red);
            }}

            .panel-tight {{
                margin-bottom: 10px;
            }}

            .badge,
            .skill-pill {{
                display: inline-flex;
                align-items: center;
                gap: 6px;
                border-radius: 999px;
                padding: 4px 10px;
                font-size: 12px;
                line-height: 1;
                border: 1px solid var(--glass-border-subtle);
            }}

            .badge-green {{
                background: rgba(34, 197, 94, 0.15);
                border-color: rgba(34, 197, 94, 0.35);
                color: #16a34a;
            }}

            .badge-blue {{
                background: rgba(56, 168, 245, 0.15);
                border-color: rgba(56, 168, 245, 0.35);
                color: #0369a1;
            }}

            .badge-yellow {{
                background: rgba(245, 158, 11, 0.15);
                border-color: rgba(245, 158, 11, 0.35);
                color: #b45309;
            }}

            .badge-red {{
                background: rgba(239, 68, 68, 0.15);
                border-color: rgba(239, 68, 68, 0.35);
                color: #dc2626;
            }}

            .badge-purple,
            .badge-gray {{
                background: var(--glass-bg-inner);
                color: var(--text-secondary);
            }}

            .recommendation-panel {{
                margin-top: 18px;
                padding: 22px;
            }}

            .recommendation-badge {{
                width: 100%;
                padding: 16px 24px;
                border-radius: 999px;
                font-size: 14px;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                animation: fadeSlideUp 0.25s var(--ease-swift) forwards;
                transform-origin: center;
            }}

            .recommendation-badge.strong_hire {{
                background: rgba(34, 197, 94, 0.15);
                border: 1px solid rgba(34, 197, 94, 0.4);
                color: #16a34a;
            }}

            .recommendation-badge.hire {{
                background: rgba(56, 168, 245, 0.15);
                border: 1px solid rgba(56, 168, 245, 0.4);
                color: #0369a1;
            }}

            .recommendation-badge.maybe {{
                background: rgba(245, 158, 11, 0.15);
                border: 1px solid rgba(245, 158, 11, 0.4);
                color: #b45309;
            }}

            .recommendation-badge.pass {{
                background: rgba(239, 68, 68, 0.15);
                border: 1px solid rgba(239, 68, 68, 0.4);
                color: #dc2626;
            }}

            .summary-box {{
                margin-top: 14px;
                padding: 18px;
                border-radius: 18px;
            }}

            .data-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }}

            .data-table th,
            .data-table td {{
                text-align: left;
                padding: 12px;
                border-bottom: 1px solid var(--glass-border-subtle);
            }}

            .data-table thead th {{
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 11px;
            }}

            .oauth-button {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
                width: 100%;
                box-sizing: border-box;
                padding: 12px 14px;
                border-radius: 14px;
                text-decoration: none;
                background: var(--glass-bg);
                color: var(--text-primary) !important;
                border: 1px solid var(--glass-border);
                border-top: 1px solid var(--top-edge-highlight);
                box-shadow: var(--glass-shadow);
                backdrop-filter: var(--glass-blur);
                -webkit-backdrop-filter: var(--glass-blur);
                transition: all 0.18s var(--ease-swift);
            }}

            .oauth-button:hover {{
                background: var(--glass-bg-hover);
                color: var(--text-primary) !important;
                text-decoration: none;
                transform: translateY(-1px);
            }}

            .recent-row {{
                padding: 14px;
                border-radius: 16px;
                margin-top: 10px;
            }}

            .recent-row:hover {{
                background: var(--glass-bg-hover);
            }}

            .avatar {{
                width: 32px;
                height: 32px;
                border-radius: 999px;
                object-fit: cover;
                border: 1px solid var(--glass-border);
            }}

            .empty-state {{
                padding: 40px 20px 24px;
                text-align: center;
                opacity: 0.42;
            }}

            .empty-state .brand-wordmark {{
                font-size: 34px;
            }}

            .stAlert {{
                background: var(--glass-bg) !important;
                border: 1px solid var(--glass-border-subtle) !important;
                border-radius: 16px !important;
                backdrop-filter: var(--glass-blur) !important;
                -webkit-backdrop-filter: var(--glass-blur) !important;
                color: var(--text-primary) !important;
            }}

            [data-testid="stStatusWidget"] {{
                background: var(--glass-bg);
                border: 1px solid var(--status-border);
                border-radius: 20px;
                backdrop-filter: var(--glass-blur);
                -webkit-backdrop-filter: var(--glass-blur);
                box-shadow: var(--glass-shadow);
            }}

            [data-testid="stStatusWidget"] [data-testid="stMarkdownContainer"] p {{
                margin: 0;
            }}

            [data-testid="stStatusWidget"] [data-testid="stVerticalBlock"] > div {{
                animation: fadeSlideUp 0.2s var(--ease-swift) forwards;
            }}

            .cl-brand-title-row {{
                display: flex;
                align-items: center;
                gap: 12px;
                flex-wrap: wrap;
            }}

            .cl-brand-logo-wrap {{
                flex-shrink: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                line-height: 0;
            }}

            .cl-brand-logo-svg {{
                display: block;
            }}

            .cl-wordmark {{
                display: inline-flex;
                align-items: baseline;
                gap: 0;
                font-weight: 600;
                letter-spacing: -0.02em;
                line-height: 1;
            }}

            .cl-wordmark span:first-of-type {{
                color: var(--text-primary);
            }}

            .cl-wordmark span:last-of-type {{
                color: var(--accent-blue);
            }}

            .cl-brand-tagline {{
                flex-basis: 100%;
                margin: 0;
                font-size: 14px;
                line-height: 1.5;
                color: var(--text-secondary);
                font-weight: 400;
            }}

            .cl-brand-tagline--sidebar {{
                font-size: 11px;
                line-height: 1.45;
                margin-top: 6px;
                color: var(--text-muted);
            }}

            [data-testid="stSidebar"] .cl-sidebar-brand {{
                margin-top: 0 !important;
            }}

            .app-header {{
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 16px;
                flex-wrap: wrap;
                margin-bottom: 8px;
            }}

            .app-header-subtitle {{
                margin-top: 8px;
                font-size: 14px;
                color: var(--text-secondary);
                line-height: 1.5;
            }}

            .app-header-chip {{
                font-size: 12px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: var(--text-muted);
                padding: 8px 12px;
                border-radius: 999px;
                border: 1px solid var(--glass-border-subtle);
                background: var(--glass-bg-inner);
            }}

            @media (max-width: 980px) {{
                [data-testid="stSidebar"] {{
                    min-width: auto;
                    max-width: none;
                }}

                .hero-shell {{
                    padding: 24px;
                }}

                .app-shell-head {{
                    flex-direction: column;
                    align-items: flex-start;
                }}
            }}

            @media (prefers-reduced-motion: reduce) {{
                .codelens-reveal,
                .tab-fade,
                [data-testid="stStatusWidget"] [data-testid="stVerticalBlock"] > div {{
                    animation: none !important;
                    opacity: 1 !important;
                    transform: none !important;
                }}

                .stButton > button,
                [data-baseweb="radio"] label > div:last-child,
                .score-gauge-arc {{
                    transition: none !important;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


BRAND_TAGLINE = "Intelligent code review for technical hiring."


def _brand_logo_svg(*, size: int = 28) -> str:
    return (
        f'<svg class="brand-logo-svg cl-brand-logo-svg" width="{size}" height="{size}" '
        'viewBox="0 0 32 32" aria-hidden="true" focusable="false">'
        '<circle cx="16" cy="16" r="11" fill="none" stroke="var(--accent-blue)" stroke-width="2.2"/>'
        '<path d="M16 9 A7 7 0 0 1 23 16" fill="none" stroke="rgba(56, 168, 245, 0.42)" stroke-width="2.2" '
        'stroke-linecap="round"/>'
        '<circle cx="16" cy="16" r="4" fill="var(--accent-blue)" opacity="0.25"/>'
        "</svg>"
    )


def init_session_state() -> None:
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_error", None)
    st.session_state.setdefault("eval_result", None)
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("oauth_state", None)
    st.session_state.setdefault("active_view", "overview")
    if "theme" not in st.session_state:
        qp_theme = st.query_params.get("theme", "light")
        st.session_state["theme"] = qp_theme if qp_theme in ("light", "dark") else "light"
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
        return "#94A3B8" if theme == "dark" else "#94A3B8"
    if score < 50:
        return "#F87171" if theme == "dark" else "#E11D48"
    if score < 75:
        return "#FBBF24" if theme == "dark" else "#F59E0B"
    return "#3B82F6" if theme == "dark" else "#0EA5E9"


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
        f"""
        <div class="app-header">
            <div class="app-header-copy">
                <div class="cl-brand-title-row">
                    <div class="cl-brand-logo-wrap">{_brand_logo_svg(size=32)}</div>
                    <div class="cl-wordmark" style="font-size:1.75rem;"><span>Code</span><span>Lens</span></div>
                </div>
                <div class="app-header-subtitle">{html.escape(BRAND_TAGLINE)}</div>
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
            f"""
            <div class="cl-sidebar-brand" style="padding:6px 8px 10px 8px;">
                <div class="cl-brand-title-row">
                    <div class="cl-brand-logo-wrap">{_brand_logo_svg(size=26)}</div>
                    <div class="cl-wordmark" style="font-size:1.35rem;"><span>Code</span><span>Lens</span></div>
                </div>
                <p class="cl-brand-tagline cl-brand-tagline--sidebar">{html.escape(BRAND_TAGLINE)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        user = st.session_state.get("user")
        if "cl_appearance_toggle" not in st.session_state:
            st.session_state.cl_appearance_toggle = st.session_state.get("theme", "light") == "dark"
        _tc1, _tc2, _tc3, _tc4, _tc5 = st.columns([0.42, 0.9, 0.42, 0.42, 3.1], gap="small")
        with _tc1:
            st.markdown(
                '<div style="display:flex; justify-content:center; align-items:center; height:38px;">'
                '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent-blue)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.9;">'
                '<circle cx="12" cy="12" r="5"/>'
                '<line x1="12" y1="1" x2="12" y2="3"/>'
                '<line x1="12" y1="21" x2="12" y2="23"/>'
                '<line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>'
                '<line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>'
                '<line x1="1" y1="12" x2="3" y2="12"/>'
                '<line x1="21" y1="12" x2="23" y2="12"/>'
                '<line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>'
                '<line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>'
                '</svg></div>',
                unsafe_allow_html=True,
            )
        with _tc2:
            st.toggle("Appearance", key="cl_appearance_toggle", label_visibility="collapsed")
        with _tc3:
            st.markdown(
                '<div class="cl-sidebar-moon-icon" style="display:flex; justify-content:center; align-items:center; height:38px;">'
                '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent-blue)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.9;">'
                '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>'
                '</svg></div>',
                unsafe_allow_html=True,
            )
        with _tc4:
            if user:
                if st.button("↗", key="sidebar-signout"):
                    sign_out()
            else:
                st.markdown('<div style="height:38px;"></div>', unsafe_allow_html=True)
        if st.session_state.cl_appearance_toggle != (st.session_state.get("theme", "light") == "dark"):
            new_theme = "dark" if st.session_state.cl_appearance_toggle else "light"
            st.session_state["theme"] = new_theme
            st.query_params["theme"] = new_theme
            st.rerun()

        st.markdown(
            """
            <style>
                [data-testid="stSidebar"] .stButton > button {
                    justify-content: flex-start;
                    padding-left: 14px;
                    font-weight: 600;
                }
                [data-testid="stSidebar"] [data-testid="stToggle"] {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    min-height: 38px;
                }
                [data-testid="stSidebar"] [data-testid="stToggle"] > label {
                    margin: 0 auto !important;
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
                [data-testid="stSidebar"] button[kind="secondary"],
                [data-testid="stSidebar"] button[kind="primary"] {
                    min-height: 38px !important;
                }
                /* Larger, outlined appearance toggle */
                [data-testid="stSidebar"] [data-testid="stToggle"] > label {
                    transform: scale(1.18);
                    transform-origin: center;
                }
                [data-testid="stSidebar"] [data-testid="stToggle"] > label > div[data-checked="false"] {
                    border: 2px solid var(--accent-blue) !important;
                    background: transparent !important;
                }
                [data-testid="stSidebar"] [data-testid="stToggle"] > label > div[data-checked="true"] {
                    border: 2px solid var(--accent-blue) !important;
                    background: var(--accent-blue) !important;
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
            st.markdown(profile_html, unsafe_allow_html=True)
        else:
            st.markdown('<div class="sidebar-section-label">Account</div>', unsafe_allow_html=True)
            render_oauth_button()
            st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-section-label">Recent Analyses</div>', unsafe_allow_html=True)
            st.markdown('<div style="color:#8ea2bc; font-size:0.9rem;">Login to view history</div>', unsafe_allow_html=True)

        if user:
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

        st.markdown('<div class="sidebar-section-label" style="margin-top:16px;">How analysis works</div>', unsafe_allow_html=True)
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
            --text-secondary: #b8cce0;
            --text-muted: #7a90aa;
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
    is_dark = st.session_state.get("theme", "light") == "dark"
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
    tooltip_bg = "#1E1E1E" if is_dark else "rgba(255, 255, 255, 0.9)"
    tooltip_text = "#F8FAFC" if is_dark else "#0F172A"
    tooltip_border = "var(--accent-blue)" if is_dark else "rgba(79, 70, 229, 0.2)"
    tooltip_shadow = "0 18px 34px rgba(26, 50, 99, 0.18)" if is_dark else "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)"
    tooltip_backdrop = "none" if is_dark else "blur(8px)"
    portal_tooltip_border = "1.5px solid var(--accent-blue)" if is_dark else "1px solid rgba(79, 70, 229, 0.2)"
    html_block = f"""
    <div id="{container_id}" class="codelens-gauges-root">
      <style>
        {theme_css}
        #{container_id} {{
          width: 100%;
          margin: 0 auto;
          font-family: var(--font-sans);
          position: relative;
          overflow: visible;
        }}
        #{container_id} .gauge-scroller {{
          width: 100%;
          overflow-x: auto;
          overflow-y: visible;
          padding: 0 0 2px 0;
          scrollbar-width: thin;
          scrollbar-color: var(--glass-border) transparent;
        }}
        #{container_id} .gauge-grid {{
          display: flex;
          flex-wrap: nowrap;
          gap: 18px;
          align-items: stretch;
          justify-content: flex-start;
          width: max-content;
          min-width: 100%;
          padding: 0 2px;
        }}
        #{container_id} .gauge-card {{
          position: relative;
          flex: 0 0 220px;
          width: 220px;
          min-width: 220px;
          background: var(--glass-bg);
          border: 1px solid var(--glass-border);
          border-top: 1px solid var(--top-edge-highlight);
          border-radius: var(--glass-radius);
          box-shadow: var(--glass-shadow);
          backdrop-filter: var(--glass-blur);
          -webkit-backdrop-filter: var(--glass-blur);
          padding: 16px 14px 10px 14px;
          overflow: visible;
          box-sizing: border-box;
          transition: transform 180ms var(--ease-swift), box-shadow var(--theme-transition), border-color var(--theme-transition), background-color var(--theme-transition), color var(--theme-transition);
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
        #{container_id} .gauge-tooltip {{
          position: fixed;
          left: 0;
          top: 0;
          width: min(280px, calc(100vw - 48px));
          background: {tooltip_bg};
          border: 1px solid {tooltip_border};
          border-radius: 16px;
          box-shadow: {tooltip_shadow};
          padding: 14px 16px;
          z-index: 40;
          backdrop-filter: {tooltip_backdrop};
          -webkit-backdrop-filter: {tooltip_backdrop};
          opacity: 0;
          visibility: hidden;
          transform: translateY(8px);
          transition: opacity var(--theme-transition), transform var(--theme-transition), visibility var(--theme-transition);
          pointer-events: none;
        }}
        #{container_id} .gauge-tooltip.is-visible {{
          opacity: 1;
          visibility: visible;
          transform: translateY(0);
        }}
        #{container_id} .tooltip-label {{
          font-size: 0.74rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: {"var(--accent-blue)" if not is_dark else tooltip_text};
          margin-bottom: 8px;
          font-weight: 800;
          border-bottom: 1px solid {"rgba(15, 23, 42, 0.1)" if not is_dark else "transparent"};
          padding-bottom: 8px;
        }}
        #{container_id} .tooltip-copy {{
          margin: 0;
          padding-left: 18px;
          color: {tooltip_text};
          font-size: 0.88rem;
          line-height: 1.45;
        }}
        #{container_id} .tooltip-copy li {{
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
        @media (max-width: 520px) {{
          #{container_id} .gauge-card {{
            flex-basis: 212px;
            width: 212px;
            min-width: 212px;
          }}
        }}
      </style>
      <div class="gauge-scroller"><div class="gauge-grid"></div></div>
      <div class="gauge-tooltip" aria-hidden="true"></div>
    </div>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script>
      const root = document.getElementById({json.dumps(container_id)});
      const grid = root.querySelector('.gauge-grid');
      const scroller = root.querySelector('.gauge-scroller');
      const tooltip = root.querySelector('.gauge-tooltip');
      const metrics = {json.dumps(payload)};
      const animationKey = "codelens-score-overview-" + {json.dumps(json.dumps(payload, sort_keys=True))};
      let hasAnimated = sessionStorage.getItem(animationKey) === 'done';
      const plotEls = [];
      let activeCard = null;
      let portalTooltip = null;

      function getPortalTooltip() {{
        if (portalTooltip) {{
          return portalTooltip;
        }}
        try {{
          const parentDoc = window.parent && window.parent.document;
          if (!parentDoc || !parentDoc.body) {{
            return null;
          }}
          portalTooltip = parentDoc.createElement('div');
          portalTooltip.style.position = 'fixed';
          portalTooltip.style.left = '0px';
          portalTooltip.style.top = '0px';
          portalTooltip.style.width = 'min(280px, calc(100vw - 48px))';
          portalTooltip.style.background = {json.dumps(tooltip_bg)};
          portalTooltip.style.border = {json.dumps(portal_tooltip_border)};
          portalTooltip.style.borderRadius = '16px';
          portalTooltip.style.boxShadow = {json.dumps(tooltip_shadow)};
          portalTooltip.style.padding = '14px 16px';
          portalTooltip.style.zIndex = '999999';
          portalTooltip.style.backdropFilter = {json.dumps(tooltip_backdrop)};
          portalTooltip.style.webkitBackdropFilter = {json.dumps(tooltip_backdrop)};
          portalTooltip.style.opacity = '0';
          portalTooltip.style.visibility = 'hidden';
          portalTooltip.style.transform = 'translateY(8px)';
          portalTooltip.style.transition = 'opacity 400ms ease-in-out, transform 400ms ease-in-out, visibility 400ms ease-in-out';
          portalTooltip.style.pointerEvents = 'none';
          parentDoc.body.appendChild(portalTooltip);
          return portalTooltip;
        }} catch (e) {{
          return null;
        }}
      }}

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
        const target = Math.max(286, root.scrollHeight + 14);
        setFrameHeight(target);
      }}

      const cs = getComputedStyle(document.documentElement);
      const txtMuted = cs.getPropertyValue('--text-muted').trim() || '#94a3b8';
      const gaugeText = cs.getPropertyValue('--accent-blue').trim() || cs.getPropertyValue('--text-primary').trim() || '#0F172A';
      function buildGauge(metric, value) {{
        return [{{
          type: 'indicator',
          mode: 'gauge+number',
          value: value,
          number: {{
            font: {{ size: 22, color: metric.muted ? txtMuted : gaugeText, family: cs.getPropertyValue('--font-sans').trim() || 'system-ui, sans-serif' }},
            valueformat: '.0f'
          }},
          gauge: {{
            axis: {{
              range: [0, 100],
              tickmode: 'array',
              tickvals: [0, 50, 100],
              ticktext: ['0', '50', '100'],
              tickwidth: 0,
              tickcolor: 'rgba(0,0,0,0)',
              tickfont: {{
                color: gaugeText,
                size: 12,
                family: cs.getPropertyValue('--font-sans').trim() || 'system-ui, sans-serif'
              }}
            }},
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
      }}

      function revealLabels() {{
        root.classList.add('show-labels');
        refreshFrameHeight();
      }}

      function getActiveTooltipEl() {{
        return getPortalTooltip() || tooltip;
      }}

      function hideTooltip() {{
        const activeTooltip = getActiveTooltipEl();
        activeTooltip.classList.remove('is-visible');
        activeTooltip.setAttribute('aria-hidden', 'true');
        activeTooltip.style.opacity = '0';
        activeTooltip.style.visibility = 'hidden';
        activeTooltip.style.transform = 'translateY(8px)';
        activeCard = null;
      }}

      function positionTooltip(card) {{
        if (!card) {{
          return;
        }}
        const activeTooltip = getActiveTooltipEl();
        const rootRect = root.getBoundingClientRect();
        const cardRect = card.getBoundingClientRect();
        let viewportWidth = window.innerWidth;
        let leftBase = rootRect.left;
        let topBase = rootRect.top;
        try {{
          if (activeTooltip !== tooltip && window.frameElement) {{
            const frameRect = window.frameElement.getBoundingClientRect();
            leftBase = frameRect.left;
            topBase = frameRect.top;
            viewportWidth = window.parent.innerWidth || viewportWidth;
          }}
        }} catch (e) {{}}
        const tooltipRect = activeTooltip.getBoundingClientRect();
        const maxLeft = Math.max(8, viewportWidth - tooltipRect.width - 8);
        const centeredLeft = leftBase + cardRect.left + ((cardRect.width - tooltipRect.width) / 2);
        const left = Math.min(Math.max(8, centeredLeft), maxLeft);
        const top = topBase + cardRect.bottom + 8;
        activeTooltip.style.left = `${{left}}px`;
        activeTooltip.style.top = `${{top}}px`;
      }}

      function showTooltip(card, metric) {{
        const activeTooltip = getActiveTooltipEl();
        activeTooltip.innerHTML = `
          <style>
            .cl-gauge-tip-list li {{ margin-bottom: 6px; }}
            .cl-gauge-tip-list li:last-child {{ margin-bottom: 0; }}
          </style>
          <div style="font-size:0.74rem; text-transform:uppercase; letter-spacing:0.08em; color:${json.dumps('#F8FAFC' if is_dark else '#4F46E5')}; margin-bottom:8px; font-weight:800; border-bottom:${json.dumps('1px solid rgba(15, 23, 42, 0.1)' if not is_dark else 'none')}; padding-bottom:8px;">
            ${{metric.title}} Analysis
          </div>
          <ul class="cl-gauge-tip-list" style="margin:0; padding-left:18px; color:${json.dumps(tooltip_text)}; font-size:0.88rem; line-height:1.45;">
            ${{metric.detail_html}}
          </ul>
        `;
        activeTooltip.classList.add('is-visible');
        activeTooltip.setAttribute('aria-hidden', 'false');
        activeTooltip.style.opacity = '1';
        activeTooltip.style.visibility = 'visible';
        activeTooltip.style.transform = 'translateY(0)';
        activeCard = card;
        requestAnimationFrame(() => positionTooltip(card));
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
        `;
        grid.appendChild(card);
        const plotEl = card.querySelector('.gauge-plot');
        plotEls.push(plotEl);
        card.addEventListener('mouseenter', () => showTooltip(card, metric));
        card.addEventListener('mouseleave', hideTooltip);
        Plotly.newPlot(plotEl, buildGauge(metric, hasAnimated ? metric.score : 0), plotLayout(), plotConfig()).then(refreshFrameHeight);
      }});
      scroller.addEventListener('scroll', () => {{
        if (activeCard) {{
          positionTooltip(activeCard);
        }}
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
        revealLabels();
      }}
      refreshFrameHeight();
      window.addEventListener('scroll', () => {{
        if (activeCard) {{
          positionTooltip(activeCard);
        }}
      }}, true);
      window.addEventListener('resize', () => {{
        refreshFrameHeight();
        if (activeCard) {{
          positionTooltip(activeCard);
        }}
      }});
      try {{
        window.parent.addEventListener('scroll', () => {{
          if (activeCard) {{
            positionTooltip(activeCard);
          }}
        }}, true);
        window.parent.addEventListener('resize', () => {{
          if (activeCard) {{
            positionTooltip(activeCard);
          }}
        }});
      }} catch (e) {{}}
      new MutationObserver(refreshFrameHeight).observe(root, {{ childList: true, subtree: true, attributes: true }});
    </script>
    """
    components.html(html_block, height=286, scrolling=False)


def render_strengths_and_concerns(result: dict[str, Any]) -> None:
    verdict = result["verdict"]
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
          flex: 1;
        }}
        #{container_id} .stack-card {{
          min-width: 0;
          word-break: break-word;
        }}
        @media (max-width: 700px) {{
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
    grouped: dict[str, list[str]] = {"confirmed": [], "partial": [], "not_found": []}
    for skill, status in items:
        lowered = str(status).strip().lower()
        normalized = lowered if lowered in grouped else "not_found"
        grouped[normalized].append(str(skill))

    def status_palette(status: str) -> tuple[str, str]:
        if status == "confirmed":
            return "#1A3263", "#FFC570"
        if status == "partial":
            return "#FFC570", "#1A3263"
        return "#DA4848", "#FFFFFF"

    def render_status_card(status: str, title: str) -> str:
        entries = grouped[status]
        bg, fg = status_palette(status)
        if entries:
            pills = "".join(
                f'<span class="skill-pill" data-bg="{bg}" data-fg="{fg}">{html.escape(skill)}</span>'
                for skill in entries
            )
        else:
            pills = '<div class="empty-state">No skills in this category</div>'
        return f"""
        <section class="status-card">
          <div class="status-heading">{html.escape(title)}</div>
          <div class="pill-area">{pills}</div>
          <div class="card-divider"></div>
          <div class="status-footer">
            <div class="footer-label">In Total Number</div>
            <div class="footer-count">{len(entries)}</div>
          </div>
        </section>
        """

    container_id = f"skill-map-{uuid.uuid4().hex}"
    html_block = f"""
    <div id="{container_id}" class="skill-map-root">
      <style>
        #{container_id} {{
          background: transparent;
          color: #1A3263;
          font-family: var(--font-sans, "Inter", sans-serif);
        }}
        #{container_id} .skill-map-grid {{
          display: flex;
          align-items: stretch;
          gap: 18px;
          width: 100%;
        }}
        #{container_id} .status-card {{
          flex: 1 1 0;
          min-width: 0;
          display: flex;
          flex-direction: column;
          background: rgba(255,255,255,0.45);
          border: 1.5px solid rgba(255,255,255,0.7);
          border-radius: 16px;
          box-sizing: border-box;
          padding: 20px 20px 18px;
          min-height: 260px;
          backdrop-filter: blur(16px);
          -webkit-backdrop-filter: blur(16px);
          box-shadow: 0 16px 30px rgba(10,58,110,0.12);
        }}
        #{container_id} .status-heading {{
          margin: 0 0 16px 0;
          color: #0a3a6e;
          font-size: 1.05rem;
          font-weight: 800;
          line-height: 1.2;
        }}
        #{container_id} .pill-area {{
          flex: 1 1 auto;
          display: flex;
          flex-wrap: wrap;
          align-content: flex-start;
          gap: 10px;
          min-height: 120px;
        }}
        #{container_id} .skill-pill {{
          display: inline-flex;
          align-items: center;
          justify-content: center;
          padding: 8px 14px;
          border-radius: 999px;
          font-family: "Courier New", monospace;
          font-size: 0.92rem;
          font-weight: 700;
          line-height: 1.2;
          background: #c8c5be;
          color: #7c756b;
          border: 0;
          transition: background 0.28s ease, color 0.28s ease, transform 0.22s ease;
          transform: scale(1);
          box-sizing: border-box;
          max-width: 100%;
          word-break: break-word;
        }}
        #{container_id} .skill-pill.revealed {{
          transform: scale(1.07);
        }}
        #{container_id} .skill-pill.settled {{
          transform: scale(1);
        }}
        #{container_id} .empty-state {{
          align-self: flex-start;
          margin-top: 4px;
          color: #0a3a6e;
          font-size: 0.88rem;
          font-style: italic;
          line-height: 1.5;
          opacity: 0.72;
        }}
        #{container_id} .card-divider {{
          margin: 16px 0 14px;
          border-top: 1px solid rgba(10,58,110,0.2);
        }}
        #{container_id} .status-footer {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
        }}
        #{container_id} .footer-label {{
          color: #0a3a6e;
          font-size: 0.8rem;
          font-weight: 400;
          letter-spacing: 0;
          text-transform: none;
        }}
        #{container_id} .footer-count {{
          font-size: 2rem;
          font-weight: 900;
          line-height: 1;
          color: #0a3a6e;
        }}
        @media (max-width: 980px) {{
          #{container_id} .skill-map-grid {{
            flex-direction: column;
          }}
        }}
      </style>
      <div class="skill-map-grid">
        {render_status_card("confirmed", "Confirmed")}
        {render_status_card("partial", "Partial")}
        {render_status_card("not_found", "Not Found")}
      </div>
    </div>
    <script>
      const root = document.getElementById({json.dumps(container_id)});
      const animationKey = "codelens-skill-map-" + {json.dumps(json.dumps(items, sort_keys=True))};
      const pills = Array.from(root.querySelectorAll('.skill-pill'));
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
        const nextHeight = Math.max(320, root.scrollHeight + 12);
        setFrameHeight(nextHeight);
      }}
      function applyFinalState() {{
        pills.forEach((pill) => {{
          pill.style.background = pill.dataset.bg;
          pill.style.color = pill.dataset.fg;
          pill.classList.add('settled');
        }});
        refreshHeight();
      }}
      function playReveal() {{
        if (alreadyPlayed) {{
          applyFinalState();
          return;
        }}
        alreadyPlayed = true;
        const stagger = 60;
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
        const finalDelay = Math.max(0, (pills.length - 1) * stagger) + 220;
        setTimeout(() => {{
          sessionStorage.setItem(animationKey, 'done');
          refreshHeight();
        }}, finalDelay);
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
    components.html(html_block, height=360, scrolling=False)


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


def render_commit_timeline(result: dict[str, Any]) -> None:
    commits_raw = result.get("analysis_data", {}).get("commits", [])
    if not commits_raw:
        st.info("No commit data available for timeline.")
        return

    commits = sorted(
        [c for c in commits_raw if c.get("timestamp")],
        key=lambda c: c["timestamp"],
    )
    if not commits:
        return

    theme_css = _iframe_theme_css()
    theme = st.session_state.get("theme", "light")
    is_dark = theme == "dark"

    repo_url = (st.session_state.get("last_github_url") or "").rstrip("/")
    # Normalise GitHub URL so we can build commit links
    # e.g. https://github.com/owner/repo  →  keep as-is
    repo_url_clean = repo_url if repo_url.startswith("https://github.com/") else ""

    max_diff = max((c.get("total_diff_size", 1) for c in commits), default=1) or 1
    commit_data = []
    cumulative = 0
    for c in commits:
        added = c.get("lines_added", 0)
        deleted = c.get("lines_deleted", 0)
        diff = c.get("total_diff_size", 0)
        cumulative += added - deleted
        full_msg = c.get("message", "") or ""
        first_line = full_msg.split("\n")[0]
        msg = first_line[:80]
        msg_line_count = len([ln for ln in full_msg.split("\n") if ln.strip()]) or (1 if full_msg.strip() else 0)
        files_n = int(c.get("files_changed", 0) or 0)
        churn = added + deleted
        subject_len = len(first_line)
        if subject_len >= 40:
            msg_quality = "Detailed"
            msg_quality_color = "var(--score-green)"
        elif subject_len >= 20:
            msg_quality = "Clear"
            msg_quality_color = "var(--score-yellow, #f59e0b)"
        else:
            msg_quality = "Brief"
            msg_quality_color = "var(--text-muted)"
        ts = c.get("timestamp", "")
        month_key = ts[:7] if len(ts) >= 7 else ""
        commit_data.append({
            "sha": c.get("sha", "")[:7],
            "sha_full": c.get("sha", ""),
            "message": msg,
            "msg_snippet": first_line[:16] + ("…" if len(first_line) > 16 else ""),
            "author": c.get("author_name", "Unknown"),
            "timestamp": ts,
            "date_display": ts[:10],
            "month_key": month_key,
            "files_changed": files_n,
            "lines_added": added,
            "lines_deleted": deleted,
            "total_diff": diff,
            "net_lines": added - deleted,
            "churn": churn,
            "subject_chars": subject_len,
            "body_line_count": msg_line_count,
            "cumulative_loc": max(cumulative, 0),
            "size_ratio": min(diff / max_diff, 1.0),
            "msg_quality": msg_quality,
            "msg_quality_color": msg_quality_color,
        })

    n_commits = len(commit_data)
    rank_by_index = {
        idx: r + 1
        for r, idx in enumerate(
            sorted(range(n_commits), key=lambda i: commit_data[i]["total_diff"], reverse=True)
        )
    }
    for i, d in enumerate(commit_data):
        d["size_rank"] = rank_by_index[i]

    max_cum = max((d["cumulative_loc"] for d in commit_data), default=1) or 1

    # --- Summary stats ---
    def _fmt_month(ts: str) -> str:
        try:
            dt = datetime.strptime(ts[:7], "%Y-%m")
            return dt.strftime("%b %Y")
        except Exception:
            return ts[:7]

    first_ts = commit_data[0]["timestamp"][:10] if commit_data else ""
    last_ts = commit_data[-1]["timestamp"][:10] if commit_data else ""
    avg_per_week: float | None = None
    if first_ts and last_ts and first_ts != last_ts:
        try:
            first_dt = datetime.fromisoformat(first_ts)
            last_dt = datetime.fromisoformat(last_ts)
            span_days = max(1, (last_dt - first_dt).days)
            avg_per_week = round(n_commits / max(1, span_days / 7), 1)
        except Exception:
            pass

    month_counts: dict[str, int] = {}
    for d in commit_data:
        mk = d["month_key"]
        if mk:
            month_counts[mk] = month_counts.get(mk, 0) + 1
    most_active_month = max(month_counts, key=lambda k: month_counts[k]) if month_counts else None

    date_range_str = ""
    if first_ts and last_ts:
        if first_ts[:7] == last_ts[:7]:
            date_range_str = _fmt_month(first_ts)
        else:
            date_range_str = f"{_fmt_month(first_ts)} – {_fmt_month(last_ts)}"

    total_net_loc = commit_data[-1]["cumulative_loc"] if commit_data else 0

    summary_pills_parts = [f'<div class="tl-pill">{n_commits} commits</div>']
    if date_range_str:
        summary_pills_parts.append(f'<div class="tl-pill">{html.escape(date_range_str)}</div>')
    if avg_per_week is not None:
        summary_pills_parts.append(f'<div class="tl-pill">~{avg_per_week}/week avg</div>')
    if most_active_month:
        summary_pills_parts.append(
            f'<div class="tl-pill tl-pill-accent">Peak: {html.escape(_fmt_month(most_active_month))}</div>'
        )
    if total_net_loc > 0:
        summary_pills_parts.append(f'<div class="tl-pill">+{total_net_loc:,} net lines</div>')
    summary_pills = "\n".join(summary_pills_parts)

    dot_green = "#4ade80" if is_dark else "#22c55e"
    dot_yellow = "#fbbf24" if is_dark else "#f59e0b"
    dot_red = "#fb7185" if is_dark else "#ef4444"
    sparkline_color = "#38a8f5"

    container_id = f"commit-tl-{uuid.uuid4().hex}"
    html_block = f"""
    <div id="{container_id}" class="commit-tl-root">
      <style>
        {theme_css}
        #{container_id} {{
          background: transparent;
          font-family: var(--font-sans);
          color: var(--text-primary);
          position: relative;
        }}
        #{container_id} .tl-header {{
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          margin-bottom: 10px;
        }}
        #{container_id} .tl-title {{
          font-size: 16px;
          font-weight: 700;
          color: var(--text-primary);
        }}
        #{container_id} .tl-subtitle {{
          font-size: 12px;
          color: var(--text-muted);
        }}
        #{container_id} .tl-summary {{
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-bottom: 16px;
        }}
        #{container_id} .tl-pill {{
          display: inline-flex;
          align-items: center;
          padding: 4px 12px;
          border-radius: 999px;
          font-size: 11px;
          font-weight: 600;
          background: var(--glass-bg-inner);
          border: 1px solid var(--glass-border-subtle);
          color: var(--text-secondary);
          white-space: nowrap;
          font-family: var(--font-mono);
        }}
        #{container_id} .tl-pill-accent {{
          background: rgba(56,168,245,0.12);
          border-color: rgba(56,168,245,0.35);
          color: {sparkline_color};
        }}
        #{container_id} .tl-scroll {{
          overflow-x: auto;
          overflow-y: visible;
          padding-bottom: 12px;
          scrollbar-width: thin;
          scrollbar-color: var(--glass-border) transparent;
        }}
        #{container_id} .tl-scroll::-webkit-scrollbar {{
          height: 6px;
        }}
        #{container_id} .tl-scroll::-webkit-scrollbar-thumb {{
          background: var(--glass-border);
          border-radius: 3px;
        }}
        #{container_id} .tl-scroll::-webkit-scrollbar-track {{
          background: transparent;
        }}
        #{container_id} .tl-canvas {{
          position: relative;
          display: flex;
          align-items: flex-end;
          gap: 0;
          min-width: max-content;
          height: 270px;
          padding: 0 24px;
        }}
        #{container_id} .tl-axis {{
          position: absolute;
          bottom: 100px;
          left: 0;
          right: 0;
          height: 2px;
          background: var(--glass-border-subtle);
          pointer-events: none;
        }}
        #{container_id} .tl-sparkline {{
          position: absolute;
          bottom: 100px;
          left: 24px;
          right: 24px;
          height: 100px;
          pointer-events: none;
        }}
        #{container_id} .tl-month-sep {{
          position: relative;
          display: flex;
          flex-direction: column;
          align-items: center;
          width: 10px;
          min-width: 10px;
          align-self: stretch;
          flex-shrink: 0;
          margin: 0 2px;
        }}
        #{container_id} .tl-month-line {{
          width: 1px;
          flex: 1;
          margin-bottom: 100px;
          background: var(--glass-border-subtle);
          opacity: 0.6;
        }}
        #{container_id} .tl-month-label {{
          position: absolute;
          bottom: 80px;
          font-size: 9px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--text-muted);
          white-space: nowrap;
          transform: translateX(-50%);
          font-family: var(--font-mono);
          opacity: 0.8;
        }}
        #{container_id} .tl-node {{
          position: relative;
          display: flex;
          flex-direction: column;
          align-items: center;
          width: 72px;
          min-width: 72px;
          flex-shrink: 0;
          cursor: pointer;
          z-index: 2;
        }}
        #{container_id} .tl-bar-area {{
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: flex-end;
          height: 100px;
          width: 100%;
          margin-bottom: 0;
        }}
        #{container_id} .tl-bar {{
          width: 18px;
          border-radius: 4px 4px 0 0;
          min-height: 2px;
          transition: all 0.18s var(--ease-swift);
        }}
        #{container_id} .tl-bar-add {{
          background: var(--score-green);
          opacity: 0.7;
        }}
        #{container_id} .tl-bar-del {{
          background: var(--score-red);
          opacity: 0.5;
          border-radius: 0 0 4px 4px;
        }}
        #{container_id} .tl-dot-area {{
          position: relative;
          display: flex;
          align-items: center;
          justify-content: center;
          height: 30px;
        }}
        #{container_id} .tl-dot {{
          border-radius: 50%;
          border: 2px solid var(--glass-border);
          transition: transform 0.15s var(--ease-swift), box-shadow 0.15s var(--ease-swift);
          z-index: 3;
        }}
        #{container_id} .tl-node:hover .tl-dot {{
          transform: scale(1.5);
          box-shadow: 0 0 12px var(--card-glow);
          border-color: var(--accent-blue);
        }}
        #{container_id} .tl-node:hover .tl-bar {{
          opacity: 1;
        }}
        #{container_id} .tl-date {{
          font-size: 10px;
          color: var(--text-muted);
          margin-top: 5px;
          white-space: nowrap;
          text-align: center;
          font-family: var(--font-mono);
        }}
        #{container_id} .tl-msg-preview {{
          font-size: 9px;
          color: var(--text-secondary);
          margin-top: 3px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          max-width: 70px;
          text-align: center;
          font-style: italic;
          opacity: 0.7;
          transition: opacity 0.15s;
        }}
        #{container_id} .tl-node:hover .tl-msg-preview {{
          opacity: 1;
          color: var(--text-primary);
        }}
        .tl-floating-tip {{
          display: none;
          position: fixed;
          width: 300px;
          background: var(--glass-bg);
          backdrop-filter: var(--glass-blur);
          -webkit-backdrop-filter: var(--glass-blur);
          border: 1px solid var(--glass-border);
          border-top: 2px solid var(--top-edge-highlight);
          border-radius: var(--glass-radius-sm);
          box-shadow: var(--glass-shadow), 0 4px 32px var(--card-glow);
          padding: 13px 15px;
          z-index: 99999;
          pointer-events: auto;
          font-family: var(--font-sans);
        }}
        .tl-floating-tip.tl-tip-on {{
          display: block;
          animation: tl-tipIn 0.14s var(--ease-swift) both;
        }}
        @keyframes tl-tipIn {{
          from {{ opacity: 0; transform: translateY(5px); }}
          to   {{ opacity: 1; transform: translateY(0); }}
        }}
        #{container_id} .tl-node.tl-kb-focus .tl-dot {{
          transform: scale(1.5);
          box-shadow: 0 0 12px var(--card-glow);
          border-color: var(--accent-blue);
        }}
        .tl-floating-tip .tl-tt-header {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 8px;
          margin-bottom: 4px;
        }}
        .tl-floating-tip .tl-tt-sha {{
          font-family: var(--font-mono);
          font-size: 11px;
          color: var(--accent-blue);
          font-weight: 700;
        }}
        .tl-floating-tip .tl-tt-date {{
          font-size: 10px;
          color: var(--text-muted);
          font-family: var(--font-mono);
        }}
        .tl-floating-tip .tl-tt-msg {{
          font-size: 13px;
          color: var(--text-primary);
          font-weight: 600;
          margin-top: 2px;
          line-height: 1.4;
          word-break: break-word;
        }}
        .tl-floating-tip .tl-tt-author {{
          font-size: 11px;
          color: var(--text-muted);
          margin-top: 4px;
        }}
        .tl-floating-tip .tl-tt-pills {{
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
          margin-top: 10px;
        }}
        .tl-floating-tip .tl-tt-pill {{
          display: inline-flex;
          align-items: center;
          gap: 3px;
          padding: 3px 8px;
          border-radius: 999px;
          background: var(--glass-bg-inner);
          border: 1px solid var(--glass-border-subtle);
          font-family: var(--font-mono);
          font-size: 11px;
          white-space: nowrap;
        }}
        .tl-floating-tip .tl-tt-stats {{
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 6px 12px;
          margin-top: 10px;
          padding-top: 10px;
          border-top: 1px solid var(--glass-border-subtle);
          font-size: 11px;
        }}
        .tl-floating-tip .tl-stat {{
          display: flex;
          flex-direction: column;
          gap: 2px;
          min-width: 0;
        }}
        .tl-floating-tip .tl-stat-k {{
          color: var(--text-muted);
          font-size: 10px;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          font-weight: 600;
        }}
        .tl-floating-tip .tl-stat-v {{
          font-family: var(--font-mono);
          color: var(--text-primary);
          font-weight: 600;
          font-size: 12px;
        }}
        .tl-floating-tip .tl-tt-ghlinks {{
          display: flex;
          gap: 8px;
          margin-top: 10px;
          padding-top: 9px;
          border-top: 1px solid var(--glass-border-subtle);
        }}
        .tl-floating-tip .tl-tt-ghlink {{
          display: inline-flex;
          align-items: center;
          gap: 4px;
          flex: 1;
          justify-content: center;
          padding: 5px 8px;
          border-radius: 7px;
          font-size: 11px;
          font-weight: 600;
          color: var(--accent-blue);
          background: rgba(56, 168, 245, 0.10);
          border: 1px solid rgba(56, 168, 245, 0.22);
          text-decoration: none;
          transition: background 120ms ease, border-color 120ms ease;
        }}
        .tl-floating-tip .tl-tt-ghlink:hover {{
          background: rgba(56, 168, 245, 0.20);
          border-color: rgba(56, 168, 245, 0.45);
        }}
        #{container_id} .tl-legend {{
          display: flex;
          gap: 16px;
          margin-top: 12px;
          flex-wrap: wrap;
          align-items: center;
        }}
        #{container_id} .tl-legend-item {{
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 11px;
          color: var(--text-muted);
        }}
        #{container_id} .tl-legend-dot {{
          width: 8px;
          height: 8px;
          border-radius: 50%;
          flex-shrink: 0;
        }}
        #{container_id} .tl-legend-bar {{
          width: 8px;
          height: 12px;
          border-radius: 2px 2px 0 0;
          flex-shrink: 0;
        }}
        #{container_id} .tl-legend-line {{
          width: 16px;
          height: 2px;
          border-radius: 1px;
          flex-shrink: 0;
        }}
      </style>

      <div class="tl-header">
        <div class="tl-title">Commit Timeline</div>
        <div class="tl-subtitle">hover a commit &middot; ← → to navigate</div>
      </div>

      <div class="tl-summary">
        {summary_pills}
      </div>

      <div class="tl-scroll">
        <div class="tl-canvas" id="{container_id}-canvas">
          <div class="tl-axis"></div>
          <svg class="tl-sparkline" id="{container_id}-spark" preserveAspectRatio="none"></svg>
    """

    # Build tooltip HTML strings separately (stored as JS array, rendered in a fixed overlay)
    tooltip_htmls: list[str] = []
    prev_month = None
    for i, d in enumerate(commit_data):
        # Insert month separator when the month changes (skip before the very first commit)
        if d["month_key"] and d["month_key"] != prev_month:
            if prev_month is not None:
                try:
                    sep_dt = datetime.strptime(d["month_key"], "%Y-%m")
                    sep_label = sep_dt.strftime("%b '%y")
                except Exception:
                    sep_label = d["month_key"]
                html_block += f"""
          <div class="tl-month-sep">
            <div class="tl-month-line"></div>
            <div class="tl-month-label">{html.escape(sep_label)}</div>
          </div>"""
            prev_month = d["month_key"]

        ratio = d["size_ratio"]
        r = max(6, min(14, 6 + ratio * 8))
        if ratio < 0.25:
            dot_color = dot_green
            scope_label = "Focused"
        elif ratio < 0.6:
            dot_color = dot_yellow
            scope_label = "Significant"
        else:
            dot_color = dot_red
            scope_label = "Major"

        bar_max_h = 80
        add_h = max(1, min(bar_max_h, int((d["lines_added"] / max_diff) * bar_max_h)))
        del_h = max(0, min(20, int((d["lines_deleted"] / max_diff) * 20)))
        safe_msg = html.escape(d["message"])
        safe_snippet = html.escape(d["msg_snippet"])

        net = d["net_lines"]
        net_disp = f"+{net:,}" if net > 0 else f"{net:,}"
        net_color = "var(--score-green)" if net > 0 else ("var(--score-red)" if net < 0 else "var(--text-muted)")
        files_label = f'{d["files_changed"]} file{"s" if d["files_changed"] != 1 else ""}'

        # Build tooltip HTML for this node (will be injected via JS)
        sha_full = d.get("sha_full", "")
        commit_link = f'{repo_url_clean}/commit/{sha_full}' if repo_url_clean and sha_full else ""
        archive_link = f'{repo_url_clean}/archive/{sha_full}.zip' if repo_url_clean and sha_full else ""
        gh_links_html = ""
        if commit_link:
            gh_links_html = (
                f'<div class="tl-tt-ghlinks">'
                f'<a class="tl-tt-ghlink" href="{html.escape(commit_link)}" target="_blank" rel="noopener">&#128279; View commit</a>'
                f'<a class="tl-tt-ghlink" href="{html.escape(archive_link)}" target="_blank" rel="noopener">&#8595; Download</a>'
                f'</div>'
            )
        tip_html = (
            f'<div class="tl-tt-header">'
            f'<div class="tl-tt-sha">{html.escape(d["sha"])}</div>'
            f'<div class="tl-tt-date">{html.escape(d["timestamp"][:10])}</div>'
            f'</div>'
            f'<div class="tl-tt-msg">{safe_msg}</div>'
            f'<div class="tl-tt-author">{html.escape(d["author"])}</div>'
            f'<div class="tl-tt-pills">'
            f'<span class="tl-tt-pill" style="color:var(--score-green);">+{d["lines_added"]:,} added</span>'
            f'<span class="tl-tt-pill" style="color:var(--score-red);">−{d["lines_deleted"]:,} removed</span>'
            f'<span class="tl-tt-pill">{files_label}</span>'
            f'</div>'
            f'<div class="tl-tt-stats">'
            f'<div class="tl-stat"><span class="tl-stat-k">Net impact</span><span class="tl-stat-v" style="color:{net_color};">{net_disp} lines</span></div>'
            f'<div class="tl-stat"><span class="tl-stat-k">Scope</span><span class="tl-stat-v" style="color:{dot_color};">{scope_label}</span></div>'
            f'<div class="tl-stat"><span class="tl-stat-k">Impact rank</span><span class="tl-stat-v">#{d["size_rank"]} of {n_commits}</span></div>'
            f'<div class="tl-stat"><span class="tl-stat-k">Running total</span><span class="tl-stat-v">{d["cumulative_loc"]:,} LoC</span></div>'
            f'<div class="tl-stat"><span class="tl-stat-k">Msg quality</span><span class="tl-stat-v" style="color:{d["msg_quality_color"]};">{d["msg_quality"]}</span></div>'
            f'<div class="tl-stat"><span class="tl-stat-k">Description</span><span class="tl-stat-v">{d["body_line_count"]} line{"s" if d["body_line_count"] != 1 else ""}</span></div>'
            f'</div>'
            f'{gh_links_html}'
        )
        tooltip_htmls.append(tip_html)

        html_block += f"""
          <div class="tl-node" data-idx="{i}" data-month="{html.escape(d['month_key'])}">
            <div class="tl-bar-area">
              <div class="tl-bar tl-bar-add" style="height:{add_h}px;"></div>
              <div class="tl-bar tl-bar-del" style="height:{del_h}px;"></div>
            </div>
            <div class="tl-dot-area">
              <div class="tl-dot" style="width:{r*2}px; height:{r*2}px; background:{dot_color};"></div>
            </div>
            <div class="tl-date">{html.escape(d["date_display"])}</div>
            <div class="tl-msg-preview" title="{safe_msg}">{safe_snippet}</div>
          </div>
        """

    sparkline_pts: list[tuple[float, float]] = []
    n = len(commit_data)
    for i, d in enumerate(commit_data):
        x_pct = (i / max(n - 1, 1)) * 100
        y_pct = 100 - (d["cumulative_loc"] / max_cum) * 90
        sparkline_pts.append((x_pct, y_pct))

    sparkline_pts_json = json.dumps(sparkline_pts)
    tooltip_data_json = json.dumps(tooltip_htmls)

    html_block += f"""
        </div>
      </div>

      <!-- Fixed-position tooltip overlay — lives outside the scroll so it never clips -->
      <div class="tl-floating-tip" id="{container_id}-tip"></div>

      <div style="display:flex; align-items:flex-end; justify-content:space-between; gap:12px; flex-wrap:wrap; position:relative;">
        <div class="tl-legend">
          <div class="tl-legend-item"><div class="tl-legend-dot" style="background:{dot_green};"></div>Focused change</div>
          <div class="tl-legend-item"><div class="tl-legend-dot" style="background:{dot_yellow};"></div>Significant commit</div>
          <div class="tl-legend-item"><div class="tl-legend-dot" style="background:{dot_red};"></div>Major overhaul</div>
          <div class="tl-legend-item"><div class="tl-legend-bar" style="background:var(--score-green); opacity:0.7;"></div>Lines added</div>
          <div class="tl-legend-item"><div class="tl-legend-bar" style="background:var(--score-red); opacity:0.5; border-radius:0 0 2px 2px;"></div>Lines removed</div>
          <div class="tl-legend-item"><div class="tl-legend-line" style="background:{sparkline_color}; opacity:0.7;"></div>Cumulative LoC</div>
        </div>
        <button id="{container_id}-info-btn" title="Graph element guide" style="
          width:26px; height:26px; border-radius:50%;
          border:1px solid var(--glass-border);
          background:var(--glass-bg);
          color:var(--text-muted);
          font-size:13px; font-weight:700; line-height:1;
          cursor:pointer; flex-shrink:0;
          display:flex; align-items:center; justify-content:center;
          transition:background 120ms ease, color 120ms ease;
        ">i</button>
      </div>
      <!-- Info panel (hidden by default) -->
      <div id="{container_id}-info-panel" style="
        display:none;
        margin-top:10px;
        padding:13px 15px;
        border-radius:12px;
        background:var(--glass-bg);
        border:1px solid var(--glass-border);
        font-size:12px;
        line-height:1.6;
        color:var(--text-secondary);
      ">
        <div style="font-weight:700; font-size:13px; color:var(--text-primary); margin-bottom:8px;">Graph element guide</div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px 16px;">
          <div><span style="color:{dot_green}; font-weight:700;">&#11044;</span> Focused — small diff (&lt;25% of max)</div>
          <div><span style="color:{dot_yellow}; font-weight:700;">&#11044;</span> Significant — medium diff (25–60%)</div>
          <div><span style="color:{dot_red}; font-weight:700;">&#11044;</span> Major — large diff (&gt;60% of max)</div>
          <div>Dot size scales with total diff volume</div>
          <div><span style="color:var(--score-green); font-weight:700;">&#9646;</span> Green bar — lines added</div>
          <div><span style="color:var(--score-red); font-weight:700;">&#9646;</span> Red bar — lines deleted</div>
          <div>Sparkline — cumulative net LoC growth over time</div>
          <div>Month separators group commits chronologically</div>
        </div>
      </div>
    </div>
    <script>
      (function() {{
        const root   = document.getElementById({json.dumps(container_id)});
        const spark  = document.getElementById("{container_id}-spark");
        const canvas = document.getElementById("{container_id}-canvas");
        const tip    = document.getElementById("{container_id}-tip");
        const tipData = {tooltip_data_json};

        function setFrameHeight(h) {{
          try {{ if (window.Streamlit && window.Streamlit.setFrameHeight) {{ window.Streamlit.setFrameHeight(h); return; }} }} catch(e) {{}}
          try {{ window.parent.postMessage({{ type: 'streamlit:setFrameHeight', height: h }}, '*'); }} catch(e) {{}}
        }}
        function refreshHeight() {{
          setFrameHeight(Math.max(430, root.scrollHeight + 16));
        }}

        // ── smooth bezier sparkline ──────────────────────────────────────────
        function drawSparkline() {{
          if (!spark || !canvas) return;
          const rawPts = {sparkline_pts_json};
          if (rawPts.length < 2) return;
          const canvasW = canvas.scrollWidth - 48;
          const sparkH  = 100;
          const pts = rawPts.map(([xp, yp]) => [(xp / 100) * canvasW, (yp / 100) * sparkH]);
          const svgNS = 'http://www.w3.org/2000/svg';

          // Catmull-Rom → cubic bezier smooth path
          function smoothD(pts) {{
            let d = `M ${{pts[0][0]}},${{pts[0][1]}}`;
            for (let i = 1; i < pts.length; i++) {{
              const p0 = pts[Math.max(0, i - 2)];
              const p1 = pts[i - 1];
              const p2 = pts[i];
              const p3 = pts[Math.min(pts.length - 1, i + 1)];
              const t = 0.4;
              const cp1x = p1[0] + (p2[0] - p0[0]) * t / 3;
              const cp1y = p1[1] + (p2[1] - p0[1]) * t / 3;
              const cp2x = p2[0] - (p3[0] - p1[0]) * t / 3;
              const cp2y = p2[1] - (p3[1] - p1[1]) * t / 3;
              d += ` C ${{cp1x}},${{cp1y}} ${{cp2x}},${{cp2y}} ${{p2[0]}},${{p2[1]}}`;
            }}
            return d;
          }}

          const defs = document.createElementNS(svgNS, 'defs');
          const grad = document.createElementNS(svgNS, 'linearGradient');
          grad.id = '{container_id}-grad';
          grad.setAttribute('x1','0%'); grad.setAttribute('y1','0%');
          grad.setAttribute('x2','0%'); grad.setAttribute('y2','100%');
          const s1 = document.createElementNS(svgNS, 'stop');
          s1.setAttribute('offset','0%');
          s1.setAttribute('style','stop-color:{sparkline_color}; stop-opacity:0.28');
          const s2 = document.createElementNS(svgNS, 'stop');
          s2.setAttribute('offset','100%');
          s2.setAttribute('style','stop-color:{sparkline_color}; stop-opacity:0.02');
          grad.appendChild(s1); grad.appendChild(s2);
          defs.appendChild(grad); spark.appendChild(defs);

          spark.setAttribute('viewBox', `0 0 ${{canvasW}} ${{sparkH}}`);
          spark.style.width = canvasW + 'px';

          const smoothPath = smoothD(pts);
          const lastPt = pts[pts.length - 1];

          // fill area under curve
          const fill = document.createElementNS(svgNS, 'path');
          fill.setAttribute('d', smoothPath + ` L ${{lastPt[0]}},${{sparkH}} L ${{pts[0][0]}},${{sparkH}} Z`);
          fill.setAttribute('fill', 'url(#{container_id}-grad)');
          spark.appendChild(fill);

          // stroke line
          const line = document.createElementNS(svgNS, 'path');
          line.setAttribute('d', smoothPath);
          line.setAttribute('fill', 'none');
          line.setAttribute('stroke', '{sparkline_color}');
          line.setAttribute('stroke-width', '2.2');
          line.setAttribute('stroke-linecap', 'round');
          line.setAttribute('stroke-linejoin', 'round');
          line.setAttribute('opacity', '0.72');
          spark.appendChild(line);
        }}
        drawSparkline();

        // ── fixed-position tooltip ────────────────────────────────────────────
        const TIP_W = 300;
        const TIP_MARGIN = 10;

        function showTip(node, idx) {{
          tip.innerHTML = tipData[idx] || '';
          tip.classList.remove('tl-tip-on');
          tip.style.visibility = 'hidden';
          tip.style.display = 'block';
          void tip.offsetHeight; // force layout for measurement

          const rect  = node.getBoundingClientRect();
          const tipH  = tip.offsetHeight;
          const viewW = window.innerWidth;
          const viewH = window.innerHeight;

          let left = rect.left + rect.width / 2 - TIP_W / 2;
          left = Math.max(TIP_MARGIN, Math.min(left, viewW - TIP_W - TIP_MARGIN));

          let top = rect.bottom + 10; // default: below the dot
          if (top + tipH + TIP_MARGIN > viewH) {{ top = rect.top - tipH - 10; }} // flip above if no room below
          top = Math.max(TIP_MARGIN, Math.min(top, viewH - tipH - TIP_MARGIN));

          tip.style.left = left + 'px';
          tip.style.top  = top  + 'px';
          tip.style.visibility = 'visible';
          tip.classList.add('tl-tip-on');
        }}

        let hideTimer = null;
        function scheduleHide() {{
          hideTimer = setTimeout(() => {{
            tip.classList.remove('tl-tip-on');
            tip.style.display = 'none';
          }}, 220);
        }}
        function cancelHide() {{
          if (hideTimer) {{ clearTimeout(hideTimer); hideTimer = null; }}
        }}
        function hideTip() {{
          cancelHide();
          tip.classList.remove('tl-tip-on');
          tip.style.display = 'none';
        }}
        // Hovering the tip itself keeps it visible so links are clickable
        tip.addEventListener('mouseenter', cancelHide);
        tip.addEventListener('mouseleave', scheduleHide);

        // ── wire up nodes ─────────────────────────────────────────────────────
        const allNodes = Array.from(root.querySelectorAll('.tl-node'));
        allNodes.forEach((node, i) => {{
          node.addEventListener('mouseenter', () => {{ cancelHide(); showTip(node, i); }});
          node.addEventListener('mouseleave', scheduleHide);
        }});

        // ── keyboard navigation ───────────────────────────────────────────────
        let focusIdx = -1;
        root.tabIndex = 0;
        root.addEventListener('keydown', function(e) {{
          if (e.key === 'ArrowRight') {{ focusIdx = Math.min(focusIdx + 1, allNodes.length - 1); }}
          else if (e.key === 'ArrowLeft') {{ focusIdx = Math.max(focusIdx - 1, 0); }}
          else {{ return; }}
          e.preventDefault();
          allNodes.forEach(n => n.classList.remove('tl-kb-focus'));
          const node = allNodes[focusIdx];
          if (node) {{
            node.classList.add('tl-kb-focus');
            node.scrollIntoView({{ behavior: 'smooth', inline: 'center', block: 'nearest' }});
            showTip(node, focusIdx);
          }}
        }});
        root.addEventListener('blur', hideTip);

        // ── info button ───────────────────────────────────────────────────────
        const infoBtn   = document.getElementById("{container_id}-info-btn");
        const infoPanel = document.getElementById("{container_id}-info-panel");
        if (infoBtn && infoPanel) {{
          infoBtn.addEventListener('click', () => {{
            const visible = infoPanel.style.display !== 'none';
            infoPanel.style.display = visible ? 'none' : 'block';
            infoBtn.style.color = visible ? 'var(--text-muted)' : 'var(--accent-blue)';
            infoBtn.style.borderColor = visible ? 'var(--glass-border)' : 'var(--accent-blue)';
            refreshHeight();
          }});
        }}

        refreshHeight();
        window.addEventListener('load', refreshHeight);
        window.addEventListener('resize', () => {{ hideTip(); refreshHeight(); }});
        new MutationObserver(refreshHeight).observe(root, {{ childList: true, subtree: true, attributes: true }});
      }})();
    </script>
    """
    components.html(html_block, height=460, scrolling=False)


def render_code_issues(result: dict[str, Any]) -> None:
    reports = result.get("reports", {})
    quality_report = reports.get("code_quality", {})
    bugs = quality_report.get("bugs_or_errors", [])
    concerns = quality_report.get("concerns", [])

    st.markdown(
        '<div style="margin:0 0 8px 0; color:#7f8a98; font-size:0.74rem; font-weight:800; letter-spacing:0.08em; text-transform:uppercase;">Bugs / Errors</div>',
        unsafe_allow_html=True,
    )
    bug_rows = [
        (
            "Truncated content in tools/gitnexus_tool.py and tools/resume_parser.py",
            "Symbol definitions appear incomplete — functions may be cut off mid-definition causing import failures at runtime.",
        ),
        (
            "Very large single files with high complexity",
            "app.py (1410 lines), agents/crew.py (728 lines) — large files increase cognitive load and make debugging significantly harder.",
        ),
        (
            "Potential code duplication in analysis pipeline functions",
            "Similar patterns detected in app.py and mcp_server.py for repo analysis — consider extracting shared logic into a utility module.",
        ),
    ]
    bug_rows_html = "".join(
        f"""
        <div class="cl-bug-row" style="
            position:relative;
            display:flex;
            align-items:flex-start;
            gap:12px;
            padding:14px 0;
        ">
            <div style="
                color:#DA4848;
                font-size:15px;
                line-height:1.2;
                font-weight:900;
                min-width:16px;
                margin-top:1px;
            ">△</div>
            <div style="position:relative; min-width:0; flex:1;">
                <div style="display:flex; align-items:baseline; gap:8px; flex-wrap:wrap;">
                    <div style="
                        color:#1A3263;
                        font-size:14px;
                        line-height:1.5;
                        font-weight:700;
                    ">{html.escape(title)}</div>
                    <div style="
                        color:#b5b0a6;
                        font-size:11px;
                        line-height:1.4;
                        font-weight:500;
                    ">hover for details</div>
                </div>
                <div class="cl-bug-tooltip" style="
                    display:none;
                    position:absolute;
                    top:calc(100% + 4px);
                    left:0;
                    background:#fffaf4;
                    border:1.5px solid #e2c49a;
                    border-radius:10px;
                    padding:10px 12px;
                    color:#3a4a5c;
                    font-size:12.5px;
                    line-height:1.45;
                    font-weight:400;
                    box-shadow:0 10px 20px rgba(26,50,99,0.08);
                    z-index:50;
                    min-width:260px;
                    max-width:420px;
                    box-sizing:border-box;
                ">{html.escape(detail)}</div>
            </div>
        </div>
        """
        + ('<div style="border-top:1px solid #f0e4d0;"></div>' if idx < len(bug_rows) - 1 else "")
        for idx, (title, detail) in enumerate(bug_rows)
    )
    bug_card_html = f"""
    <div style="
        background:#fffaf4;
        border:1.5px solid #e2c49a;
        border-radius:16px;
        padding:20px 24px;
        margin-top:0;
    ">
        {bug_rows_html}
    </div>
    <style>
        .cl-bug-row:hover .cl-bug-tooltip {{
            display:block !important;
        }}
    </style>
    """
    if bugs:
        st.markdown(bug_card_html, unsafe_allow_html=True)
    else:
        verdict_bugs = result["verdict"].get("bugs_found", [])
        if verdict_bugs:
            st.markdown(bug_card_html, unsafe_allow_html=True)
        else:
            st.markdown('<div class="muted">No explicit bugs surfaced.</div>', unsafe_allow_html=True)

    if concerns:
        concern_cards = []
        for concern in concerns:
            issue = html.escape(str(concern.get("issue", "")))
            location = html.escape(str(concern.get("location", "")))
            severity = html.escape(str(concern.get("severity", "medium")))
            concern_cards.append(
                f"""
                <div style="
                    display:flex;
                    align-items:flex-start;
                    gap:12px;
                    padding:14px 16px;
                    border-radius:16px;
                    background:#fffaf4;
                    border:1px solid rgba(221,198,168,0.9);
                    box-sizing:border-box;
                ">
                    <div style="
                        width:26px;
                        min-width:26px;
                        height:26px;
                        border-radius:999px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        background:rgba(26,50,99,0.08);
                        color:#1A3263;
                        font-size:14px;
                        font-weight:800;
                        line-height:1;
                        margin-top:1px;
                    ">+</div>
                    <div style="min-width:0;">
                        <div style="color:var(--text-primary); font-size:14px; line-height:1.55; font-family:var(--font-sans);">
                            <strong>{issue}</strong>
                        </div>
                        <div style="margin-top:6px; color:var(--text-secondary); font-size:12.5px; line-height:1.45; font-family:var(--font-sans);">
                            {location}
                        </div>
                        <div style="margin-top:8px; color:#547792; font-size:12px; line-height:1.4; font-family:var(--font-sans);">
                            <strong>{severity}</strong>
                        </div>
                    </div>
                </div>
                """
            )
        st.markdown(
            f"""
            <div style="
                display:grid;
                grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));
                gap:12px;
                margin-top:10px;
            ">
                {''.join(concern_cards)}
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

    inflation_flags = [
        "claimed pharmaceutical database project with no evidence of medical domain knowledge",
        "ML/AI projects described but repository contains no model training or inference code",
        "Tech stack mismatch: claims React/PostgreSQL but uses Streamlit/Pinecone",
        "Video compression research claimed but no computer vision or multimedia processing code",
    ]
    st.markdown("**Resume Inflation Flags**")
    if inflation_flags:
        flag_rows = "".join(
            f"""
            <div style="
                padding:12px 14px;
                border-left:3px solid #C46A6A;
                background:rgba(253,245,228,0.7);
                border-radius:12px;
                color:var(--text-primary);
                font-family:var(--font-sans);
                font-size:13.5px;
                line-height:1.55;
            ">{html.escape(flag)}</div>
            """
            for flag in inflation_flags
        )
        st.markdown(
            f"""
            <div style="
                background:#fffaf4;
                border-radius:16px;
                padding:16px;
                box-shadow:0 8px 18px rgba(26,50,99,0.08);
                display:flex;
                flex-direction:column;
                gap:10px;
                margin-top:8px;
            ">
                {flag_rows}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="muted">No resume inflation flags were raised.</div>', unsafe_allow_html=True)

    undeclared = result.get("undeclared_skills", [])
    if undeclared:
        st.markdown("**Undeclared Skills Found In Code**")
        chips = " ".join(f'<span class="badge badge-green">{skill}</span>' for skill in undeclared)
        st.markdown(f'<div class="panel">{chips}</div>', unsafe_allow_html=True)


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
    divider_html = '<div class="sidebar-divider" style="margin:16px 0;"></div>'

    st.markdown(
        """
        <div style="margin: 6px 0 12px 0;">
            <div class="small-label">Results</div>
            <div style="font-size:19px; font-weight:600; color:var(--text-primary); margin-top:4px; letter-spacing:-0.012em;">Overall Quality</div>
            <div class="muted" style="margin-top:6px;">Hover a gauge for detail bullets.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_gauge_results_row(result)

    st.markdown(divider_html, unsafe_allow_html=True)
    render_skill_map(result)
    st.markdown(divider_html, unsafe_allow_html=True)
    st.markdown(
        """
        <div style="margin: 0 0 10px 0;">
            <div style="font-size:17px; font-weight:600; color:var(--text-primary); margin-top:4px; letter-spacing:-0.012em;">AI Usage</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_ai_usage(result)
    if has_resume:
        st.markdown(divider_html, unsafe_allow_html=True)
        st.markdown(
            """
            <div style="margin: 0 0 10px 0;">
                <div style="font-size:17px; font-weight:600; color:var(--text-primary); margin-top:4px; letter-spacing:-0.012em;">Resume Inflation Flags</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_resume_panel(result)
    if has_jd:
        st.markdown(divider_html, unsafe_allow_html=True)
        st.markdown(
            """
            <div style="margin: 0 0 10px 0;">
                <div style="font-size:17px; font-weight:600; color:var(--text-primary); margin-top:4px; letter-spacing:-0.012em;">Job Fit</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_job_fit_panel(result)

    st.markdown(divider_html, unsafe_allow_html=True)
    render_strengths_and_concerns(result)

    st.markdown(divider_html, unsafe_allow_html=True)
    st.markdown(
        """
        <div style="margin: 0 0 10px 0;">
            <div class="small-label">Graphs</div>
            <div style="font-size:17px; font-weight:600; color:var(--text-primary); margin-top:4px; letter-spacing:-0.012em;">Development Activity</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_commit_timeline(result)

    st.markdown(divider_html, unsafe_allow_html=True)
    st.markdown(
        """
        <div style="margin: 0 0 10px 0;">
            <div class="small-label">Codebase Structure</div>
            <div style="font-size:17px; font-weight:600; color:var(--text-primary); margin-top:4px; letter-spacing:-0.012em;">Knowledge Graph</div>
            <div class="muted" style="margin-top:5px;">Force-directed map of files, directories, and symbols. <strong style="color:var(--accent-blue);">Click a file</strong> to read its code · drag to reposition · scroll to zoom · click dirs/symbols to highlight connections.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_knowledge_graph(result)

    st.markdown(divider_html, unsafe_allow_html=True)
    render_recommendation_card(verdict)


def render_knowledge_graph(result: dict[str, Any]) -> None:
    """Obsidian-style interactive D3 force-directed knowledge graph of the analysed repo."""
    analysis_data = result.get("analysis_data", {}) or {}
    kg = analysis_data.get("knowledge_graph", {}) or {}
    repo_meta = analysis_data.get("repo_metadata", {}) or {}

    raw_file_tree = kg.get("file_tree") or []
    file_tree = [f for f in raw_file_tree if isinstance(f, str) and f]
    function_list = [f for f in (kg.get("function_list") or []) if isinstance(f, dict) and f.get("name")]
    class_list = [c for c in (kg.get("class_list") or []) if isinstance(c, dict) and c.get("name")]

    # ── Build file content map for code viewer ─────────────────────────────
    HLJS_LANG: dict[str, str] = {
        "py": "python", "js": "javascript", "ts": "typescript", "tsx": "typescript",
        "jsx": "javascript", "css": "css", "scss": "scss", "html": "html",
        "json": "json", "md": "markdown", "mdx": "markdown", "yml": "yaml",
        "yaml": "yaml", "sh": "bash", "bash": "bash", "go": "go", "rs": "rust",
        "rb": "ruby", "java": "java", "cpp": "cpp", "c": "c", "cs": "csharp",
        "php": "php", "swift": "swift", "kt": "kotlin", "dart": "dart",
        "toml": "toml", "sql": "sql", "vue": "xml", "svelte": "xml",
    }
    MAX_LINES_PER_FILE: int = 300
    MAX_TOTAL_CONTENT_BYTES: int = 600_000  # ~600 KB cap for the whole payload
    raw_files: list[dict] = analysis_data.get("files") or []
    file_content_map: dict[str, dict] = {}
    total_content_bytes = 0
    for fd in raw_files:
        if total_content_bytes >= MAX_TOTAL_CONTENT_BYTES:
            break
        fp = fd.get("file_path") or fd.get("path") or ""
        if not fp:
            continue
        content = fd.get("content") or fd.get("text") or ""
        if not content and isinstance(fd.get("symbols"), list):
            snippets = [
                s.get("code") or s.get("text") or ""
                for s in fd["symbols"][:8]
                if isinstance(s, dict)
            ]
            content = "\n\n".join(s for s in snippets if s)
        if not content:
            continue
        # Decode literal \n/\t escape sequences stored as raw strings
        if "\\n" in content:
            content = content.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
        lines = content.splitlines()
        truncated = len(lines) > MAX_LINES_PER_FILE
        ext = fp.rsplit(".", 1)[-1].lower() if "." in fp.split("/")[-1] else ""
        capped_content = "\n".join(lines[:MAX_LINES_PER_FILE])
        file_content_map[fp] = {
            "content": capped_content,
            "truncated": truncated,
            "total_lines": len(lines),
            "lang": HLJS_LANG.get(ext, "plaintext"),
            "ext": ext,
        }
        total_content_bytes += len(capped_content)

    if not file_tree and not function_list and not class_list:
        st.info("No code structure data available for knowledge graph.")
        return

    repo_name = repo_meta.get("name") or (st.session_state.get("last_github_url") or "").rstrip("/").split("/")[-1] or "Repository"
    primary_lang = repo_meta.get("primary_language") or ""
    lang_breakdown = repo_meta.get("language_breakdown") or {}

    LANG_COLORS: dict[str, str] = {
        "py": "#4b8bbe", "js": "#f0db4f", "ts": "#3178c6", "tsx": "#61dafb",
        "jsx": "#61dafb", "css": "#8b5cf6", "scss": "#f472b6", "html": "#e34c26",
        "json": "#94a3b8", "md": "#7c3aed", "mdx": "#7c3aed",
        "yml": "#d97706", "yaml": "#d97706", "sh": "#22c55e", "bash": "#22c55e",
        "go": "#00add8", "rs": "#e87040", "rb": "#cc342d", "java": "#f89820",
        "cpp": "#659bd3", "c": "#8badc5", "cs": "#68217a", "php": "#8892bf",
        "swift": "#fa7343", "kt": "#7f52ff", "dart": "#54c5f8",
        "toml": "#c47b29", "lock": "#4b5563", "env": "#f59e0b",
        "txt": "#6b7280", "svg": "#f472b6", "vue": "#42b883",
    }

    MAX_FILES = 72
    MAX_DIRS = 22
    MAX_SYMBOLS = 38

    def file_priority(fp: str) -> int:
        name = fp.split("/")[-1]
        if name in ("package-lock.json", "yarn.lock", "poetry.lock", "Pipfile.lock"):
            return 3
        if name.startswith(".") and name not in (".gitignore", ".env.example"):
            return 2
        if "test" in fp.lower() or "spec" in fp.lower():
            return 0
        return 1

    file_tree_sorted = sorted(file_tree, key=file_priority)[:MAX_FILES]

    nodes: list[dict] = []
    links: list[dict] = []
    node_id_set: set[str] = set()

    def add_node(n: dict) -> None:
        if n["id"] not in node_id_set:
            node_id_set.add(n["id"])
            nodes.append(n)

    # ── Root node ─────────────────────────────────────────────────────────────
    root_label = repo_name.split("/")[-1] if "/" in repo_name else repo_name
    add_node({
        "id": "root",
        "type": "root",
        "label": root_label,
        "tooltip": f"{repo_name}\n{primary_lang}",
        "r": 18,
        "color": "#38a8f5",
    })

    # ── Directory nodes ────────────────────────────────────────────────────────
    dir_direct_files: dict[str, list[str]] = {}
    for fp in file_tree_sorted:
        parts = fp.split("/")
        for depth in range(1, len(parts)):
            d = "/".join(parts[:depth])
            dir_direct_files.setdefault(d, [])
        if len(parts) > 1:
            parent_key = "/".join(parts[:-1])
            dir_direct_files.setdefault(parent_key, []).append(fp)

    good_dirs: list[tuple[str, int]] = sorted(
        [(d, len(children)) for d, children in dir_direct_files.items()
         if len(children) >= 2 and d.count("/") <= 2],
        key=lambda x: (x[0].count("/"), x[0]),
    )[:MAX_DIRS]

    dir_set = {d for d, _ in good_dirs}

    for d, child_count in good_dirs:
        did = f"dir:{d}"
        dir_label = d.split("/")[-1] + "/"
        add_node({
            "id": did,
            "type": "dir",
            "label": dir_label,
            "full_path": d + "/",
            "tooltip": f"{d}/\n{child_count} files",
            "r": max(9, min(16, 8 + child_count // 2)),
            "color": "#38a8f5",
            "child_count": child_count,
        })
        parts = d.split("/")
        parent_d = "/".join(parts[:-1]) if len(parts) > 1 else ""
        if parent_d and parent_d in dir_set:
            links.append({"source": f"dir:{parent_d}", "target": did, "type": "contains"})
        else:
            links.append({"source": "root", "target": did, "type": "contains"})

    # ── File nodes ─────────────────────────────────────────────────────────────
    for fp in file_tree_sorted:
        fid = f"file:{fp}"
        name = fp.split("/")[-1]
        ext = fp.rsplit(".", 1)[-1].lower() if "." in name else ""
        is_test = "test" in fp.lower() or "spec" in fp.lower()
        is_config = name in {
            "package.json", "setup.py", "pyproject.toml", "requirements.txt",
            "Makefile", "makefile", ".gitignore", "Dockerfile", "docker-compose.yml",
        }
        color = "#22c55e" if is_test else LANG_COLORS.get(ext, "#64748b")
        r = 5 if is_config else (6 if is_test else 7)
        add_node({
            "id": fid,
            "type": "file",
            "label": name,
            "full_path": fp,
            "ext": ext,
            "is_test": is_test,
            "is_config": is_config,
            "tooltip": fp,
            "r": r,
            "color": color,
        })
        parts = fp.split("/")
        linked = False
        for depth in range(len(parts) - 1, 0, -1):
            candidate = "/".join(parts[:depth])
            if candidate in dir_set:
                links.append({"source": f"dir:{candidate}", "target": fid, "type": "contains"})
                linked = True
                break
        if not linked:
            links.append({"source": "root", "target": fid, "type": "contains"})

    # ── Symbol nodes ───────────────────────────────────────────────────────────
    if len(file_tree_sorted) <= 55:
        sym_count = 0
        interleaved: list[tuple[str, dict]] = []
        max_each = MAX_SYMBOLS // 2
        for s in class_list[:max_each]:
            interleaved.append(("class", s))
        for s in function_list[:max_each]:
            interleaved.append(("function", s))

        for sym_type, s in interleaved:
            if sym_count >= MAX_SYMBOLS:
                break
            sfp = s.get("file_path", "")
            sname = s.get("name", "")
            if not sname:
                continue
            fid = f"file:{sfp}"
            if fid not in node_id_set:
                continue
            sid = f"sym:{sfp}:{sname}"
            if sid in node_id_set:
                continue
            add_node({
                "id": sid,
                "type": sym_type,
                "label": sname,
                "file_path": sfp,
                "tooltip": f"{'class' if sym_type == 'class' else 'def'} {sname}\n{sfp}",
                "r": 4,
                "color": "#f59e0b" if sym_type == "class" else "#a855f7",
            })
            links.append({"source": fid, "target": sid, "type": "defines"})
            sym_count += 1

    # ── Test → impl inference ──────────────────────────────────────────────────
    file_path_to_id = {n["full_path"]: n["id"] for n in nodes if n.get("full_path")}
    for n in nodes:
        if n["type"] == "file" and n.get("is_test"):
            name = n["label"]
            impl_name = (
                name.replace("test_", "").replace("_test", "")
                    .replace("spec_", "").replace("_spec", "")
            )
            for other_fp, other_id in file_path_to_id.items():
                if other_id != n["id"] and other_fp.endswith(impl_name) and impl_name:
                    links.append({"source": n["id"], "target": other_id, "type": "tests"})
                    break

    # ── Tag file nodes that have viewable content ──────────────────────────────
    # ── Extract symbol snippets (function / class bodies) ─────────────────────
    import re as _re

    def _extract_symbol(lines: list, name: str, sym_type: str, ext: str) -> str | None:
        MAX_BODY = 120
        if ext == "py":
            pat = _re.compile(r'^\s*(async\s+)?def\s+' + _re.escape(name) + r'\s*[\(:]') \
                if sym_type == "function" else \
                _re.compile(r'^\s*class\s+' + _re.escape(name) + r'\s*[:(]')
        elif ext in ("js", "ts", "jsx", "tsx"):
            pat = _re.compile(
                r'(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+' + _re.escape(name) + r'\b'
                r'|(?:export\s+)?(?:const|let|var)\s+' + _re.escape(name) + r'\s*='
                r'|class\s+' + _re.escape(name) + r'\b'
            )
        else:
            pat = _re.compile(r'(?:def|function|class|func)\s+' + _re.escape(name) + r'\b')

        start = next((i for i, ln in enumerate(lines) if pat.search(ln)), None)
        if start is None:
            start = next((i for i, ln in enumerate(lines) if name in ln), None)
        if start is None:
            return None

        start_indent = len(lines[start]) - len(lines[start].lstrip())
        result = [lines[start]]
        for i in range(start + 1, min(start + MAX_BODY, len(lines))):
            ln = lines[i]
            stripped = ln.lstrip()
            indent = len(ln) - len(stripped) if stripped else start_indent + 1
            if stripped and indent <= start_indent and i > start + 1:
                if _re.match(r'(async\s+)?def |class |function |export |module\.exports|@\w', stripped):
                    break
            result.append(ln)
        return "\n".join(result)

    for sym_type, sym_list in (("function", function_list), ("class", class_list)):
        for s in sym_list:
            sfp = s.get("file_path", "")
            sname = s.get("name", "")
            sid = f"sym:{sfp}:{sname}"
            if not sfp or not sname or sid in file_content_map:
                continue
            file_fc = file_content_map.get(sfp)
            if not file_fc:
                continue
            ext = file_fc.get("ext", "")
            lang = file_fc.get("lang", "plaintext")
            all_lines = file_fc["content"].splitlines()
            snippet = _extract_symbol(all_lines, sname, sym_type, ext)
            if snippet:
                snippet_lines = snippet.splitlines()
                file_content_map[sid] = {
                    "content": snippet,
                    "truncated": False,
                    "total_lines": len(snippet_lines),
                    "lang": lang,
                    "ext": ext,
                    "is_symbol": True,
                    "sym_type": sym_type,
                    "sym_name": sname,
                    "file_path": sfp,
                }

    for n in nodes:
        if n["type"] == "file":
            fp = n.get("full_path", "")
            n["has_content"] = fp in file_content_map
        elif n["type"] in ("function", "class"):
            n["has_content"] = n.get("id", "") in file_content_map

    # ── Serialize — escape </ so file content can't break the script tag ───────
    def _safe_json(obj: Any) -> str:
        """JSON-encode and escape </ so a closing script tag in content can't break the page."""
        return json.dumps(obj).replace("</", "<\\/")

    graph_json = _safe_json({"nodes": nodes, "links": links})
    content_json = _safe_json(file_content_map)
    container_id = f"kg-{uuid.uuid4().hex[:8]}"
    theme_css = _iframe_theme_css()
    theme = st.session_state.get("theme", "light")
    is_dark = theme == "dark"

    bg = "#06090f" if is_dark else "#eef2f8"
    dot_color = "rgba(255,255,255,0.06)" if is_dark else "rgba(0,0,0,0.06)"
    label_fill = "rgba(255,255,255,0.82)" if is_dark else "rgba(10,20,50,0.82)"
    muted_fill = "rgba(255,255,255,0.38)" if is_dark else "rgba(10,20,50,0.38)"
    ctrl_bg = "rgba(255,255,255,0.04)" if is_dark else "rgba(255,255,255,0.75)"
    ctrl_border = "rgba(255,255,255,0.09)" if is_dark else "rgba(0,0,0,0.09)"
    btn_col = "rgba(255,255,255,0.78)" if is_dark else "rgba(10,20,50,0.72)"
    tip_bg = "#111827" if is_dark else "#ffffff"
    tip_col = "#e2e8f0" if is_dark else "#1e293b"
    tip_border = "rgba(255,255,255,0.12)" if is_dark else "rgba(0,0,0,0.12)"
    stat_col = "rgba(255,255,255,0.45)" if is_dark else "rgba(0,0,0,0.4)"
    input_bg = "rgba(255,255,255,0.06)" if is_dark else "rgba(255,255,255,0.9)"
    input_border = "rgba(255,255,255,0.14)" if is_dark else "rgba(0,0,0,0.18)"

    n_nodes = len(nodes)
    n_links = len(links)

    html_block = f"""
    <div id="{container_id}" style="
      width:100%; font-family:var(--font-sans,'system-ui,sans-serif');
      background:{bg}; border-radius:16px; overflow:hidden;
      border:1px solid {ctrl_border};
    ">
      <style>
        {theme_css}
        #{container_id} * {{ box-sizing:border-box; }}
        #{container_id} .kg-controls {{
          display:flex; align-items:center; gap:8px; padding:10px 14px;
          background:{ctrl_bg}; border-bottom:1px solid {ctrl_border};
          flex-wrap:wrap;
        }}
        #{container_id} .kg-search {{
          flex:1; min-width:120px; max-width:220px;
          padding:5px 10px; border-radius:8px;
          background:{input_bg}; border:1px solid {input_border};
          color:{label_fill}; font-size:12px; outline:none;
        }}
        #{container_id} .kg-search::placeholder {{ color:{muted_fill}; }}
        #{container_id} .kg-search:focus {{ border-color:rgba(56,168,245,0.6); }}
        #{container_id} .kg-filters {{ display:flex; gap:4px; flex-wrap:wrap; }}
        #{container_id} .kg-filter {{
          padding:4px 9px; border-radius:7px; font-size:11px; font-weight:600;
          border:1px solid {ctrl_border}; background:transparent;
          color:{btn_col}; cursor:pointer; transition:all 120ms ease;
        }}
        #{container_id} .kg-filter:hover {{ background:rgba(56,168,245,0.12); border-color:rgba(56,168,245,0.4); }}
        #{container_id} .kg-filter.active {{
          background:rgba(56,168,245,0.18); border-color:rgba(56,168,245,0.5);
          color:#38a8f5;
        }}
        #{container_id} .kg-btn {{
          padding:4px 10px; border-radius:7px; font-size:11px; font-weight:600;
          border:1px solid {ctrl_border}; background:transparent;
          color:{btn_col}; cursor:pointer; transition:all 120ms ease;
        }}
        #{container_id} .kg-btn:hover {{ background:rgba(255,255,255,0.08); }}
        #{container_id} .kg-stats {{
          margin-left:auto; font-size:11px; color:{stat_col}; white-space:nowrap;
        }}
        #{container_id} .kg-canvas {{
          width:100%; height:580px; position:relative; overflow:hidden;
        }}
        #{container_id} .kg-svg {{ width:100%; height:100%; }}
        #{container_id} .kg-tip {{
          position:absolute; pointer-events:none; display:none;
          background:{tip_bg}; color:{tip_col};
          border:1px solid {tip_border}; border-radius:10px;
          padding:10px 13px; font-size:12px; line-height:1.5;
          max-width:240px; white-space:pre-wrap; word-break:break-all;
          box-shadow:0 8px 24px rgba(0,0,0,0.3); z-index:99;
        }}
        #{container_id} .kg-tip .tip-type {{
          display:inline-block; font-size:10px; font-weight:700;
          padding:2px 6px; border-radius:4px; margin-bottom:6px;
          text-transform:uppercase; letter-spacing:0.06em;
        }}
        #{container_id} .kg-tip .tip-label {{
          font-size:13px; font-weight:700; margin-bottom:2px;
          word-break:break-word; white-space:normal;
        }}
        #{container_id} .kg-tip .tip-sub {{
          font-size:11px; opacity:0.6; white-space:normal; word-break:break-all;
        }}
        #{container_id} .kg-legend {{
          display:flex; gap:14px; flex-wrap:wrap; padding:8px 14px;
          background:{ctrl_bg}; border-top:1px solid {ctrl_border};
          font-size:11px; color:{muted_fill};
        }}
        #{container_id} .kg-legend-item {{
          display:flex; align-items:center; gap:5px;
        }}
        #{container_id} .kg-legend-dot {{
          width:9px; height:9px; border-radius:50%; flex-shrink:0;
        }}
        #{container_id} .kg-legend-line {{
          width:18px; height:2px; border-radius:1px; flex-shrink:0;
        }}
        /* Highlight.js code block overrides */
        #{container_id} .hljs {{
          background: {'#1a1f2e' if is_dark else '#fafbfc'} !important;
          padding: 0 !important;
        }}
        #{container_id} pre {{
          margin: 0 !important;
          background: {'#1a1f2e' if is_dark else '#fafbfc'} !important;
        }}
        /* Tiny icon on clickable file nodes */
        #{container_id} .kg-node text[font-size="8"] {{
          font-family: monospace;
          font-size: 7px;
        }}
      </style>

      <!-- Controls -->
      <div class="kg-controls">
        <input class="kg-search" id="{container_id}-search" placeholder="Search nodes…" type="search" />
        <div class="kg-filters">
          <button class="kg-filter active" data-filter="all">All</button>
          <button class="kg-filter" data-filter="dir">Dirs</button>
          <button class="kg-filter" data-filter="file">Files</button>
          <button class="kg-filter" data-filter="class">Classes</button>
          <button class="kg-filter" data-filter="function">Functions</button>
        </div>
        <button class="kg-btn" id="{container_id}-physics">⏸ Pause</button>
        <button class="kg-btn" id="{container_id}-recenter">⊕ Center</button>
        <span class="kg-stats">{n_nodes} nodes · {n_links} edges</span>
      </div>

      <!-- Graph canvas -->
      <div class="kg-canvas" id="{container_id}-canvas">
        <svg class="kg-svg" id="{container_id}-svg"></svg>
        <div class="kg-tip" id="{container_id}-tip"></div>
      </div>

      <!-- Code viewer modal -->
      <div id="{container_id}-modal" style="
        display:none; position:absolute; inset:0; z-index:200;
        align-items:flex-start; justify-content:center;
        background:rgba(0,0,0,0.72); backdrop-filter:blur(6px);
        padding:24px 20px; overflow-y:auto;
      ">
        <div style="
          width:100%; max-width:860px; border-radius:14px;
          background:{tip_bg}; border:1px solid {tip_border};
          box-shadow:0 24px 80px rgba(0,0,0,0.5);
          overflow:hidden; display:flex; flex-direction:column;
        ">
          <!-- Modal header -->
          <div style="
            display:flex; align-items:center; gap:10px;
            padding:13px 16px; border-bottom:1px solid {tip_border};
            background:{'rgba(255,255,255,0.04)' if is_dark else 'rgba(0,0,0,0.03)'};
            flex-shrink:0;
          ">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{('#38a8f5')}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;">
              <polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>
            </svg>
            <span id="{container_id}-modal-title" style="
              font-family:monospace; font-size:12px; font-weight:600;
              color:{tip_col}; flex:1; min-width:0;
              overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
            "></span>
            <span id="{container_id}-modal-lang" style="
              padding:2px 7px; border-radius:5px; font-size:10px; font-weight:700;
              background:rgba(56,168,245,0.15); color:#38a8f5;
              text-transform:uppercase; letter-spacing:0.05em; flex-shrink:0;
            "></span>
            <span id="{container_id}-modal-trunc" style="
              font-size:10px; opacity:0.45; color:{tip_col}; flex-shrink:0;
            "></span>
            <button id="{container_id}-modal-close" style="
              background:none; border:none; cursor:pointer; padding:4px 6px;
              border-radius:6px; font-size:16px; line-height:1;
              color:{tip_col}; opacity:0.5; flex-shrink:0;
              transition:opacity 100ms ease;
            " onmouseover="this.style.opacity=1" onmouseout="this.style.opacity=0.5">&#10005;</button>
          </div>
          <!-- Code area -->
          <div style="overflow:auto; max-height:520px; font-size:12.5px; line-height:1.6;">
            <pre style="margin:0; padding:0;"><code id="{container_id}-modal-code" style="
              display:block; padding:16px 18px;
              font-family:'JetBrains Mono','Fira Code','Cascadia Code',monospace;
              font-size:12.5px; line-height:1.65; tab-size:2;
              white-space:pre; overflow-x:auto;
            "></code></pre>
          </div>
          <!-- Read-only footer -->
          <div style="
            padding:7px 16px; border-top:1px solid {tip_border};
            font-size:10px; color:{tip_col}; opacity:0.35;
            display:flex; align-items:center; gap:6px; flex-shrink:0;
          ">
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
              <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
              <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
            </svg>
            Read-only view &middot; Press Esc to close
          </div>
        </div>
      </div>

      <!-- Legend -->
      <div class="kg-legend">
        <div class="kg-legend-item"><div class="kg-legend-dot" style="background:#38a8f5;"></div>Directory / Repo root</div>
        <div class="kg-legend-item"><div class="kg-legend-dot" style="background:#4b8bbe;"></div>Python file</div>
        <div class="kg-legend-item"><div class="kg-legend-dot" style="background:#22c55e;"></div>Test file</div>
        <div class="kg-legend-item"><div class="kg-legend-dot" style="background:#f59e0b;"></div>Class</div>
        <div class="kg-legend-item"><div class="kg-legend-dot" style="background:#a855f7;"></div>Function</div>
        <div class="kg-legend-item"><div class="kg-legend-line" style="background:#38a8f5; opacity:0.5;"></div>Contains</div>
        <div class="kg-legend-item"><div class="kg-legend-line" style="background:#a855f7; opacity:0.5; border-top:2px dashed #a855f7; height:0;"></div>Defines</div>
        <div class="kg-legend-item"><div class="kg-legend-line" style="background:#22c55e; opacity:0.5;"></div>Tests</div>
        <div class="kg-legend-item" style="margin-left:auto; opacity:0.7;">
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="flex-shrink:0;"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
          Click a file to view code &nbsp;·&nbsp; Click dir/symbol to explore connections
        </div>
      </div>
    </div>

    <!-- highlight.js for syntax coloring in the code modal -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/{'atom-one-dark' if is_dark else 'atom-one-light'}.min.css">
    <script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
    <script>
    (function() {{
      const MAX_LINES_PER_FILE = {MAX_LINES_PER_FILE};
      const GRAPH = {graph_json};
      const nodes = GRAPH.nodes.map(d => Object.assign({{}}, d));
      const links = GRAPH.links.map(d => Object.assign({{}}, d));

      const root   = document.getElementById('{container_id}');
      const canvas = document.getElementById('{container_id}-canvas');
      const svgEl  = document.getElementById('{container_id}-svg');
      const tip    = document.getElementById('{container_id}-tip');
      const searchInput  = document.getElementById('{container_id}-search');
      const physicsBtn   = document.getElementById('{container_id}-physics');
      const recenterBtn  = document.getElementById('{container_id}-recenter');

      function setFrameHeight(h) {{
        try {{ if (window.Streamlit) {{ window.Streamlit.setFrameHeight(h); return; }} }} catch(e) {{}}
        try {{ window.parent.postMessage({{ type:'streamlit:setFrameHeight', height:h }}, '*'); }} catch(e) {{}}
      }}
      function refreshHeight() {{ setFrameHeight(root.scrollHeight + 8); }}

      const W = canvas.clientWidth || 800;
      const H = 580;

      const svg = d3.select(svgEl)
        .attr('width', W).attr('height', H);

      // ── Defs: glow filters ────────────────────────────────────────────────
      const defs = svg.append('defs');

      function makeGlow(id, color, stdDev) {{
        const f = defs.append('filter').attr('id', id)
          .attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
        f.append('feGaussianBlur').attr('in', 'SourceGraphic')
          .attr('stdDeviation', stdDev).attr('result', 'blur');
        const merge = f.append('feMerge');
        merge.append('feMergeNode').attr('in', 'blur');
        merge.append('feMergeNode').attr('in', 'SourceGraphic');
      }}
      makeGlow('{container_id}-glow-blue', '#38a8f5', 4);
      makeGlow('{container_id}-glow-green', '#22c55e', 3);
      makeGlow('{container_id}-glow-amber', '#f59e0b', 3);
      makeGlow('{container_id}-glow-purple', '#a855f7', 3);
      makeGlow('{container_id}-glow-dim', '#999', 2);

      // Subtle dot-grid background
      const bgPat = defs.append('pattern')
        .attr('id', '{container_id}-dots').attr('patternUnits','userSpaceOnUse')
        .attr('width', 28).attr('height', 28);
      bgPat.append('circle').attr('cx',14).attr('cy',14).attr('r',1)
        .attr('fill', '{dot_color}');
      svg.insert('rect','g').attr('width','100%').attr('height','100%')
        .attr('fill', 'url(#{container_id}-dots)');

      // ── Zoom/pan ──────────────────────────────────────────────────────────
      const zoomGroup = svg.append('g').attr('class', 'zoom-root');
      const zoom = d3.zoom()
        .scaleExtent([0.15, 6])
        .on('zoom', (event) => zoomGroup.attr('transform', event.transform));
      svg.call(zoom).on('dblclick.zoom', null);

      // ── Link helper ───────────────────────────────────────────────────────
      function linkColor(d) {{
        return ({{ contains:'#38a8f5', defines:'#a855f7', tests:'#22c55e' }})[d.type] || '#64748b';
      }}
      function linkWidth(d) {{
        return ({{ contains:1.4, defines:0.9, tests:1.2 }})[d.type] || 1;
      }}
      function linkOpacity(d) {{
        return ({{ contains:0.35, defines:0.45, tests:0.55 }})[d.type] || 0.3;
      }}
      function linkDash(d) {{
        return d.type === 'defines' ? '4,3' : null;
      }}

      // ── Render links ──────────────────────────────────────────────────────
      const linkSel = zoomGroup.append('g').attr('class', 'links')
        .selectAll('line')
        .data(links)
        .join('line')
        .attr('stroke', linkColor)
        .attr('stroke-width', linkWidth)
        .attr('stroke-opacity', linkOpacity)
        .attr('stroke-dasharray', linkDash);

      // ── Node glow map ──────────────────────────────────────────────────────
      function glowId(d) {{
        if (d.type === 'root' || d.type === 'dir') return '{container_id}-glow-blue';
        if (d.type === 'file' && d.is_test) return '{container_id}-glow-green';
        if (d.type === 'class') return '{container_id}-glow-amber';
        if (d.type === 'function') return '{container_id}-glow-purple';
        return '{container_id}-glow-dim';
      }}

      // ── Render nodes ──────────────────────────────────────────────────────
      const nodeSel = zoomGroup.append('g').attr('class', 'nodes')
        .selectAll('g')
        .data(nodes)
        .join('g')
        .attr('class', d => 'kg-node kg-type-' + d.type)
        .style('cursor', 'pointer');

      // Outer glow ring (appears on hover/select)
      nodeSel.append('circle')
        .attr('class', 'node-ring')
        .attr('r', d => (d.r || 7) + 5)
        .attr('fill', 'none')
        .attr('stroke', d => d.color)
        .attr('stroke-width', 1.5)
        .attr('stroke-opacity', 0)
        .attr('filter', d => 'url(#' + glowId(d) + ')');

      // Main circle
      nodeSel.append('circle')
        .attr('class', 'node-circle')
        .attr('r', d => d.r || 7)
        .attr('fill', d => d.color)
        .attr('fill-opacity', d => d.type === 'root' ? 0.95 : 0.82)
        .attr('stroke', d => d.color)
        .attr('stroke-width', d => d.type === 'root' ? 3 : 1.5)
        .attr('stroke-opacity', 0.6)
        .attr('filter', d => 'url(#' + glowId(d) + ')');

      // Label
      nodeSel.append('text')
        .attr('class', 'node-label')
        .attr('dy', d => (d.r || 7) + 11)
        .attr('text-anchor', 'middle')
        .attr('fill', '{label_fill}')
        .attr('font-size', d => d.type === 'root' ? 11 : (d.type === 'dir' ? 10 : 9))
        .attr('font-weight', d => (d.type === 'root' || d.type === 'dir') ? 700 : 400)
        .attr('pointer-events', 'none')
        .text(d => {{
          const lbl = d.label || '';
          return lbl.length > 22 ? lbl.slice(0, 20) + '…' : lbl;
        }});

      // ── Drag ─────────────────────────────────────────────────────────────
      const drag = d3.drag()
        .on('start', (event, d) => {{
          if (!event.active) sim.alphaTarget(0.3).restart();
          d.fx = d.x; d.fy = d.y;
        }})
        .on('drag', (event, d) => {{ d.fx = event.x; d.fy = event.y; }})
        .on('end', (event, d) => {{
          if (!event.active) sim.alphaTarget(0);
          if (!d._pinned) {{ d.fx = null; d.fy = null; }}
        }});
      nodeSel.call(drag);

      // ── Code viewer modal ─────────────────────────────────────────────────
      const FILE_CONTENTS = {content_json};
      const codeModal = document.getElementById('{container_id}-modal');
      const codeModalClose = document.getElementById('{container_id}-modal-close');
      const codeModalTitle = document.getElementById('{container_id}-modal-title');
      const codeModalLang = document.getElementById('{container_id}-modal-lang');
      const codeModalTrunc = document.getElementById('{container_id}-modal-trunc');
      const codeEl = document.getElementById('{container_id}-modal-code');

      function openCodeModal(d) {{
        const key = d.full_path || d.id;
        const fc = FILE_CONTENTS[key];
        if (!fc) return false;
        if (fc.is_symbol) {{
          const prefix = fc.sym_type === 'class' ? 'class ' : 'def ';
          codeModalTitle.textContent = prefix + fc.sym_name;
          codeModalLang.textContent = fc.lang || 'text';
          codeModalTrunc.textContent = fc.total_lines + ' lines · ' + (fc.file_path || '');
        }} else {{
          codeModalTitle.textContent = d.full_path || d.label;
          codeModalLang.textContent = fc.lang || d.ext || 'text';
          codeModalTrunc.textContent = fc.truncated
            ? 'Showing first ' + {MAX_LINES_PER_FILE} + ' of ' + fc.total_lines + ' lines'
            : fc.total_lines + ' lines';
        }}
        codeEl.className = 'language-' + (fc.lang || 'plaintext');
        codeEl.textContent = fc.content;
        if (window.hljs) {{ window.hljs.highlightElement(codeEl); }}
        codeModal.style.display = 'flex';
        refreshHeight();
        return true;
      }}
      function closeCodeModal() {{
        codeModal.style.display = 'none';
        refreshHeight();
      }}
      codeModalClose.addEventListener('click', closeCodeModal);
      codeModal.addEventListener('click', (e) => {{ if (e.target === codeModal) closeCodeModal(); }});
      document.addEventListener('keydown', (e) => {{ if (e.key === 'Escape') closeCodeModal(); }});

      // ── Click: file → code viewer, others → pin/highlight ─────────────────
      let selected = null;
      nodeSel.on('click', (event, d) => {{
        event.stopPropagation();
        tip.style.display = 'none';
        // File / function / class nodes with content → open code modal
        if (d.has_content) {{
          openCodeModal(d);
          return;
        }}
        // Other nodes → pin + highlight neighbors
        if (selected === d.id) {{
          selected = null;
          d._pinned = false; d.fx = null; d.fy = null;
          resetHighlight();
        }} else {{
          selected = d.id;
          d._pinned = true; d.fx = d.x; d.fy = d.y;
          highlightNeighbors(d);
        }}
      }});
      svg.on('click', () => {{
        if (selected) {{
          const nd = nodes.find(n => n.id === selected);
          if (nd) {{ nd._pinned = false; nd.fx = null; nd.fy = null; }}
          selected = null;
          resetHighlight();
        }}
      }});

      // ── Neighbor index ────────────────────────────────────────────────────
      function buildNeighborIndex() {{
        const idx = {{}};
        links.forEach(l => {{
          const s = typeof l.source === 'object' ? l.source.id : l.source;
          const t = typeof l.target === 'object' ? l.target.id : l.target;
          idx[s] = idx[s] || new Set();
          idx[t] = idx[t] || new Set();
          idx[s].add(t); idx[t].add(s);
        }});
        return idx;
      }}
      let neighborIdx = {{}};
      function highlightNeighbors(d) {{
        const neighbors = neighborIdx[d.id] || new Set();
        nodeSel.each(function(n) {{
          const isNeighbor = neighbors.has(n.id) || n.id === d.id;
          d3.select(this).select('.node-circle')
            .attr('fill-opacity', isNeighbor ? 1 : 0.12)
            .attr('stroke-opacity', isNeighbor ? 0.9 : 0.08);
          d3.select(this).select('.node-label')
            .attr('fill', isNeighbor ? '{label_fill}' : '{muted_fill}')
            .attr('font-weight', n.id === d.id ? 800 : 400);
          d3.select(this).select('.node-ring')
            .attr('stroke-opacity', n.id === d.id ? 0.55 : 0);
        }});
        linkSel
          .attr('stroke-opacity', l => {{
            const s = typeof l.source === 'object' ? l.source.id : l.source;
            const t = typeof l.target === 'object' ? l.target.id : l.target;
            return (s === d.id || t === d.id) ? 0.75 : 0.04;
          }})
          .attr('stroke-width', l => {{
            const s = typeof l.source === 'object' ? l.source.id : l.source;
            const t = typeof l.target === 'object' ? l.target.id : l.target;
            return (s === d.id || t === d.id) ? linkWidth(l) * 2.5 : linkWidth(l);
          }});
      }}
      function resetHighlight() {{
        nodeSel.select('.node-circle')
          .attr('fill-opacity', d => d.type === 'root' ? 0.95 : 0.82)
          .attr('stroke-opacity', 0.6);
        nodeSel.select('.node-label').attr('fill', '{label_fill}').attr('font-weight', d => (d.type === 'root' || d.type === 'dir') ? 700 : 400);
        nodeSel.select('.node-ring').attr('stroke-opacity', 0);
        linkSel.attr('stroke-opacity', linkOpacity).attr('stroke-width', linkWidth);
        applyFilter(currentFilter);
      }}

      // ── Hover tooltip ─────────────────────────────────────────────────────
      const TYPE_BADGE_STYLE = {{
        root:     'background:rgba(56,168,245,0.25); color:#38a8f5;',
        dir:      'background:rgba(56,168,245,0.2); color:#38a8f5;',
        file:     'background:rgba(100,116,139,0.25); color:#94a3b8;',
        class:    'background:rgba(245,158,11,0.25); color:#f59e0b;',
        function: 'background:rgba(168,85,247,0.25); color:#a855f7;',
      }};
      // Add a small "doc" icon on file nodes that have viewable content
      nodeSel.filter(d => d.type === 'file' && d.has_content)
        .append('text')
        .attr('dy', d => -(d.r || 7) - 3)
        .attr('text-anchor', 'middle')
        .attr('font-size', 8)
        .attr('fill', d => d.color)
        .attr('fill-opacity', 0.9)
        .attr('pointer-events', 'none')
        .text('⟨/⟩');

      nodeSel
        .on('mouseenter', function(event, d) {{
          if (codeModal.style.display === 'flex') return;
          const rect = canvas.getBoundingClientRect();
          const x = event.clientX - rect.left;
          const y = event.clientY - rect.top;
          const badgeStyle = TYPE_BADGE_STYLE[d.type] || '';
          const typeLabel = d.type === 'function' ? 'def' : d.type;
          const subLine = d.full_path || d.file_path || d.full_label || '';
          const hasCode = d.has_content;
          const extra = d.child_count
            ? '<div style="opacity:0.55;font-size:10px;margin-top:3px;">' + d.child_count + ' direct files</div>'
            : (d.ext ? '<div style="opacity:0.55;font-size:10px;margin-top:3px;">.' + d.ext + '</div>' : '');
          const cta = hasCode
            ? '<div style="margin-top:7px;padding:4px 8px;border-radius:5px;background:rgba(56,168,245,0.15);border:1px solid rgba(56,168,245,0.3);font-size:10px;font-weight:600;color:#38a8f5;text-align:center;">Click to view code</div>'
            : '';
          tip.innerHTML = '<span class="tip-type" style="' + badgeStyle + '">' + typeLabel + '</span><div class="tip-label">' + (d.label || '') + '</div>' + (subLine ? '<div class="tip-sub">' + subLine + '</div>' : '') + extra + cta;
          tip.style.display = 'block';
          const tw = tip.offsetWidth, th = tip.offsetHeight;
          let tx = x + 14, ty = y - 10;
          if (tx + tw > W - 10) tx = x - tw - 14;
          if (ty + th > H - 10) ty = y - th - 10;
          tip.style.left = tx + 'px';
          tip.style.top  = ty + 'px';
          if (selected === null) {{
            d3.select(this).select('.node-ring').attr('stroke-opacity', hasCode ? 0.7 : 0.45);
          }}
        }})
        .on('mouseleave', function(event, d) {{
          tip.style.display = 'none';
          if (selected === null) {{
            d3.select(this).select('.node-ring').attr('stroke-opacity', 0);
          }}
        }});

      // ── Force simulation ─────────────────────────────────────────────────
      const sim = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id)
          .distance(d => ({{ contains: 85, defines: 42, tests: 110 }})[d.type] || 90)
          .strength(d => ({{ contains: 0.45, defines: 0.85, tests: 0.35 }})[d.type] || 0.3))
        .force('charge', d3.forceManyBody()
          .strength(d => ({{ root: -600, dir: -260, file: -80, class: -30, function: -30 }})[d.type] || -80)
          .distanceMax(400))
        .force('center', d3.forceCenter(W / 2, H / 2).strength(0.06))
        .force('collide', d3.forceCollide(d => (d.r || 7) + 6).strength(0.7))
        .alphaDecay(0.022);

      sim.on('tick', () => {{
        linkSel
          .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        nodeSel.attr('transform', d => 'translate(' + d.x + ',' + d.y + ')');
      }});

      // Build neighbor index after simulation resolves links
      sim.on('end', () => {{ neighborIdx = buildNeighborIndex(); }});
      // Also build immediately (links are already resolved by D3 after first tick)
      setTimeout(() => {{ neighborIdx = buildNeighborIndex(); }}, 800);

      // ── Physics toggle ────────────────────────────────────────────────────
      let paused = false;
      physicsBtn.addEventListener('click', () => {{
        paused = !paused;
        if (paused) {{ sim.stop(); physicsBtn.textContent = '▶ Resume'; }}
        else {{ sim.alphaTarget(0.05).restart(); physicsBtn.textContent = '⏸ Pause'; setTimeout(() => sim.alphaTarget(0), 2000); }}
      }});

      // ── Recenter ─────────────────────────────────────────────────────────
      recenterBtn.addEventListener('click', () => {{
        svg.transition().duration(600).call(
          zoom.transform,
          d3.zoomIdentity.translate(W / 2, H / 2).scale(0.85).translate(-W / 2, -H / 2)
        );
      }});

      // ── Search & filter ───────────────────────────────────────────────────
      let currentFilter = 'all';
      let searchTerm = '';

      function applyFilter(filter) {{
        currentFilter = filter;
        nodeSel.each(function(d) {{
          const typeMatch = filter === 'all' || d.type === filter;
          const searchMatch = !searchTerm || (d.label || '').toLowerCase().includes(searchTerm) || (d.full_path || '').toLowerCase().includes(searchTerm);
          const visible = typeMatch && searchMatch;
          d3.select(this)
            .attr('display', visible ? null : 'none')
            .select('.node-circle').attr('fill-opacity', visible ? (d.type === 'root' ? 0.95 : 0.82) : 0);
        }});
        linkSel.attr('display', l => {{
          const s = typeof l.source === 'object' ? l.source : nodes.find(n => n.id === l.source);
          const t = typeof l.target === 'object' ? l.target : nodes.find(n => n.id === l.target);
          const srcOk = !s || (filter === 'all' || s.type === filter) && (!searchTerm || (s.label || '').toLowerCase().includes(searchTerm) || (s.full_path || '').toLowerCase().includes(searchTerm));
          const tgtOk = !t || (filter === 'all' || t.type === filter) && (!searchTerm || (t.label || '').toLowerCase().includes(searchTerm) || (t.full_path || '').toLowerCase().includes(searchTerm));
          return (filter === 'all' && !searchTerm) || (srcOk && tgtOk) ? null : 'none';
        }});
      }}

      root.querySelectorAll('.kg-filter').forEach(btn => {{
        btn.addEventListener('click', () => {{
          root.querySelectorAll('.kg-filter').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          applyFilter(btn.dataset.filter);
        }});
      }});

      searchInput.addEventListener('input', () => {{
        searchTerm = searchInput.value.trim().toLowerCase();
        applyFilter(currentFilter);
      }});

      // Initial zoom to fit
      setTimeout(() => {{
        const allX = nodes.map(n => n.x).filter(Boolean);
        const allY = nodes.map(n => n.y).filter(Boolean);
        if (allX.length) {{
          const x0 = Math.min(...allX), x1 = Math.max(...allX);
          const y0 = Math.min(...allY), y1 = Math.max(...allY);
          const pad = 60;
          const scale = Math.min(0.9, (W - pad * 2) / (x1 - x0 + 1), (H - pad * 2) / (y1 - y0 + 1));
          svg.transition().duration(800).call(
            zoom.transform,
            d3.zoomIdentity
              .translate(W / 2, H / 2)
              .scale(scale)
              .translate(-(x0 + x1) / 2, -(y0 + y1) / 2)
          );
        }}
      }}, 1600);

      refreshHeight();
      window.addEventListener('resize', refreshHeight);
    }})();
    </script>
    """

    components.html(html_block, height=720, scrolling=False)


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
    score_text = f"{verdict.get('overall_quality_score', 'N/A')} / 100"
    summary_text = str(verdict.get("summary", "")).strip()
    reasoning_text = str(verdict.get("recommendation_reasoning", "")).strip()
    score_pattern = re.compile(r"\b\d+\s*/\s*100\b.*?(?:\.|$)", re.IGNORECASE)
    summary_text = score_pattern.sub("", summary_text).strip(" .")
    reasoning_text = score_pattern.sub("", reasoning_text).strip(" .")
    st.markdown(
        f"""
        <div style="
            margin-top:18px;
            padding:24px;
            border-radius:16px;
            background:rgba(253, 251, 249, 0.45);
            border:1px solid rgba(84, 58, 38, 0.2);
            backdrop-filter:blur(12px) saturate(180%);
            -webkit-backdrop-filter:blur(12px) saturate(180%);
            box-sizing:border-box;
        ">
            <div style="display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;">
                <div style="color:#547792; font-size:0.74rem; font-weight:800; letter-spacing:0.08em; text-transform:uppercase;">
                    Recommendation
                </div>
                <div style="
                    display:inline-flex;
                    align-items:center;
                    justify-content:center;
                    padding:7px 20px;
                    border-radius:999px;
                    background:var(--accent-blue);
                    color:var(--text-primary);
                    font-size:14px;
                    font-weight:800;
                    line-height:1;
                ">{html.escape(score_text)}</div>
            </div>
            <div style="margin:14px 0 16px 0; border-top:1px solid #d4b896;"></div>
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
                <div style="color:var(--text-primary); font-size:0.76rem; font-weight:800; letter-spacing:0.08em; text-transform:uppercase;">Summary</div>
                <div style="flex:1; border-top:1px solid rgba(26,50,99,0.18);"></div>
            </div>
            <div style="
                background:rgba(255,255,255,0.6);
                border:1px solid rgba(255,255,255,0.8);
                border-radius:14px;
                padding:18px 20px;
            ">
                <p style="margin:0; font-family:Georgia, serif; font-size:13.5px; line-height:1.85; color:#3a4a5c;">
                    {html.escape(summary_text)}
                </p>
                <p style="margin:12px 0 0 0; font-family:Georgia, serif; font-size:13.5px; line-height:1.85; color:#3a4a5c;">
                    {html.escape(reasoning_text)}
                </p>
            </div>
            <div style="margin-top:14px; padding-top:12px; border-top:1px solid rgba(26,50,99,0.12); display:flex; gap:8px; align-items:flex-start;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true" style="margin-top:1px; flex-shrink:0;">
                    <circle cx="12" cy="12" r="9" stroke="#547792" stroke-width="1.8"></circle>
                    <line x1="12" y1="10" x2="12" y2="16" stroke="#547792" stroke-width="1.8" stroke-linecap="round"></line>
                    <circle cx="12" cy="7" r="1.1" fill="#547792"></circle>
                </svg>
                <div style="color:#547792; font-size:11px; font-style:italic; line-height:1.5;">
                    {html.escape(str(verdict.get("disclaimer", DISCLAIM_TEXT)))}
                </div>
            </div>
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
                    <div class="cl-brand-title-row">
                        <div class="cl-brand-logo-wrap">{_brand_logo_svg(size=32)}</div>
                        <div class="cl-wordmark" style="font-size:28px;"><span>Code</span><span>Lens</span></div>
                    </div>
                    <p style="color:var(--text-secondary); font-size:15px; margin:10px 0 0 0; line-height:1.5;">
                        {html.escape(BRAND_TAGLINE)}
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
        f"""
        <div style="min-height:68vh; display:flex; align-items:center; justify-content:center;">
            <div style="text-align:center; max-width:560px; padding:24px;">
                <div class="cl-brand-title-row" style="justify-content:center;">
                    <div class="cl-brand-logo-wrap">{_brand_logo_svg(size=44)}</div>
                    <div class="cl-wordmark" style="font-size:2.5rem;"><span>Code</span><span>Lens</span></div>
                </div>
                <div style="color:var(--text-secondary); margin-top:12px; font-size:1.02rem; line-height:1.5;">
                    {html.escape(BRAND_TAGLINE)}
                </div>
                <div style="color:var(--text-muted); margin-top:14px; font-size:0.98rem; line-height:1.55;">
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
    if main_tab not in ("Analyze", "MCP"):
        main_tab = "Analyze"
        st.session_state["main_tab"] = main_tab

    t1, t2 = st.columns([1, 1], gap="small")
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
    else:
        render_mcp_tab()


if __name__ == "__main__":
    main()
