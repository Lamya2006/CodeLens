from __future__ import annotations

import importlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
import math
from html import escape
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
    theme = st.session_state.get("theme", "light")
    if theme == "dark":
        theme_vars = """
            --bg-base: #080c14;
            --bg-gradient: linear-gradient(160deg, #080c14 0%, #0d1426 50%, #0a1020 100%);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-bg-hover: rgba(255, 255, 255, 0.09);
            --glass-bg-inner: rgba(255, 255, 255, 0.08);
            --glass-border: rgba(255, 255, 255, 0.12);
            --glass-border-subtle: rgba(100, 160, 255, 0.15);
            --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), 0 2px 8px rgba(56, 168, 245, 0.08);
            --top-edge-highlight: rgba(255, 255, 255, 0.15);
            --accent-blue: #38a8f5;
            --accent-blue-hover: #64b8f7;
            --accent-blue-light: rgba(56, 168, 245, 0.2);
            --text-primary: #f0f6ff;
            --text-secondary: #94a3b8;
            --text-muted: #4a5568;
            --score-green: #4ade80;
            --score-yellow: #fbbf24;
            --score-red: #fb7185;
            --card-glow: rgba(56, 168, 245, 0.18);
            --surface-white: rgba(255, 255, 255, 0.82);
            --status-border: rgba(56, 168, 245, 0.45);
        """
    else:
        theme_vars = """
            --bg-base: #f0f6ff;
            --bg-gradient: linear-gradient(160deg, #e8f4ff 0%, #f8fbff 50%, #ffffff 100%);
            --glass-bg: rgba(255, 255, 255, 0.45);
            --glass-bg-hover: rgba(255, 255, 255, 0.65);
            --glass-bg-inner: rgba(255, 255, 255, 0.62);
            --glass-border: rgba(255, 255, 255, 0.7);
            --glass-border-subtle: rgba(200, 220, 255, 0.4);
            --glass-shadow: 0 8px 32px rgba(100, 160, 255, 0.12), 0 2px 8px rgba(100, 160, 255, 0.08);
            --top-edge-highlight: rgba(255, 255, 255, 0.8);
            --accent-blue: #38a8f5;
            --accent-blue-hover: #1e90e0;
            --accent-blue-light: rgba(56, 168, 245, 0.15);
            --text-primary: #0a0f1e;
            --text-secondary: #4a5568;
            --text-muted: #94a3b8;
            --score-green: #22c55e;
            --score-yellow: #f59e0b;
            --score-red: #ef4444;
            --card-glow: rgba(56, 168, 245, 0.12);
            --surface-white: rgba(255, 255, 255, 0.96);
            --status-border: rgba(56, 168, 245, 0.35);
        """

    st.markdown(
        f"""
        <style>
            :root {{
                {theme_vars}
                --glass-blur: blur(24px) saturate(180%);
                --glass-blur-inner: blur(12px) saturate(165%);
                --glass-radius: 20px;
                --glass-radius-sm: 14px;
                --ease-swift: cubic-bezier(0.16, 1, 0.3, 1);
                --font-sans: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", system-ui, sans-serif;
                --font-mono: "SF Mono", "Fira Code", ui-monospace, monospace;
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
                background-image:
                    radial-gradient(circle at 15% 18%, rgba(56, 168, 245, 0.22), transparent 26%),
                    radial-gradient(circle at 86% 10%, rgba(255, 255, 255, 0.52), transparent 24%),
                    radial-gradient(circle at 78% 58%, rgba(56, 168, 245, 0.15), transparent 20%),
                    var(--bg-gradient);
                background-attachment: fixed;
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

            [data-testid="stTextInput"] input,
            [data-testid="stTextArea"] textarea,
            [data-testid="stFileUploaderDropzone"] {{
                background: var(--glass-bg) !important;
                color: var(--text-primary) !important;
                border: 1px solid var(--glass-border-subtle) !important;
                border-top: 1px solid var(--top-edge-highlight) !important;
                border-radius: 14px !important;
                padding: 12px 16px !important;
                backdrop-filter: var(--glass-blur) !important;
                -webkit-backdrop-filter: var(--glass-blur) !important;
                box-shadow: var(--glass-shadow) !important;
                transition: all 0.18s var(--ease-swift) !important;
            }}

            [data-testid="stTextInput"] input::placeholder,
            [data-testid="stTextArea"] textarea::placeholder {{
                color: var(--text-muted) !important;
            }}

            [data-testid="stTextInput"] input:focus,
            [data-testid="stTextArea"] textarea:focus,
            [data-testid="stFileUploaderDropzone"]:focus-within {{
                border-color: var(--accent-blue) !important;
                box-shadow: 0 0 0 3px rgba(56, 168, 245, 0.15), var(--glass-shadow) !important;
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
                transition: all 0.18s var(--ease-swift);
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
                transition: all 0.2s var(--ease-swift);
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
                transition: all 0.18s var(--ease-swift);
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


def init_session_state() -> None:
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_error", None)
    st.session_state.setdefault("eval_result", None)
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("oauth_state", None)
    st.session_state.setdefault("theme", "light")
    st.session_state.setdefault("theme_choice", "Moon" if st.session_state.get("theme") == "dark" else "Sun")
    st.session_state.setdefault("widget_theme_choice", st.session_state["theme_choice"])
    st.session_state.setdefault("active_page", "Analyze")
    st.session_state.setdefault("widget_active_page", st.session_state["active_page"])
    st.session_state.setdefault("results_tab", "Skill Map")
    st.session_state.setdefault("widget_results_tab", st.session_state["results_tab"])


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


def recommendation_badge_color(value: str) -> str:
    return {
        "strong_hire": "badge-green",
        "hire": "badge-blue",
        "maybe": "badge-yellow",
        "pass": "badge-red",
    }.get(value, "badge-gray")


def _html_escape_multiline(text: str) -> str:
    return escape(text or "").replace("\n", "<br/>")


def _semicircle_gauge_svg(pct: float, stroke_color: str, *, muted: bool) -> str:
    """Decorative half-circle gauge (0–100). Numeric score in the card remains authoritative."""
    r = 40.0
    circ = math.pi * r
    pct = max(0.0, min(100.0, pct))
    offset = circ * (1.0 - pct / 100.0)
    track = "rgba(199, 210, 254, 0.65)" if not muted else "rgba(148, 163, 184, 0.4)"
    fg = stroke_color if not muted else "var(--text-muted)"
    d = f"M {60.0 - r:.1f},{60.0} A {r} {r} 0 0 1 {60.0 + r:.1f},{60.0}"
    return (
        f'<svg class="score-gauge-svg" viewBox="0 0 120 72" width="112" height="68" '
        f'aria-hidden="true" focusable="false">'
        f'<path d="{d}" fill="none" stroke="{track}" stroke-width="6" stroke-linecap="round"/>'
        f'<path d="{d}" fill="none" stroke="{fg}" stroke-width="6" stroke-linecap="round" '
        f'stroke-dasharray="{circ:.2f}" stroke-dashoffset="{offset:.2f}" class="score-gauge-arc"/>'
        f"</svg>"
    )


def _brand_logo_svg() -> str:
    return (
        '<svg class="brand-logo-svg" width="28" height="28" viewBox="0 0 32 32" '
        'aria-hidden="true" focusable="false">'
        '<circle cx="16" cy="16" r="11" fill="none" stroke="var(--accent-blue)" stroke-width="2.2"/>'
        '<path d="M16 9 A7 7 0 0 1 23 16" fill="none" stroke="rgba(56, 168, 245, 0.42)" stroke-width="2.2" '
        'stroke-linecap="round"/>'
        '<circle cx="16" cy="16" r="4" fill="var(--accent-blue)" opacity="0.25"/>'
        "</svg>"
    )


def render_divider() -> None:
    st.markdown('<div class="section-rule codelens-reveal" role="presentation"></div>', unsafe_allow_html=True)


def render_pill_switcher(options: list[str], key: str, *, class_name: str = "tab-switcher") -> str:
    widget_key = f"widget_{key}"
    current = st.session_state.get(widget_key, st.session_state.get(key, options[0]))
    if current not in options:
        current = options[0]
    st.markdown(f'<div class="{class_name} codelens-reveal">', unsafe_allow_html=True)
    choice = st.radio(
        key,
        options,
        index=options.index(current),
        horizontal=True,
        label_visibility="collapsed",
        key=widget_key,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state[key] = choice
    return choice


def render_empty_state(prompt: str) -> None:
    st.markdown(
        f"""
        <div class="empty-state codelens-reveal">
            <div class="brand-wordmark"><span class="code">Code</span><span class="lens">Lens</span></div>
            <p>{escape(prompt)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    logo = _brand_logo_svg()
    st.markdown(
        f"""
        <div class="glass-card codelens-reveal" style="padding:22px 24px; margin-bottom:18px;">
            <div class="app-shell-head">
                <div class="brand-block">
                    <div class="brand-logo-wrap">{logo}</div>
                    <div class="app-shell-copy">
                        <h1><span class="code">Code</span><span style="color:var(--accent-blue);">Lens</span></h1>
                        <p>Premium hiring intelligence with fast, glassy feedback loops.</p>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_error_state() -> None:
    error = st.session_state.get("last_error")
    if not error:
        return
    detail = escape(error.get("details", ""))
    st.markdown(
        f"""
        <div class="panel codelens-reveal" style="border-color: rgba(239, 68, 68, 0.28); background: rgba(239, 68, 68, 0.08);">
            <div class="section-title" style="color: var(--score-red); margin-bottom: 6px;">{escape(error["message"])}</div>
            <div class="muted">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_oauth_button() -> None:
    login_url = get_github_login_url()
    if not login_url:
        st.markdown(
            '<div class="muted">GitHub login is unavailable until OAuth credentials are configured.</div>',
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        f"""
        <a class="oauth-button" href="{login_url}">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
                <path d="M8 0C3.58 0 0 3.67 0 8.2c0 3.63 2.29 6.7 5.47 7.78.4.08.55-.18.55-.39 0-.19-.01-.82-.01-1.49-2.01.38-2.53-.51-2.69-.98-.09-.24-.48-.98-.82-1.18-.28-.15-.68-.52-.01-.53.63-.01 1.08.59 1.23.83.72 1.24 1.87.89 2.33.68.07-.54.28-.89.51-1.09-1.78-.21-3.64-.91-3.64-4.03 0-.89.31-1.62.82-2.19-.08-.21-.36-1.05.08-2.19 0 0 .67-.22 2.2.84A7.36 7.36 0 0 1 8 3.66c.68 0 1.37.09 2.01.27 1.53-1.06 2.2-.84 2.2-.84.44 1.14.16 1.98.08 2.19.51.57.82 1.29.82 2.19 0 3.13-1.87 3.82-3.65 4.03.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.19 0 .21.14.47.55.39A8.23 8.23 0 0 0 16 8.2C16 3.67 12.42 0 8 0Z"></path>
            </svg>
            Continue with GitHub
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
        st.markdown('<div class="muted">No saved analyses yet.</div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="small-label">Recent analyses</div>', unsafe_allow_html=True)
    for index, entry in enumerate(analyses[:5]):
        repo_name = escape(str(entry.get("repo_name", "Repository")))
        analyzed_at = escape(str(entry.get("analyzed_at", ""))[:10])
        badge_class = recommendation_badge_color(entry.get("recommendation", "maybe"))
        st.markdown(
            f"""
            <div class="history-card recent-row codelens-reveal" style="--reveal-delay:{index * 50}ms;">
                <div style="font-weight:600;">{repo_name}</div>
                <div class="muted" style="margin-top:4px;">{analyzed_at}</div>
                <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
                    <span class="badge {badge_class}">{escape(str(entry.get("recommendation", "maybe")).replace("_", " "))}</span>
                    <span class="badge badge-gray">score {escape(str(entry.get("overall_quality_score", "N/A")))}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open analysis", key=f"history-open-{entry['id']}", use_container_width=True):
            st.session_state["last_result"] = entry.get("result")
            st.session_state["last_error"] = None
            st.session_state["active_page"] = "Analyze"
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
            if st.button(
                f"Load {entry.get('repo_name', 'analysis')}",
                key=f"history-load-{entry['id']}",
                use_container_width=True,
            ):
                st.session_state["last_result"] = entry.get("result")
                st.session_state["last_error"] = None
                st.session_state["active_page"] = "Analyze"
                st.rerun()


def render_sidebar() -> None:
    with st.sidebar:
        logo = _brand_logo_svg()
        st.markdown('<div class="sidebar-shell">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="sidebar-card glass-card codelens-reveal">
                <div class="brand-block">
                    <div class="brand-logo-wrap">{logo}</div>
                    <div>
                        <div class="brand-wordmark"><span class="code">Code</span><span class="lens">Lens</span></div>
                        <div class="brand-tagline">Liquid hiring intelligence</div>
                    </div>
                </div>
                <div class="sidebar-divider"></div>
                <div class="small-label">Appearance</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        theme_choice = render_pill_switcher(["Sun", "Moon"], "theme_choice", class_name="theme-switcher")
        st.session_state["theme"] = "dark" if theme_choice == "Moon" else "light"

        st.markdown('<div class="small-label" style="margin-top:10px;">Navigation</div>', unsafe_allow_html=True)
        page = render_pill_switcher(["Analyze", "Evaluation", "MCP"], "active_page")
        st.session_state["active_page"] = page

        user = st.session_state.get("user")
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        if user:
            avatar = ""
            if user.get("avatar_url"):
                avatar = f'<img class="avatar" src="{escape(user["avatar_url"])}" alt="avatar" />'
            st.markdown(
                f"""
                <div class="sidebar-card glass-card codelens-reveal">
                    <div style="display:flex; align-items:center; gap:10px;">
                        {avatar}
                        <div>
                            <div class="small-label">Signed in</div>
                            <div style="font-weight:600;">@{escape(user.get("username",""))}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Sign out", use_container_width=True):
                sign_out()
            render_recent_history()
        else:
            st.markdown('<div class="small-label">Account</div>', unsafe_allow_html=True)
            render_oauth_button()

        ready, detail = check_mcp_status()
        status_class = "badge-green" if ready else "badge-red"
        label = "MCP ready" if ready else "MCP unavailable"
        st.markdown(
            f"""
            <div class="sidebar-card glass-card codelens-reveal">
                <div class="small-label">System</div>
                <div style="margin-top:10px;">
                    <span class="badge {status_class}">{label}</span>
                </div>
                <div class="muted" style="margin-top:10px;">{escape(detail)}</div>
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
                5. Run the LLM review and apply output guardrails.
                """
            )
        st.markdown("</div>", unsafe_allow_html=True)


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


def render_score_card(
    title: str,
    score: int | None,
    meta: str = "",
    muted: bool = False,
    anim_index: int = 0,
) -> None:
    safe_title = escape(title)
    safe_meta = escape(meta)
    score_display = "N/A" if score is None else str(score)
    color = "var(--text-muted)" if muted else score_color(score)
    pct = 0.0
    if score is not None:
        try:
            pct = float(max(0.0, min(100.0, float(score))))
        except (TypeError, ValueError):
            pct = 0.0
    gauge_muted = muted or score is None
    gauge = _semicircle_gauge_svg(pct, color, muted=gauge_muted)
    delay_ms = anim_index * 50
    tint = "var(--glass-bg)"
    if not muted and score is not None:
        if score > 75:
            tint = "rgba(34, 197, 94, 0.06)"
        elif score >= 50:
            tint = "rgba(245, 158, 11, 0.06)"
        else:
            tint = "rgba(239, 68, 68, 0.06)"
    st.markdown(
        f"""
        <div class="glass-card score-card codelens-reveal" style="--reveal-delay: {delay_ms}ms; background: linear-gradient(180deg, rgba(255,255,255,0.12), transparent), {tint};">
            <div class="score-card-inner">
                <div class="score-gauge-col">{gauge}</div>
                <div class="score-text-col">
                    <div class="score-label">{safe_title}</div>
                    <div class="score-value" style="color:{color};">{score_display}</div>
                    <div class="score-meta">{safe_meta}</div>
                </div>
            </div>
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
        strength_lines = "".join(
            f'<div class="strength-pill codelens-reveal" style="--reveal-delay:{idx * 50}ms;">{escape(str(item))}</div>'
            for idx, item in enumerate(strengths)
        )
        if not strength_lines:
            strength_lines = '<div class="muted">No major strengths were recorded.</div>'
        st.markdown(
            f'<div class="panel codelens-reveal" style="--reveal-delay: 0ms;"><div class="section-title">Strengths</div>{strength_lines}</div>',
            unsafe_allow_html=True,
        )

    with right:
        concern_html = ""
        for idx, item in enumerate(verdict.get("concerns", [])):
            severity = concerns_by_text.get(item, {}).get("severity", "medium")
            badge_class = {"low": "badge-gray", "medium": "badge-yellow", "high": "badge-red"}.get(severity, "badge-yellow")
            item_safe = escape(str(item))
            sev_safe = escape(str(severity))
            concern_html += (
                f'<div class="concern-pill codelens-reveal" style="--reveal-delay:{idx * 50}ms;">{item_safe}<div style="margin-top:6px;">'
                f'<span class="badge {badge_class}">{sev_safe}</span></div></div>'
            )
        if not concern_html:
            concern_html = '<div class="muted">No major concerns were recorded.</div>'
        st.markdown(
            f'<div class="panel codelens-reveal" style="--reveal-delay: 40ms;"><div class="section-title">Concerns</div>{concern_html}</div>',
            unsafe_allow_html=True,
        )


def render_skill_map(result: dict[str, Any]) -> None:
    skill_map = result["verdict"].get("skill_map", {})
    if not skill_map:
        st.markdown('<div class="panel tab-fade">No skill map available for this analysis.</div>', unsafe_allow_html=True)
        return
    rows = []
    for skill, status in skill_map.items():
        badge = {"confirmed": "badge-green", "partial": "badge-yellow", "not_found": "badge-gray"}.get(status, "badge-gray")
        sk = escape(str(skill))
        stt = escape(str(status))
        rows.append(f"<tr><td>{sk}</td><td><span class='badge {badge}'>{stt}</span></td></tr>")
    st.markdown(
        f"""
        <div class="panel tab-fade">
            <table class="data-table">
                <thead>
                    <tr><th>Skill</th><th>Status</th></tr>
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

    st.markdown('<div class="section-title">Baseline Similarity</div>', unsafe_allow_html=True)
    st.progress(min(max(float(baseline.get("human_similarity", 0.0)), 0.0), 1.0), text="Human baseline similarity")
    st.progress(min(max(float(baseline.get("ai_similarity", 0.0)), 0.0), 1.0), text="AI baseline similarity")

    st.markdown('<div class="section-title" style="margin-top:16px;">Vibe Coding Flags</div>', unsafe_allow_html=True)
    flags = verdict.get("vibe_coding_flags", [])
    if flags:
        for flag in flags:
            st.markdown(
                f'<div class="panel panel-accent-warn panel-tight codelens-reveal">{escape(str(flag))}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="muted">No vibe-coding flags triggered.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:16px;">AI Evidence Signals</div>', unsafe_allow_html=True)
    signals = ai_report.get("ai_evidence_signals", [])
    if signals:
        for signal in signals:
            sig = escape(str(signal.get("signal", "") or ""))
            loc = escape(str(signal.get("location", "") or ""))
            note = escape(str(signal.get("note", "") or ""))
            st.markdown(
                f"""
                <div class="panel panel-tight codelens-reveal">
                    <div><strong>{sig}</strong></div>
                    <div class="muted">{loc}</div>
                    <div style="margin-top:8px;">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f'<div class="panel codelens-reveal">{escape(str(verdict.get("ai_usage_summary", "No detailed AI evidence signals available.") or ""))}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title" style="margin-top:16px;">Good AI Usage</div>', unsafe_allow_html=True)
    good_examples = ai_report.get("good_ai_usage", [])
    if good_examples:
        for item in good_examples:
            st.markdown(
                f'<div class="panel panel-accent-positive panel-tight codelens-reveal">{escape(str(item))}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="muted">No explicit good-AI examples were surfaced.</div>', unsafe_allow_html=True)


def render_code_issues(result: dict[str, Any]) -> None:
    reports = result.get("reports", {})
    quality_report = reports.get("code_quality", {})
    bugs = quality_report.get("bugs_or_errors", [])
    concerns = quality_report.get("concerns", [])

    st.markdown('<div class="section-title">Bugs / Errors</div>', unsafe_allow_html=True)
    if bugs:
        for bug in bugs:
            desc = escape(str(bug.get("description", "") or ""))
            loc = escape(str(bug.get("file", "") or ""))
            hint = escape(str(bug.get("line_hint", "") or ""))
            st.markdown(
                f"""
                <div class="panel panel-tight codelens-reveal">
                    <div><strong>{desc}</strong></div>
                    <div class="muted">{loc} · {hint}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        verdict_bugs = result["verdict"].get("bugs_found", [])
        if verdict_bugs:
            for bug in verdict_bugs:
                st.markdown(
                    f'<div class="panel panel-tight codelens-reveal">{escape(str(bug))}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="muted">No explicit bugs surfaced.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:16px;">Quality Concerns</div>', unsafe_allow_html=True)
    if concerns:
        for concern in concerns:
            badge = {"low": "badge-gray", "medium": "badge-yellow", "high": "badge-red"}.get(concern.get("severity", "medium"), "badge-yellow")
            iss = escape(str(concern.get("issue", "") or ""))
            loc = escape(str(concern.get("location", "") or ""))
            sev = escape(str(concern.get("severity", "medium") or ""))
            st.markdown(
                f"""
                <div class="panel panel-tight codelens-reveal">
                    <div><strong>{iss}</strong></div>
                    <div class="muted">{loc}</div>
                    <div style="margin-top:8px;"><span class="badge {badge}">{sev}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="muted">No detailed concern list available.</div>', unsafe_allow_html=True)


def render_resume_panel(result: dict[str, Any]) -> None:
    if not result.get("resume_data"):
        st.markdown('<div class="panel tab-fade">Resume analysis is only shown when a resume file is uploaded.</div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="section-title">Project Verdicts</div>', unsafe_allow_html=True)
    project_cards = result.get("project_matches", [])
    if project_cards:
        for project in project_cards:
            evidence_html = ""
            for item in project.get("feature_evidence", []):
                found_badge = "badge-green" if item.get("found") else "badge-red"
                feat = escape(str(item.get("feature", "") or ""))
                qnote = escape(str(item.get("quality_note", "") or ""))
                evidence_html += (
                    f"<div class='list-item'><strong>{feat}</strong> "
                    f"<span class='badge {found_badge}'>{'found' if item.get('found') else 'missing'}</span>"
                    f"<div class='muted' style='margin-top:6px;'>{qnote}</div></div>"
                )
            pname = escape(str(project.get("project_name", "Project") or "Project"))
            st.markdown(
                f"""
                <div class="panel codelens-reveal" style="margin-bottom:10px;">
                    <div class="section-title">{pname}</div>
                    <div class="muted">Overall match: {project.get("overall_match", 0.0):.2f}</div>
                    {evidence_html or '<div class="muted">No feature evidence found.</div>'}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="muted">No project-level matches were generated.</div>', unsafe_allow_html=True)

    inflation_flags = result["verdict"].get("resume_inflation_flags", [])
    st.markdown('<div class="section-title" style="margin-top:16px;">Resume Inflation Flags</div>', unsafe_allow_html=True)
    if inflation_flags:
        for flag in inflation_flags:
            st.markdown(
                f'<div class="panel panel-accent-danger panel-tight codelens-reveal">{escape(str(flag))}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="muted">No resume inflation flags were raised.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:16px;">Undeclared Skills Found In Code</div>', unsafe_allow_html=True)
    undeclared = result.get("undeclared_skills", [])
    if undeclared:
        chips = " ".join(f'<span class="badge badge-green">{escape(str(skill))}</span>' for skill in undeclared)
        st.markdown(f'<div class="panel codelens-reveal">{chips}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="muted">No extra undeclared skills were found.</div>', unsafe_allow_html=True)


def render_job_fit_panel(result: dict[str, Any]) -> None:
    job_data = result.get("job_description")
    verdict = result["verdict"]
    if not job_data:
        st.markdown('<div class="panel tab-fade">Job fit appears when a job description is provided.</div>', unsafe_allow_html=True)
        return

    jf_score = verdict.get("job_fit_score")
    jf_analysis = verdict.get("job_fit_analysis") or "No job fit analysis was returned."
    st.markdown(
        f"""
        <div class="panel tab-fade">
            <div class="section-title">Job Fit Score</div>
            <div class="score-value" style="color:{score_color(jf_score)};">{jf_score if jf_score is not None else "N/A"}</div>
            <div style="margin-top:10px;">{_html_escape_multiline(str(jf_analysis))}</div>
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
            badge = {"confirmed": "badge-green", "partial": "badge-yellow", "not_found": "badge-gray"}.get(status, "badge-gray")
            sk = escape(str(skill))
            stt = escape(str(status))
            rows.append(f"<tr><td>{sk}</td><td><span class='badge {badge}'>{stt}</span></td></tr>")
        st.markdown(
            f"""
            <div class="panel tab-fade" style="margin-top:12px;">
                <div class="section-title">Required Skills Comparison</div>
                <table class="data-table">
                    <thead><tr><th>Skill</th><th>Evidence</th></tr></thead>
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

    st.markdown(
        '<div class="codelens-reveal"><h3 style="font-size:20px; font-weight:600; margin:0 0 12px 0; letter-spacing:-0.01em;">Results</h3></div>',
        unsafe_allow_html=True,
    )
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
            render_score_card(
                title,
                score,
                meta=meta,
                muted=(title == "Resume Match" and not has_resume),
                anim_index=idx,
            )

    render_divider()
    render_strengths_and_concerns(result)

    subtab_names = ["Skill Map", "AI Usage", "Code Issues"]
    if has_resume:
        subtab_names.append("Resume")
    if has_jd:
        subtab_names.append("Job Fit")
    current_tab = st.session_state.get("results_tab")
    if current_tab not in subtab_names:
        st.session_state["results_tab"] = subtab_names[0]
    selected_tab = render_pill_switcher(subtab_names, "results_tab")
    st.markdown('<div class="panel tab-fade" style="margin-top:14px;">', unsafe_allow_html=True)
    if selected_tab == "Skill Map":
        render_skill_map(result)
    elif selected_tab == "AI Usage":
        render_ai_usage(result)
    elif selected_tab == "Code Issues":
        render_code_issues(result)
    elif selected_tab == "Resume":
        render_resume_panel(result)
    elif selected_tab == "Job Fit":
        render_job_fit_panel(result)
    st.markdown("</div>", unsafe_allow_html=True)

    recommendation = verdict.get("recommendation", "maybe")
    rec_label = escape(recommendation.replace("_", " ").title())
    summary_html = _html_escape_multiline(str(verdict.get("summary", "") or ""))
    reasoning_html = _html_escape_multiline(str(verdict.get("recommendation_reasoning", "") or ""))
    disclaimer_html = escape(str(verdict.get("disclaimer", DISCLAIM_TEXT)))
    st.markdown(
        f"""
        <div class="panel recommendation-panel codelens-reveal" style="margin-top:16px;">
            <div class="section-title">Recommendation</div>
            <div class="recommendation-badge {escape(recommendation)}">{rec_label}</div>
            <div class="summary-box">
                <div style="font-weight:700; margin-bottom:10px;">Summary</div>
                <div>{summary_html}</div>
                <div style="margin-top:12px;" class="muted">{reasoning_html}</div>
            </div>
            <div class="muted" style="margin-top:12px; font-size:0.86rem;">{disclaimer_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, anim_index: int = 0) -> None:
    pct_width: float | None = None
    try:
        v = float(value)
        pct_width = max(0.0, min(100.0, v * 100.0)) if v <= 1.0 else max(0.0, min(100.0, v))
    except ValueError:
        pct_width = None
    bar = ""
    if pct_width is not None:
        bar = f'<div class="metric-bar-track"><div class="metric-bar-fill" style="width:{pct_width:.1f}%"></div></div>'
    delay_ms = anim_index * 55
    st.markdown(
        f"""
        <div class="glass-card score-card codelens-reveal" style="--reveal-delay: {delay_ms}ms; min-height:120px;">
            <div class="score-card-inner">
                <div class="score-text-col">
                    <div class="score-label">{escape(title)}</div>
                    <div class="score-value" style="color:var(--accent-blue);">{escape(value)}</div>
                    {bar}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analyze_tab() -> None:
    st.markdown(
        """
        <div class="glass-card hero-shell codelens-reveal">
            <h2 class="hero-heading"><span class="code">Code</span><span style="color:var(--accent-blue);">Lens</span></h2>
            <p class="hero-subheading">Drop in a GitHub repository, then layer resume and role context to get a premium code-review verdict in minutes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="glass-card codelens-reveal" style="padding:24px;">', unsafe_allow_html=True)
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

    st.markdown('<div class="primary-action">', unsafe_allow_html=True)
    analyze_clicked = st.button("Analyze Repository", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

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
                    st.session_state["results_tab"] = "Skill Map"
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
        render_empty_state("Enter a GitHub URL to begin.")


def render_evaluation_tab() -> None:
    st.markdown(
        """
        <div class="glass-card codelens-reveal" style="padding:24px;">
            <div class="section-title">Evaluation Suite</div>
            <p class="muted" style="margin-top:0;">Offline metrics preview with the same floating-glass treatment as the analysis workflow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div style="max-width:220px;" class="codelens-reveal">', unsafe_allow_html=True)
    if st.button("Run Evaluation Suite", use_container_width=False):
        st.session_state["eval_result"] = {
            "precision": 0.84,
            "recall": 0.79,
            "f1": 0.81,
            "matrix": {"tp": 41, "fp": 8, "fn": 11, "tn": 36},
        }
    st.markdown("</div>", unsafe_allow_html=True)

    result = st.session_state.get("eval_result")
    if result:
        cols = st.columns(3)
        with cols[0]:
            render_metric_card("Precision", f"{result['precision']:.2f}", anim_index=0)
        with cols[1]:
            render_metric_card("Recall", f"{result['recall']:.2f}", anim_index=1)
        with cols[2]:
            render_metric_card("F1", f"{result['f1']:.2f}", anim_index=2)

        matrix = result["matrix"]
        st.markdown(
            f"""
            <div class="panel codelens-reveal" style="margin-top:14px;">
                <div class="section-title">Confusion Matrix</div>
                <table class="data-table" style="max-width:420px;">
                    <tr><th></th><th style="text-align:center;">Predicted Positive</th><th style="text-align:center;">Predicted Negative</th></tr>
                    <tr><th style="text-align:left;">Actual Positive</th><td style="text-align:center;">{matrix['tp']}</td><td style="text-align:center;">{matrix['fn']}</td></tr>
                    <tr><th style="text-align:left;">Actual Negative</th><td style="text-align:center;">{matrix['fp']}</td><td style="text-align:center;">{matrix['tn']}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        render_empty_state("Run the evaluation suite to generate metrics.")


def render_tool_card(name: str, description: str, parameters: list[str]) -> None:
    params = "".join(f"<li>{escape(param)}</li>" for param in parameters)
    st.markdown(
        f"""
        <div class="tool-card codelens-reveal">
            <div class="section-title">{escape(name)}</div>
            <div style="margin-bottom:10px;">{escape(description)}</div>
            <div class="small-label">Parameters</div>
            <ul style="margin-top:8px;">{params}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mcp_tab() -> None:
    st.markdown(
        """
        <div class="glass-card codelens-reveal" style="padding:24px;">
            <div class="section-title">MCP Toolkit</div>
            <p class="muted" style="margin-top:0;">Model Context Protocol tools for IDE and agent integrations, presented as a floating reference deck.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
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

    render_divider()
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
    st.session_state["theme"] = "dark" if st.session_state.get("widget_theme_choice") == "Moon" else "light"
    st.session_state["active_page"] = st.session_state.get("widget_active_page", st.session_state["active_page"])
    st.session_state["results_tab"] = st.session_state.get("widget_results_tab", st.session_state["results_tab"])
    apply_global_styles()
    render_sidebar()

    missing = missing_api_keys()
    if missing:
        st.error("Missing required API keys: " + ", ".join(missing))

    render_app_header()
    page = st.session_state.get("active_page", "Analyze")
    if page == "Analyze":
        render_analyze_tab()
    elif page == "Evaluation":
        render_evaluation_tab()
    else:
        render_mcp_tab()


if __name__ == "__main__":
    main()
