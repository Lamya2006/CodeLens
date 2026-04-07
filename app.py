import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.project_env import load_project_env

load_project_env()

st.set_page_config(
    page_title="CodeLens",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def render_header() -> None:
    st.title("CodeLens")
    st.caption("Intelligent code review for technical hiring.")


def render_analyze_tab() -> None:
    st.subheader("Repository Analysis")
    github_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/owner/repo")
    resume_text = st.text_area(
        "Resume Text (Optional)",
        placeholder="Paste candidate resume text here...",
        height=180,
    )
    company_github_url = st.text_input(
        "Company GitHub URL for Style Comparison (Optional)",
        placeholder="https://github.com/company/repo",
    )

    if st.button("Analyze Repository", type="primary"):
        st.info("Results will appear here.")
    else:
        st.info("Results will appear here.")

    _ = github_url, resume_text, company_github_url


def render_evaluation_tab() -> None:
    st.subheader("Evaluation")
    st.write("Evaluation suite coming soon.")


def render_mcp_tab() -> None:
    st.subheader("MCP Info")
    st.write("MCP server info coming soon.")


def apply_theme() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #0b1020 0%, #111827 100%);
                color: #e5e7eb;
            }

            [data-testid="stHeader"] {
                background: rgba(11, 16, 32, 0.85);
            }

            [data-testid="stTabs"] button {
                color: #d1d5db;
            }

            [data-testid="stTabs"] button[aria-selected="true"] {
                color: #f9fafb;
            }

            [data-testid="stTextInput"] input,
            [data-testid="stTextArea"] textarea {
                background-color: #111827;
                color: #f9fafb;
            }

            div[data-baseweb="base-input"] {
                background-color: #111827;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_theme()
    render_header()
    analyze_tab, evaluation_tab, mcp_tab = st.tabs(["Analyze", "Evaluation", "MCP Info"])

    with analyze_tab:
        render_analyze_tab()

    with evaluation_tab:
        render_evaluation_tab()

    with mcp_tab:
        render_mcp_tab()


if __name__ == "__main__":
    main()
