# To connect to Claude Desktop, add to claude_desktop_config.json:
# {
#   "mcpServers": {
#     "codelens": {
#       "command": "python",
#       "args": ["/absolute/path/to/mcp_server.py"]
#     }
#   }
# }

"""FastMCP server exposing the CodeLens analysis pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Repo root on sys.path so `python mcp_server.py` can import local packages.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastmcp import FastMCP

from agents.crew import CodeLensCrew
from guardrails.output_filter import OutputFilter
from rag.indexer import CodeIndexer
from rag.retriever import CodeRetriever
from tools.github_api import GithubAnalyzer
from tools.gitnexus_tool import GitNexusAnalyzer
from tools.pinecone_tool import PineconeStore
from tools.resume_parser import ResumeParser, SkillMatcher


mcp = FastMCP(name="codelens")


def _json_response(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, default=str)


def _error_response(message: str, *, tool: str, details: str | None = None) -> str:
    payload: dict[str, Any] = {"ok": False, "tool": tool, "error": message}
    if details:
        payload["details"] = details
    return _json_response(payload)


def _candidate_username_from_url(github_url: str) -> str:
    parsed = urlparse(github_url)
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    return parts[0] if parts else "candidate"


def _read_pdf_file(pdf_path: str) -> bytes:
    path = Path(pdf_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise ValueError(f"Resume PDF not found: {path}")
    return path.read_bytes()


def _build_code_sample(files: list[dict[str, Any]], max_chars: int = 6000) -> str:
    samples: list[str] = []
    total = 0
    for file_data in files:
        text = file_data.get("content") or file_data.get("text") or ""
        if not text:
            symbols = file_data.get("symbols") or file_data.get("parsed_symbols") or []
            if isinstance(symbols, dict):
                symbols = symbols.get("items", [])
            symbol_snippets = []
            for symbol in symbols[:5]:
                if isinstance(symbol, dict):
                    snippet = symbol.get("code") or symbol.get("text") or symbol.get("content") or ""
                    if snippet:
                        symbol_snippets.append(str(snippet))
            text = "\n\n".join(symbol_snippets)
        if not text:
            continue

        header = f"# File: {file_data.get('file_path') or file_data.get('path') or 'unknown'}\n"
        chunk = header + str(text).strip()
        remaining = max_chars - total
        if remaining <= 0:
            break
        chunk = chunk[:remaining]
        samples.append(chunk)
        total += len(chunk)
        if total >= max_chars:
            break
    return "\n\n".join(samples)


def _build_company_style_summary(company_github_url: str) -> str:
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


def _run_repo_analysis_pipeline(
    github_url: str,
    resume_pdf_path: str | None = None,
    job_description: str | None = None,
    company_github_url: str | None = None,
) -> dict[str, Any]:
    output_filter = OutputFilter()
    output_filter.validate_repo_url(github_url)

    github_analyzer = GithubAnalyzer(github_url)
    repo_metadata = github_analyzer.get_repo_metadata()
    commits = github_analyzer.get_commits()
    commit_patterns = github_analyzer.get_commit_patterns()

    candidate_username = _candidate_username_from_url(github_url)

    with GitNexusAnalyzer(github_url) as gitnexus:
        files = gitnexus.get_file_contents()
        knowledge_graph = gitnexus.get_knowledge_graph()
        output_filter.validate_repo_size(files, commits)
        output_filter.set_analysis_context(commits, files)

        pinecone_store = PineconeStore()
        indexer = CodeIndexer()
        retriever = CodeRetriever()
        indexer.index_repo_files(files, candidate_username, pinecone_store)

        code_sample = _build_code_sample(files)
        baseline_comparison = retriever.get_baseline_comparison(code_sample, pinecone_store)

    resume_data = None
    skill_matches = None
    project_matches = None
    parsed_job_description = None

    if resume_pdf_path:
        pdf_bytes = _read_pdf_file(resume_pdf_path)
        resume_parser = ResumeParser()
        resume_data = resume_parser.parse_from_pdf(pdf_bytes)
        skill_matcher = SkillMatcher(PineconeStore())
        skill_matches = skill_matcher.match_skills_to_code(
            resume_data.get("skills", []),
            f"candidate-{candidate_username}",
        )
        project_matches = [
            skill_matcher.match_project_claims(project, f"candidate-{candidate_username}")
            for project in resume_data.get("projects", [])
        ]

    if job_description:
        parsed_job_description = ResumeParser().parse_job_description(job_description)

    company_style_summary = None
    if company_github_url:
        output_filter.validate_repo_url(company_github_url)
        company_style_summary = _build_company_style_summary(company_github_url)

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

    verdict = CodeLensCrew(analysis_data).run()
    verdict = output_filter.filter_verdict(verdict)
    verdict["vibe_coding_flags"] = output_filter.flag_vibe_coding(files, commit_patterns)
    verdict["indexed_namespace"] = f"candidate-{candidate_username}"
    return verdict


@mcp.tool(
    description=(
        "Run the full CodeLens analysis pipeline on a public GitHub repository. "
        "Returns structured code quality, AI usage assessment, and optional resume/JD matching."
    )
)
def analyze_repo(
    github_url: str,
    resume_pdf_path: str | None = None,
    job_description: str | None = None,
    company_github_url: str | None = None,
) -> str:
    try:
        verdict = _run_repo_analysis_pipeline(
            github_url=github_url,
            resume_pdf_path=resume_pdf_path,
            job_description=job_description,
            company_github_url=company_github_url,
        )
        return _json_response({"ok": True, "result": verdict})
    except Exception as exc:
        return _error_response(
            "Failed to complete repository analysis.",
            tool="analyze_repo",
            details=str(exc),
        )


@mcp.tool(
    description=(
        "Fetch and analyze commit behavior patterns. Returns velocity, diff distribution, timing, "
        "and message analysis."
    )
)
def get_commit_patterns(github_url: str) -> str:
    try:
        OutputFilter().validate_repo_url(github_url)
        patterns = GithubAnalyzer(github_url).get_commit_patterns()
        return _json_response({"ok": True, "result": patterns})
    except Exception as exc:
        return _error_response(
            "Failed to fetch commit patterns.",
            tool="get_commit_patterns",
            details=str(exc),
        )


@mcp.tool(
    description=(
        "Compare a code snippet against human-written and AI-generated baselines. "
        "Returns similarity scores and examples."
    )
)
def compare_to_baseline(code_snippet: str, language: str = "python") -> str:
    try:
        store = PineconeStore()
        retriever = CodeRetriever()
        comparison = retriever.get_baseline_comparison(code_snippet, store)
        comparison["language"] = language
        return _json_response({"ok": True, "result": comparison})
    except Exception as exc:
        return _error_response(
            "Failed to compare snippet against the baseline corpus.",
            tool="compare_to_baseline",
            details=str(exc),
        )


@mcp.tool(
    description=(
        "Semantically match resume claims against actual repo code. "
        "Returns per-skill and per-project evidence scores."
    )
)
def match_resume_to_repo(resume_pdf_path: str, github_url: str) -> str:
    try:
        output_filter = OutputFilter()
        output_filter.validate_repo_url(github_url)
        pdf_bytes = _read_pdf_file(resume_pdf_path)
        candidate_username = _candidate_username_from_url(github_url)

        with GitNexusAnalyzer(github_url) as gitnexus:
            files = gitnexus.get_file_contents()
            commits = GithubAnalyzer(github_url).get_commits()
            output_filter.validate_repo_size(files, commits)
            pinecone_store = PineconeStore()
            CodeIndexer().index_repo_files(files, candidate_username, pinecone_store)

        parser = ResumeParser()
        resume_data = parser.parse_from_pdf(pdf_bytes)
        matcher = SkillMatcher(PineconeStore())
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
        return _json_response(
            {
                "ok": True,
                "result": {
                    "resume_data": resume_data,
                    "skill_matches": skill_matches,
                    "project_matches": project_matches,
                    "undeclared_skills": undeclared_skills,
                },
            }
        )
    except Exception as exc:
        return _error_response(
            "Failed to match resume claims to repository code.",
            tool="match_resume_to_repo",
            details=str(exc),
        )


if __name__ == "__main__":
    mcp.run()
