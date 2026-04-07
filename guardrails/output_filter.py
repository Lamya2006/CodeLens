"""Output validation and post-processing guardrails for CodeLens."""

from __future__ import annotations

import copy
import re
import statistics
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import requests


class OutputFilter:
    """Apply validation and safety framing to repository analysis output."""

    DISCLAIMER = (
        "CodeLens provides probabilistic signals to assist human judgment. "
        "All findings should be verified in a technical interview."
    )
    SOURCE_FILE_EXTENSIONS = {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".cs",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".swift",
        ".kt",
        ".scala",
        ".sql",
    }
    ABSOLUTE_LANGUAGE_REPLACEMENTS = {
        "definitely ai": "strong signals of AI involvement",
        "proof of": "evidence suggests",
        "confirmed cheating": "patterns consistent with inappropriate assistance",
        "certainly": "likely",
    }

    def __init__(self) -> None:
        self._last_commit_count = 0
        self._last_total_lines = 0
        self._last_pre_llm_warning = False

    def validate_repo_url(self, url: str) -> None:
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or parsed.netloc not in {"github.com", "www.github.com"}:
            raise ValueError("Please enter a valid GitHub repository URL.")

        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if len(parts) < 2:
            raise ValueError("Please provide a specific GitHub repository URL, not just a user or organization page.")

        repo_url = f"https://github.com/{parts[0]}/{parts[1].removesuffix('.git')}"
        try:
            response = requests.head(repo_url, timeout=5, allow_redirects=True)
        except requests.RequestException as exc:
            raise ValueError("Unable to verify the GitHub repository right now. Please try again.") from exc

        if response.status_code == 404:
            raise ValueError("GitHub repository not found. Please check the URL and try again.")
        if response.status_code == 403:
            raise ValueError("This GitHub repository is not accessible. It may be private or rate-limited.")
        if response.status_code >= 400:
            raise ValueError("Unable to verify the GitHub repository. Please check the URL and try again.")

    def validate_repo_size(self, files: list[dict[str, Any]], commits: list[dict[str, Any]]) -> None:
        commit_count = len(commits)

        code_files = [file_data for file_data in files if self._is_code_file(file_data)]
        if not code_files:
            raise ValueError("No source code files detected")

        total_lines = sum(self._estimate_lines(file_data) for file_data in code_files)
        if total_lines < 100:
            raise ValueError("Repository too small")

        self._last_commit_count = commit_count
        self._last_total_lines = total_lines

    def filter_verdict(self, verdict: dict[str, Any]) -> dict[str, Any]:
        cleaned = self._replace_language(copy.deepcopy(verdict))
        cleaned = self._clamp_scores(cleaned)
        cleaned = self._penalize_sparse_commit_history(cleaned)
        cleaned["data_confidence"] = self._compute_confidence()
        cleaned["pre_llm_warning"] = self._compute_pre_llm_warning()
        cleaned["disclaimer"] = self.DISCLAIMER
        return cleaned

    def _penalize_sparse_commit_history(self, verdict: dict[str, Any]) -> dict[str, Any]:
        """Few commits = weak signal; cap scores instead of blocking analysis."""
        n = self._last_commit_count
        if n >= 3:
            return verdict

        cap_commit_health = {0: 15, 1: 25, 2: 35}.get(n, 35)
        ch = verdict.get("commit_health_score")
        if isinstance(ch, int):
            verdict["commit_health_score"] = min(ch, cap_commit_health)

        overall_penalty = {0: 18, 1: 12, 2: 8}.get(n, 8)
        oq = verdict.get("overall_quality_score")
        if isinstance(oq, int):
            verdict["overall_quality_score"] = max(0, oq - overall_penalty)

        concerns = verdict.get("concerns")
        if isinstance(concerns, list):
            msg = (
                f"Sparse Git history ({n} commit{'s' if n != 1 else ''} in the analyzed sample) — "
                "commit-based signals are limited; verify practices in interview."
            )
            merged = list(concerns)
            if msg not in merged:
                merged.append(msg)
            verdict["concerns"] = merged[:5]

        return verdict

    def flag_vibe_coding(
        self,
        files: list[dict[str, Any]],
        commit_patterns: dict[str, Any],
    ) -> list[str]:
        flags: list[str] = []

        if 0 < self._last_commit_count < 3:
            flags.append(
                f"Sparse commit history ({self._last_commit_count} commits in sample) — "
                "weak signal for development practices"
            )

        pct_top3 = float(commit_patterns.get("pct_code_in_top3_commits", 0.0))
        if pct_top3 > 70:
            flags.append(
                f"Large initial commit — {pct_top3:.0f}% of code introduced in first 3 commits"
            )

        comment_densities = [
            float(file_data.get("comment_density", 0.0))
            for file_data in files
            if file_data.get("comment_density") is not None
        ]
        if len(comment_densities) >= 2:
            comment_std_dev = statistics.pstdev(comment_densities)
            if comment_std_dev < 0.05:
                flags.append(
                    "Suspiciously uniform comment density across all files "
                    f"(std dev: {comment_std_dev:.2f})"
                )

        if files and not any(
            bool(file_data.get("has_todos")) or bool(file_data.get("has_debug_prints")) or bool(file_data.get("has_commented_out_code"))
            for file_data in files
        ):
            flags.append(
                "No debug artifacts found — zero TODOs, debug prints, or commented-out code "
                f"across {len(files)} files"
            )

        if bool(commit_patterns.get("single_branch")):
            flags.append("Single branch, linear commit history")

        diff_std_dev = float(commit_patterns.get("diff_size_std_dev", 0.0))
        if diff_std_dev < 15:
            flags.append(
                f"Low commit variance — all diffs suspiciously similar in size (std dev: {diff_std_dev:.1f} lines)"
            )

        message_avg_length = float(commit_patterns.get("message_avg_length", 0.0))
        if message_avg_length > 60:
            flags.append(
                f"Commit message formality unusually high (avg length: {message_avg_length:.0f} chars)"
            )

        return flags

    def _compute_confidence(self) -> str:
        if self._last_commit_count < 5 or self._last_total_lines < 300:
            return "low"
        if self._last_commit_count < 20:
            return "medium"
        return "high"

    def _compute_pre_llm_warning(self) -> bool:
        return self._last_pre_llm_warning

    def set_analysis_context(
        self,
        commits: list[dict[str, Any]],
        files: list[dict[str, Any]],
    ) -> None:
        self._last_commit_count = len(commits)
        self._last_total_lines = sum(self._estimate_lines(file_data) for file_data in files if self._is_code_file(file_data))
        self._last_pre_llm_warning = self._calculate_pre_llm_warning(commits)

    def _calculate_pre_llm_warning(self, commits: list[dict[str, Any]]) -> bool:
        if not commits:
            return False
        cutoff = datetime.fromisoformat("2022-01-01T00:00:00")
        dated_commits = 0
        older_commits = 0
        for commit in commits:
            timestamp = commit.get("timestamp")
            if not timestamp:
                continue
            try:
                commit_dt = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            except ValueError:
                continue
            dated_commits += 1
            if commit_dt.replace(tzinfo=None) < cutoff:
                older_commits += 1
        if dated_commits == 0:
            return False
        return (older_commits / dated_commits) > 0.5

    def _replace_language(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._replace_language(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._replace_language(item) for item in value]
        if isinstance(value, str):
            updated = value
            for source, replacement in self.ABSOLUTE_LANGUAGE_REPLACEMENTS.items():
                updated = re.sub(source, replacement, updated, flags=re.IGNORECASE)
            return updated
        return value

    def _clamp_scores(self, value: Any) -> Any:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in value.items():
                if isinstance(item, int) and key.endswith("_score"):
                    out[key] = max(0, min(100, item))
                else:
                    out[key] = self._clamp_scores(item)
            return out
        if isinstance(value, list):
            return [self._clamp_scores(item) for item in value]
        return value

    def _is_code_file(self, file_data: dict[str, Any]) -> bool:
        language = file_data.get("language")
        if not language:
            return False
        file_path = str(file_data.get("file_path") or file_data.get("path") or "")
        suffix = re.search(r"(\.[A-Za-z0-9]+)$", file_path)
        if suffix and suffix.group(1).lower() in self.SOURCE_FILE_EXTENSIONS:
            return True
        return bool(language)

    @staticmethod
    def _estimate_lines(file_data: dict[str, Any]) -> int:
        if "line_count" in file_data and isinstance(file_data["line_count"], int):
            return max(file_data["line_count"], 0)
        content = file_data.get("content") or file_data.get("text") or ""
        if isinstance(content, str) and content.strip():
            return len(content.splitlines())
        symbols = file_data.get("symbols") or file_data.get("parsed_symbols") or []
        total = 0
        if isinstance(symbols, dict):
            symbols = symbols.get("items", [])
        for symbol in symbols:
            if not isinstance(symbol, dict):
                continue
            symbol_text = symbol.get("code") or symbol.get("text") or symbol.get("content") or ""
            if isinstance(symbol_text, str):
                total += len(symbol_text.splitlines())
        return total
