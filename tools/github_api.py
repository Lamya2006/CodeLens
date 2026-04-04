"""GitHub repository analysis utilities powered by PyGithub."""

from __future__ import annotations

import os
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from github import Auth, Github
from github.ContentFile import ContentFile
from github.GithubException import GithubException, RateLimitExceededException, UnknownObjectException
from github.Repository import Repository


class GithubAnalyzer:
    """Analyze repository metadata, commits, and collaboration signals."""

    def __init__(self, repo_url: str, max_retries: int = 3, retry_delay: int = 5) -> None:
        self.repo_url = repo_url.strip()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.github = self._build_client()
        self.owner, self.repo_name = self._parse_repo_url(self.repo_url)
        self.repo = self._load_repo()

    @staticmethod
    def _parse_repo_url(repo_url: str) -> tuple[str, str]:
        parsed = urlparse(repo_url)
        if parsed.netloc not in {"github.com", "www.github.com"}:
            raise ValueError("Please provide a valid GitHub repository URL.")

        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if len(parts) < 2:
            raise ValueError("GitHub repository URL must include both owner and repository name.")

        owner = parts[0]
        repo_name = parts[1].removesuffix(".git")
        if not owner or not repo_name:
            raise ValueError("GitHub repository URL must include both owner and repository name.")
        return owner, repo_name

    @staticmethod
    def _build_client() -> Github:
        token = os.getenv("GITHUB_TOKEN", "").strip()
        if token:
            return Github(auth=Auth.Token(token))
        return Github()

    def _retry(self, operation: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return operation()
            except RateLimitExceededException as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                retry_after = self._rate_limit_wait_seconds()
                time.sleep(max(retry_after, self.retry_delay))
            except GithubException as exc:
                last_error = exc
                if exc.status == 403 and attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("GitHub operation failed unexpectedly.")

    def _rate_limit_wait_seconds(self) -> int:
        try:
            rate_limit = self.github.get_rate_limit().core
            reset_at = rate_limit.reset
            if reset_at.tzinfo is None:
                reset_at = reset_at.replace(tzinfo=timezone.utc)
            seconds = int((reset_at - datetime.now(timezone.utc)).total_seconds()) + 1
            return max(seconds, self.retry_delay)
        except Exception:
            return self.retry_delay

    def _load_repo(self) -> Repository:
        full_name = f"{self.owner}/{self.repo_name}"
        try:
            return self._retry(lambda: self.github.get_repo(full_name))
        except UnknownObjectException as exc:
            raise ValueError(
                "Repository not found. Please check the GitHub URL or access permissions."
            ) from exc
        except GithubException as exc:
            if exc.status in {401, 403, 404}:
                raise ValueError(
                    "Unable to access this repository. It may be private, missing, or require a valid GITHUB_TOKEN."
                ) from exc
            raise ValueError(f"GitHub request failed: {exc.data}") from exc

    @staticmethod
    def _to_iso(timestamp: datetime | None) -> str | None:
        if timestamp is None:
            return None
        return timestamp.isoformat()

    @staticmethod
    def _language_percentages(languages: dict[str, int]) -> dict[str, float]:
        total_bytes = sum(languages.values())
        if total_bytes == 0:
            return {}
        return {
            language: round((byte_count / total_bytes) * 100, 2)
            for language, byte_count in languages.items()
        }

    def _get_commit_objects(self, limit: int = 100) -> list[Any]:
        commits = self._retry(lambda: self.repo.get_commits())
        commit_list: list[Any] = []
        for index, commit in enumerate(commits):
            if index >= limit:
                break
            commit_list.append(commit)
        return commit_list

    def _commit_file_stats(self, commit: Any) -> tuple[int, int, int, int]:
        detailed_commit = self._retry(lambda: self.repo.get_commit(commit.sha))
        files = list(detailed_commit.files)
        changed_files = len(files)
        additions = sum(file.additions for file in files)
        deletions = sum(file.deletions for file in files)
        total_changes = sum(file.changes for file in files)
        return changed_files, additions, deletions, total_changes

    def get_repo_metadata(self) -> dict[str, Any]:
        languages = self._retry(lambda: self.repo.get_languages())
        contributors = self._retry(lambda: self.repo.get_contributors().totalCount)
        return {
            "name": self.repo.full_name,
            "description": self.repo.description or "",
            "primary_language": self.repo.language,
            "language_breakdown": self._language_percentages(languages),
            "star_count": self.repo.stargazers_count,
            "fork_count": self.repo.forks_count,
            "open_issue_count": self.repo.open_issues_count,
            "created_at": self._to_iso(self.repo.created_at),
            "last_pushed_at": self._to_iso(self.repo.pushed_at),
            "contributor_count": contributors,
            "has_wiki": bool(self.repo.has_wiki),
            "has_projects": bool(getattr(self.repo, "has_projects", False)),
            "default_branch_name": self.repo.default_branch,
        }

    def get_commits(self, limit: int = 100) -> list[dict[str, Any]]:
        commit_objects = self._get_commit_objects(limit=limit)
        commits: list[dict[str, Any]] = []

        for commit in commit_objects:
            changed_files, additions, deletions, total_changes = self._commit_file_stats(commit)
            author_name = None
            if commit.commit.author is not None:
                author_name = commit.commit.author.name
            elif commit.author is not None:
                author_name = commit.author.login

            commits.append(
                {
                    "sha": commit.sha,
                    "message": commit.commit.message,
                    "author_name": author_name or "Unknown",
                    "timestamp": self._to_iso(commit.commit.author.date),
                    "files_changed": changed_files,
                    "lines_added": additions,
                    "lines_deleted": deletions,
                    "total_diff_size": total_changes,
                }
            )

        return commits

    def get_commit_patterns(self) -> dict[str, Any]:
        commits = self.get_commits(limit=100)
        if not commits:
            return {
                "commits_per_day": 0.0,
                "largest_single_commit_lines": 0,
                "pct_code_in_top3_commits": 0.0,
                "avg_diff_size": 0.0,
                "diff_size_std_dev": 0.0,
                "commit_time_distribution": {
                    "morning": 0,
                    "afternoon": 0,
                    "evening": 0,
                    "night": 0,
                },
                "message_avg_length": 0.0,
                "single_branch": True,
            }

        additions = [commit["lines_added"] for commit in commits]
        diff_sizes = [commit["total_diff_size"] for commit in commits]
        message_lengths = [len(commit["message"]) for commit in commits]

        active_days = {
            datetime.fromisoformat(commit["timestamp"]).date()
            for commit in commits
            if commit["timestamp"] is not None
        }
        commits_per_day = round(len(commits) / max(len(active_days), 1), 2)

        total_lines_added = sum(additions)
        top_three_additions = sum(sorted(additions, reverse=True)[:3])
        pct_code_in_top3 = round(
            (top_three_additions / total_lines_added) * 100, 2
        ) if total_lines_added else 0.0

        time_distribution = Counter({"morning": 0, "afternoon": 0, "evening": 0, "night": 0})
        for commit in commits:
            if commit["timestamp"] is None:
                continue
            hour = datetime.fromisoformat(commit["timestamp"]).hour
            if 5 <= hour < 12:
                bucket = "morning"
            elif 12 <= hour < 17:
                bucket = "afternoon"
            elif 17 <= hour < 22:
                bucket = "evening"
            else:
                bucket = "night"
            time_distribution[bucket] += 1

        branch_count = self._retry(lambda: self.repo.get_branches().totalCount)

        return {
            "commits_per_day": commits_per_day,
            "largest_single_commit_lines": max(additions),
            "pct_code_in_top3_commits": pct_code_in_top3,
            "avg_diff_size": round(statistics.mean(diff_sizes), 2),
            "diff_size_std_dev": round(statistics.pstdev(diff_sizes), 2),
            "commit_time_distribution": dict(time_distribution),
            "message_avg_length": round(statistics.mean(message_lengths), 2),
            "single_branch": branch_count == 1,
        }

    def get_readme(self) -> str:
        try:
            readme_file = self._retry(lambda: self.repo.get_readme())
        except UnknownObjectException:
            return ""
        except GithubException as exc:
            if exc.status == 404:
                return ""
            raise

        if isinstance(readme_file, ContentFile):
            content = readme_file.decoded_content
            return content.decode("utf-8", errors="replace")
        return ""

    def get_file_tree(self) -> list[str]:
        tree = self._retry(lambda: self.repo.get_git_tree(self.repo.default_branch, recursive=True))
        return sorted(
            item.path
            for item in tree.tree
            if item.type == "blob" and ".git" not in item.path.split("/")
        )

    def get_pr_and_issue_stats(self) -> dict[str, int]:
        total_prs = self._retry(lambda: self.repo.get_pulls(state="all").totalCount)
        total_issues = self._retry(
            lambda: self.github.search_issues(f"repo:{self.repo.full_name} is:issue").totalCount
        )
        closed_issues = self._retry(
            lambda: self.github.search_issues(
                f"repo:{self.repo.full_name} is:issue is:closed"
            ).totalCount
        )

        merged_count = 0
        if total_prs:
            pulls = self._retry(lambda: self.repo.get_pulls(state="closed", sort="updated"))
            for pull in pulls:
                if pull.merged_at is not None:
                    merged_count += 1

        return {
            "total_prs": total_prs,
            "merged_prs": merged_count,
            "total_issues": total_issues,
            "closed_issues": closed_issues,
        }


if __name__ == "__main__":
    TEST_REPO_URL = "https://github.com/pallets/flask"

    analyzer = GithubAnalyzer(TEST_REPO_URL)
    print("Metadata:")
    print(analyzer.get_repo_metadata())
    print("\nCommits:")
    print(analyzer.get_commits(limit=5))
    print("\nCommit Patterns:")
    print(analyzer.get_commit_patterns())
    print("\nREADME Preview:")
    print(analyzer.get_readme()[:500])
    print("\nFile Tree Sample:")
    print(analyzer.get_file_tree()[:25])
    print("\nPR and Issue Stats:")
    print(analyzer.get_pr_and_issue_stats())
