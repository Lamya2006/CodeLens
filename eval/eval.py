"""Evaluation helpers for the CodeLens golden dataset."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Repo root on sys.path so `python eval/eval.py` can import local packages.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.crew import CodeLensCrew
from guardrails.output_filter import OutputFilter
from rag.indexer import CodeIndexer
from rag.retriever import CodeRetriever
from tools.github_api import GithubAnalyzer
from tools.gitnexus_tool import open_repo_analyzer
from tools.pinecone_tool import PineconeStore
from tools.project_env import load_project_env

load_project_env()


class EvalSuite:
    """Run CodeLens against a labeled golden dataset."""

    def __init__(self) -> None:
        self._last_per_repo_results: list[dict[str, Any]] = []

    def run(self, dataset_path: str = "eval/golden_dataset.json", lite: bool = True) -> dict[str, Any]:
        dataset = self._load_dataset(dataset_path)
        predictions: list[str] = []
        actuals: list[str] = []
        per_repo_results: list[dict[str, Any]] = []

        for entry in dataset:
            url = entry["url"]
            actual = entry["label"]
            try:
                verdict = self._run_single_repo(url, lite=lite)
                scores = {
                    "overall_quality_score": verdict.get("overall_quality_score"),
                    "ai_usage_score": verdict.get("ai_usage_score"),
                    "commit_health_score": verdict.get("commit_health_score"),
                    "recommendation": verdict.get("recommendation"),
                }
                ai_usage_score = float(verdict.get("ai_usage_score", 0))
                predicted = "ai_heavy" if ai_usage_score < 45 else "human"
                correct = predicted == actual
                per_repo_results.append(
                    {
                        "url": url,
                        "actual": actual,
                        "predicted": predicted,
                        "correct": correct,
                        "scores": scores,
                    }
                )
            except Exception as exc:
                predicted = "ai_heavy"
                per_repo_results.append(
                    {
                        "url": url,
                        "actual": actual,
                        "predicted": predicted,
                        "correct": predicted == actual,
                        "scores": {
                            "overall_quality_score": None,
                            "ai_usage_score": None,
                            "commit_health_score": None,
                            "recommendation": "error",
                            "error": str(exc),
                        },
                    }
                )

            predictions.append(predicted)
            actuals.append(actual)

        self._last_per_repo_results = per_repo_results
        return {
            "dataset_path": dataset_path,
            "lite": lite,
            "predictions": predictions,
            "actuals": actuals,
            "per_repo_results": per_repo_results,
        }

    def compute_metrics(self, predictions: list[str], actuals: list[str]) -> dict[str, Any]:
        tp = fp = tn = fn = 0
        for predicted, actual in zip(predictions, actuals, strict=False):
            if predicted == "ai_heavy" and actual == "ai_heavy":
                tp += 1
            elif predicted == "ai_heavy" and actual == "human":
                fp += 1
            elif predicted == "human" and actual == "human":
                tn += 1
            elif predicted == "human" and actual == "ai_heavy":
                fn += 1

        total = len(actuals)
        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
            "per_repo_results": self._last_per_repo_results,
        }

    def save_results(self, results: dict[str, Any], path: str = "eval/eval_results.json") -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **results,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_dataset(self, dataset_path: str) -> list[dict[str, Any]]:
        path = Path(dataset_path)
        return json.loads(path.read_text(encoding="utf-8"))

    def _run_single_repo(self, github_url: str, lite: bool = True) -> dict[str, Any]:
        output_filter = OutputFilter()
        output_filter.validate_repo_url(github_url)

        github_analyzer = GithubAnalyzer(github_url)
        repo_metadata = github_analyzer.get_repo_metadata()
        commits = github_analyzer.get_commits()
        commit_patterns = github_analyzer.get_commit_patterns()
        candidate_username = self._candidate_username_from_url(github_url)

        with open_repo_analyzer(github_url) as gitnexus:
            files = gitnexus.get_file_contents()
            knowledge_graph = gitnexus.get_knowledge_graph()
            output_filter.validate_repo_size(files, commits)
            output_filter.set_analysis_context(commits, files)

            store = PineconeStore()
            indexer = CodeIndexer()
            retriever = CodeRetriever()
            indexer.index_repo_files(files, candidate_username, store)
            baseline_comparison = retriever.get_baseline_comparison(
                self._build_code_sample(files),
                store,
            )

        analysis_data = {
            "repo_metadata": repo_metadata,
            "commits": commits,
            "commit_patterns": commit_patterns,
            "files": files,
            "knowledge_graph": knowledge_graph,
            "baseline_comparison": baseline_comparison,
            "resume_data": None,
            "skill_matches": None,
            "project_matches": None,
            "job_description": None,
            "company_style_summary": None,
        }
        if not lite:
            analysis_data["notes"] = "Full mode requested, but golden dataset entries do not include resume or JD context."

        verdict = CodeLensCrew(analysis_data).run()
        verdict = output_filter.filter_verdict(verdict)
        verdict["vibe_coding_flags"] = output_filter.flag_vibe_coding(files, commit_patterns)
        return verdict

    @staticmethod
    def _candidate_username_from_url(github_url: str) -> str:
        parsed = urlparse(github_url)
        parts = [part for part in parsed.path.strip("/").split("/") if part]
        return parts[0] if parts else "candidate"

    @staticmethod
    def _build_code_sample(files: list[dict[str, Any]], max_chars: int = 6000) -> str:
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
            header = f"# File: {file_data.get('file_path') or file_data.get('path') or 'unknown'}\n"
            chunk = header + str(text).strip()
            remaining = max_chars - total
            if remaining <= 0:
                break
            trimmed = chunk[:remaining]
            samples.append(trimmed)
            total += len(trimmed)
            if total >= max_chars:
                break
        return "\n\n".join(samples)


if __name__ == "__main__":
    suite = EvalSuite()
    results = suite.run()
    metrics = suite.compute_metrics(results["predictions"], results["actuals"])
    suite.save_results(metrics)
    print(f"Accuracy: {metrics['accuracy']:.1%}")
    print(f"F1: {metrics['f1']:.2f}")
