"""Retrieval helpers for CodeLens baseline and claim comparison."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Repo root on sys.path so `python rag/retriever.py` can import `tools.*`
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llama_index.core import Settings
from llama_index.embeddings.voyageai import VoyageEmbedding

from tools.pinecone_tool import PineconeStore
from tools.project_env import load_project_env

load_project_env()

Settings.embed_model = VoyageEmbedding(
    model_name="voyage-code-3",
    voyage_api_key=os.getenv("VOYAGE_API_KEY"),
)
Settings.llm = None


class CodeRetriever:
    """Compare code style baselines and fetch support for claims."""

    def get_baseline_comparison(
        self,
        code_text: str,
        pinecone_store: PineconeStore,
    ) -> dict[str, Any]:
        human_matches = pinecone_store.query_similar(code_text, namespace="human-baseline", top_k=3)
        ai_matches = pinecone_store.query_similar(code_text, namespace="ai-baseline", top_k=3)

        human_similarity = self._average_top_scores(human_matches)
        ai_similarity = self._average_top_scores(ai_matches)

        if human_similarity > ai_similarity + 0.05:
            style_lean = "human"
        elif ai_similarity > human_similarity + 0.05:
            style_lean = "ai"
        else:
            style_lean = "mixed"

        return {
            "human_similarity": human_similarity,
            "ai_similarity": ai_similarity,
            "top_human_matches": self._top_texts(human_matches, limit=2),
            "top_ai_matches": self._top_texts(ai_matches, limit=2),
            "style_lean": style_lean,
        }

    def get_context_for_claim(
        self,
        claim: str,
        candidate_namespace: str,
        pinecone_store: PineconeStore,
        top_k: int = 4,
    ) -> list[dict[str, Any]]:
        matches = pinecone_store.query_similar(claim, namespace=candidate_namespace, top_k=top_k)
        return [
            {
                "id": match.get("id", ""),
                "score": float(match.get("score", 0.0)),
                "metadata": match.get("metadata", {}),
                "text": match.get("text", ""),
            }
            for match in matches
        ]

    @staticmethod
    def _average_top_scores(matches: list[dict[str, Any]]) -> float:
        if not matches:
            return 0.0
        scores = [float(match.get("score", 0.0)) for match in matches[:3]]
        return round(sum(scores) / len(scores), 3)

    @staticmethod
    def _top_texts(matches: list[dict[str, Any]], limit: int = 2) -> list[str]:
        texts: list[str] = []
        for match in matches[:limit]:
            text = (match.get("text") or "").strip()
            if text:
                texts.append(text[:500])
        return texts
