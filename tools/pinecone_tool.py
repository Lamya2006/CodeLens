"""Pinecone-backed vector store utilities for CodeLens."""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root on sys.path so `python tools/pinecone_tool.py` can import `tools.*`
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import math
import os
import time
from typing import Any

from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException
import voyageai

from tools.project_env import load_project_env

load_project_env()


class PineconeStore:
    """Manage code embeddings and retrieval for CodeLens."""

    INDEX_NAME = "codelens"
    DIMENSION = 1024
    MODEL_NAME = "voyage-code-3"
    MAX_TEXT_CHARS = 16000
    BATCH_SIZE = 64
    UPSERT_BATCH_SIZE = 100

    def __init__(self) -> None:
        pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
        voyage_api_key = os.getenv("VOYAGE_API_KEY", "").strip()

        if not pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is required to use PineconeStore.")
        if not voyage_api_key:
            raise RuntimeError("VOYAGE_API_KEY is required to use PineconeStore.")

        self.pc = Pinecone(api_key=pinecone_api_key)
        self.voyage = voyageai.Client(api_key=voyage_api_key)
        self.index = self._create_or_connect_index()

    def _create_or_connect_index(self) -> Any:
        if not self._index_exists():
            self.pc.create_index(
                name=self.INDEX_NAME,
                dimension=self.DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            self._wait_for_index()
        return self.pc.Index(self.INDEX_NAME)

    def _index_exists(self) -> bool:
        indexes = self.pc.list_indexes()
        if hasattr(indexes, "names"):
            names = indexes.names()
        else:
            names = [
                item["name"] if isinstance(item, dict) else getattr(item, "name", "")
                for item in indexes
            ]
        return self.INDEX_NAME in names

    def _wait_for_index(self, timeout_seconds: int = 60) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            description = self.pc.describe_index(self.INDEX_NAME)
            status = description.status if hasattr(description, "status") else description.get("status", {})
            ready = status.ready if hasattr(status, "ready") else status.get("ready", False)
            if ready:
                return
            time.sleep(2)
        raise RuntimeError(f"Pinecone index '{self.INDEX_NAME}' was not ready before timeout.")

    @staticmethod
    def _truncate(text: str) -> str:
        return text[: PineconeStore.MAX_TEXT_CHARS]

    def embed_text(self, text: str) -> list[float]:
        truncated = self._truncate(text)
        response = self.voyage.embed(
            [truncated],
            model=self.MODEL_NAME,
            input_type="document",
        )
        return response.embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH_SIZE):
            batch = [self._truncate(text) for text in texts[start : start + self.BATCH_SIZE]]
            if not batch:
                continue
            response = self.voyage.embed(
                batch,
                model=self.MODEL_NAME,
                input_type="document",
            )
            vectors.extend(response.embeddings)
        return vectors

    def upsert_chunks(self, chunks: list[dict[str, Any]], namespace: str) -> None:
        if not chunks:
            return

        texts = [chunk["text"] for chunk in chunks]
        vectors = self.embed_batch(texts)
        payload: list[dict[str, Any]] = []

        for chunk, vector in zip(chunks, vectors, strict=True):
            source_metadata = chunk.get("metadata", {})
            metadata = {
                "file_path": source_metadata.get("file_path", ""),
                "language": source_metadata.get("language", ""),
                "repo": source_metadata.get("repo", ""),
                "chunk_type": source_metadata.get("chunk_type", ""),
                "symbol_name": source_metadata.get("symbol_name", ""),
                "text": chunk["text"],
            }
            if "type" in source_metadata:
                metadata["type"] = source_metadata["type"]

            payload.append(
                {
                    "id": chunk["id"],
                    "values": vector,
                    "metadata": metadata,
                }
            )

        for start in range(0, len(payload), self.UPSERT_BATCH_SIZE):
            self.index.upsert(vectors=payload[start : start + self.UPSERT_BATCH_SIZE], namespace=namespace)

    def query_similar(self, text: str, namespace: str, top_k: int = 5) -> list[dict[str, Any]]:
        vector = self.voyage.embed(
            [self._truncate(text)],
            model=self.MODEL_NAME,
            input_type="query",
        ).embeddings[0]

        response = self.index.query(
            namespace=namespace,
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )

        matches = response.matches if hasattr(response, "matches") else response.get("matches", [])
        results: list[dict[str, Any]] = []
        for match in matches:
            match_id = match.id if hasattr(match, "id") else match.get("id", "")
            score = match.score if hasattr(match, "score") else match.get("score", 0.0)
            metadata = match.metadata if hasattr(match, "metadata") else match.get("metadata", {})
            results.append(
                {
                    "id": match_id,
                    "score": score,
                    "metadata": metadata,
                    "text": metadata.get("text", ""),
                }
            )
        return results

    def delete_namespace(self, namespace: str) -> None:
        """Remove all vectors in a namespace. No-op if the namespace does not exist."""
        try:
            self.index.delete(delete_all=True, namespace=namespace)
        except NotFoundException:
            # Pinecone returns 404 when the namespace has never existed or is already gone.
            pass

    def namespace_exists(self, namespace: str) -> bool:
        stats = self.index.describe_index_stats()
        namespaces = stats.namespaces if hasattr(stats, "namespaces") else stats.get("namespaces", {})
        namespace_stats = namespaces.get(namespace, {})
        if hasattr(namespace_stats, "vector_count"):
            return namespace_stats.vector_count > 0
        return namespace_stats.get("vector_count", 0) > 0


def _describe_stats_dict(stats: Any) -> dict[str, Any]:
    """Normalize Pinecone describe_index_stats() for printing."""
    out: dict[str, Any] = {}
    for key in ("total_vector_count", "dimension", "index_fullness"):
        if hasattr(stats, key):
            out[key] = getattr(stats, key)
        elif isinstance(stats, dict) and key in stats:
            out[key] = stats[key]
    namespaces: dict[str, Any] = {}
    raw_ns = stats.namespaces if hasattr(stats, "namespaces") else stats.get("namespaces", {})
    if raw_ns:
        for ns_name, ns_val in raw_ns.items():
            if hasattr(ns_val, "vector_count"):
                namespaces[ns_name] = {"vector_count": ns_val.vector_count}
            elif isinstance(ns_val, dict):
                namespaces[ns_name] = {"vector_count": ns_val.get("vector_count", 0)}
    out["namespaces"] = namespaces
    return out


def _assert_embedding_sane(store: PineconeStore, vector: list[float], label: str) -> None:
    if len(vector) != store.DIMENSION:
        raise RuntimeError(f"{label}: expected embedding length {store.DIMENSION}, got {len(vector)}")
    if not vector:
        raise RuntimeError(f"{label}: empty embedding")
    if max(abs(x) for x in vector) < 1e-12:
        raise RuntimeError(f"{label}: embedding appears to be all zeros")
    if not all(math.isfinite(x) for x in vector[: min(32, len(vector))]):
        raise RuntimeError(f"{label}: embedding contains non-finite values")


if __name__ == "__main__":
    store = PineconeStore()
    test_namespace = "codelens-smoke-test"

    doc_probe = store.embed_text("def probe(): return True")
    _assert_embedding_sane(store, doc_probe, "Voyage document embed")
    query_probe = store.voyage.embed(
        [store._truncate("find helper functions")],
        model=store.MODEL_NAME,
        input_type="query",
    ).embeddings[0]
    _assert_embedding_sane(store, query_probe, "Voyage query embed")

    test_chunks = [
        {
            "id": "test-1",
            "text": "def add(a, b):\n    return a + b",
            "metadata": {
                "file_path": "math_utils.py",
                "language": "python",
                "repo": "demo-repo",
                "chunk_type": "function",
                "symbol_name": "add",
            },
        },
        {
            "id": "test-2",
            "text": "class Calculator:\n    def multiply(self, a, b):\n        return a * b",
            "metadata": {
                "file_path": "calculator.py",
                "language": "python",
                "repo": "demo-repo",
                "chunk_type": "class",
                "symbol_name": "Calculator",
            },
        },
        {
            "id": "test-3",
            "text": "function sum(values) {\n  return values.reduce((acc, value) => acc + value, 0);\n}",
            "metadata": {
                "file_path": "sum.js",
                "language": "javascript",
                "repo": "demo-repo",
                "chunk_type": "function",
                "symbol_name": "sum",
            },
        },
    ]

    store.delete_namespace(test_namespace)
    store.upsert_chunks(test_chunks, namespace=test_namespace)

    batch_vecs = store.embed_batch([c["text"] for c in test_chunks])
    for i, vec in enumerate(batch_vecs):
        _assert_embedding_sane(store, vec, f"batch embed[{i}]")

    stats = store.index.describe_index_stats()
    print("describe_index_stats (after upsert):", _describe_stats_dict(stats))

    nl_query = (
        "I need a JavaScript utility that adds up all numbers in an array using reduce"
    )
    matches = store.query_similar(nl_query, namespace=test_namespace, top_k=3)
    print(f'query_similar({nl_query!r}) top {len(matches)}:')
    for m in matches:
        print(f"  score={m['score']:.4f} id={m['id']} preview={m['text'][:80]!r}...")

    store.delete_namespace(test_namespace)
    stats_after = store.index.describe_index_stats()
    print("describe_index_stats (after test namespace delete):", _describe_stats_dict(stats_after))
    print("Smoke test OK.")
