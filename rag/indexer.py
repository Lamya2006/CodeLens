"""Repository and resume chunk indexing helpers for CodeLens."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Repo root on sys.path so `python rag/indexer.py` can import `tools.*`
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


class CodeIndexer:
    """Convert parsed repository and resume content into Pinecone chunks."""

    WINDOW_LINES = 150
    WINDOW_OVERLAP = 20
    SKIP_EXTENSIONS = {
        ".lock",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".ico",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".mp3",
        ".mp4",
        ".mov",
        ".webm",
        ".bin",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
    }
    CONFIG_FILENAMES = {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "Pipfile.lock",
        "Cargo.lock",
        "composer.lock",
        ".env",
        ".gitignore",
    }

    def index_repo_files(
        self,
        files: list[dict[str, Any]],
        candidate_username: str,
        pinecone_store: PineconeStore,
    ) -> int:
        namespace = f"candidate-{candidate_username}"
        chunks: list[dict[str, Any]] = []

        for file_data in files:
            if self._should_skip_file(file_data):
                continue

            file_chunks = self._build_symbol_chunks(file_data, candidate_username)
            if not file_chunks:
                file_chunks = self._build_window_chunks(file_data, candidate_username)
            chunks.extend(file_chunks)

        pinecone_store.upsert_chunks(chunks, namespace=namespace)
        return len(chunks)

    def index_resume(
        self,
        resume_data: dict[str, Any],
        resume_text: str,
        candidate_username: str,
        pinecone_store: PineconeStore,
    ) -> None:
        namespace = f"resume-{candidate_username}"
        chunks: list[dict[str, Any]] = []

        for index, project in enumerate(resume_data.get("projects", []), start=1):
            project_text = "\n".join(
                [
                    f"Project: {project.get('name', '')}",
                    f"Description: {project.get('description', '')}",
                    "Technologies: " + ", ".join(project.get("technologies", [])),
                    "Claimed features: " + ", ".join(project.get("claimed_features", [])),
                ]
            ).strip()
            chunks.append(
                {
                    "id": f"resume-project-{candidate_username}-{index}",
                    "text": project_text,
                    "metadata": {
                        "file_path": "resume/projects",
                        "language": "text",
                        "repo": f"resume-{candidate_username}",
                        "candidate": candidate_username,
                        "chunk_type": "project",
                        "symbol_name": project.get("name", f"project-{index}"),
                        "line_start": 1,
                    },
                }
            )

        skills_text = "Skills: " + ", ".join(resume_data.get("skills", []))
        chunks.append(
            {
                "id": f"resume-skills-{candidate_username}",
                "text": skills_text,
                "metadata": {
                    "file_path": "resume/skills",
                    "language": "text",
                    "repo": f"resume-{candidate_username}",
                    "candidate": candidate_username,
                    "chunk_type": "skills",
                    "symbol_name": "skills",
                    "line_start": 1,
                },
            }
        )

        experience_text = "\n".join(
            [
                f"Experience level: {resume_data.get('experience_level', '')}",
                f"Years experience: {resume_data.get('years_experience', 0)}",
                "Education: "
                + "; ".join(
                    ", ".join(
                        filter(
                            None,
                            [
                                education.get("degree", ""),
                                education.get("field", ""),
                                education.get("institution", ""),
                            ],
                        )
                    )
                    for education in resume_data.get("education", [])
                ),
                f"Resume summary: {resume_text[:4000]}",
            ]
        ).strip()
        chunks.append(
            {
                "id": f"resume-summary-{candidate_username}",
                "text": experience_text,
                "metadata": {
                    "file_path": "resume/summary",
                    "language": "text",
                    "repo": f"resume-{candidate_username}",
                    "candidate": candidate_username,
                    "chunk_type": "experience",
                    "symbol_name": "experience-summary",
                    "line_start": 1,
                },
            }
        )

        pinecone_store.upsert_chunks(chunks, namespace=namespace)

    def _should_skip_file(self, file_data: dict[str, Any]) -> bool:
        language = file_data.get("language")
        file_path = str(file_data.get("file_path") or file_data.get("path") or "")
        if not language:
            return True

        path_obj = Path(file_path.lower())
        if path_obj.name in self.CONFIG_FILENAMES:
            return True
        if path_obj.suffix in self.SKIP_EXTENSIONS:
            return True
        if any(part in {"node_modules", ".git", "dist", "build", "__pycache__"} for part in path_obj.parts):
            return True
        return False

    def _build_symbol_chunks(
        self, file_data: dict[str, Any], candidate_username: str
    ) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        symbols = file_data.get("symbols") or file_data.get("parsed_symbols") or []
        if isinstance(symbols, dict):
            symbols = symbols.get("items", [])

        for index, symbol in enumerate(symbols, start=1):
            if not isinstance(symbol, dict):
                continue
            symbol_type = str(symbol.get("type", "")).lower()
            if symbol_type not in {"function", "class", "method"}:
                continue

            code_text = (
                symbol.get("code")
                or symbol.get("text")
                or symbol.get("content")
                or symbol.get("body")
                or ""
            )
            if not str(code_text).strip():
                continue

            symbol_name = symbol.get("name") or f"{symbol_type}-{index}"
            line_start = int(symbol.get("line_start") or symbol.get("start_line") or 1)
            chunks.append(
                {
                    "id": f"{candidate_username}-{self._safe_id(file_data, symbol_name, index)}",
                    "text": str(code_text),
                    "metadata": {
                        "file_path": file_data.get("file_path") or file_data.get("path") or "",
                        "language": file_data.get("language") or "",
                        "repo": file_data.get("repo") or candidate_username,
                        "candidate": candidate_username,
                        "chunk_type": symbol_type,
                        "symbol_name": symbol_name,
                        "line_start": line_start,
                    },
                }
            )
        return chunks

    def _build_window_chunks(
        self, file_data: dict[str, Any], candidate_username: str
    ) -> list[dict[str, Any]]:
        content = str(file_data.get("content") or file_data.get("text") or "")
        if not content.strip():
            return []

        lines = content.splitlines()
        if not lines:
            return []

        step = self.WINDOW_LINES - self.WINDOW_OVERLAP
        chunks: list[dict[str, Any]] = []
        for start in range(0, len(lines), step):
            end = min(start + self.WINDOW_LINES, len(lines))
            window_text = "\n".join(lines[start:end]).strip()
            if not window_text:
                continue
            chunk_index = len(chunks) + 1
            chunks.append(
                {
                    "id": f"{candidate_username}-{self._safe_id(file_data, 'window', chunk_index)}",
                    "text": window_text,
                    "metadata": {
                        "file_path": file_data.get("file_path") or file_data.get("path") or "",
                        "language": file_data.get("language") or "",
                        "repo": file_data.get("repo") or candidate_username,
                        "candidate": candidate_username,
                        "chunk_type": "file",
                        "symbol_name": f"window-{chunk_index}",
                        "line_start": start + 1,
                    },
                }
            )
            if end == len(lines):
                break
        return chunks

    @staticmethod
    def _safe_id(file_data: dict[str, Any], label: str, index: int) -> str:
        file_path = str(file_data.get("file_path") or file_data.get("path") or "unknown")
        safe_path = file_path.replace("/", "-").replace(" ", "-")
        safe_label = str(label).replace("/", "-").replace(" ", "-")
        return f"{safe_path}-{safe_label}-{index}"
