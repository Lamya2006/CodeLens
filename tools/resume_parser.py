"""Resume parsing and claim matching utilities for CodeLens."""

from __future__ import annotations

from io import BytesIO
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Repo root on sys.path so `python tools/resume_parser.py` can import `tools.*`
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import requests
from pypdf import PdfReader

from tools.pinecone_tool import PineconeStore
from tools.project_env import load_project_env

load_project_env()


class ResumeParser:
    """Extract structured resume and job description data via OpenRouter."""

    MAX_TOKENS = 2000

    @staticmethod
    def _openrouter_model_ids() -> list[str]:
        """API model ids (no `openrouter/` prefix). Honors OPENROUTER_MODEL, then fallbacks."""
        raw = os.getenv("OPENROUTER_MODEL", "openrouter/google/gemma-4-26b-a4b-it").strip()
        if raw.startswith("openrouter/"):
            raw = raw[len("openrouter/") :]
        primary = raw or "google/gemma-4-26b-a4b-it"
        fallbacks = [
            "google/gemma-4-26b-a4b-it",
            "anthropic/claude-sonnet-4",
            "anthropic/claude-3.5-sonnet",
            "claude-sonnet-4-5",
        ]
        ordered: list[str] = []
        for m in [primary, *fallbacks]:
            if m and m not in ordered:
                ordered.append(m)
        return ordered

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
        self.app_name = os.getenv("OPENROUTER_APP_NAME", "CodeLens")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for ResumeParser.")

    @staticmethod
    def _resume_schema() -> str:
        return json.dumps(
            {
                "skills": ["Python", "React"],
                "projects": [
                    {
                        "name": "string",
                        "description": "string",
                        "technologies": ["string"],
                        "claimed_features": ["string"],
                    }
                ],
                "experience_level": "student",
                "years_experience": 0,
                "education": [{"degree": "string", "field": "string", "institution": "string"}],
                "github_urls": ["https://github.com/example"],
            },
            indent=2,
        )

    @staticmethod
    def _job_schema() -> str:
        return json.dumps(
            {
                "required_skills": ["string"],
                "preferred_skills": ["string"],
                "seniority_level": "string",
                "key_responsibilities": ["string"],
                "tech_stack": ["string"],
                "domain": "backend",
            },
            indent=2,
        )

    @staticmethod
    def _strip_fences(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip()

    @classmethod
    def _extract_json(cls, text: str) -> dict[str, Any]:
        cleaned = cls._strip_fences(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("Model did not return valid JSON.")
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError as exc:
                raise ValueError("Model returned malformed JSON.") from exc

    def _create_message(self, user_text: str, system_prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for model_name in self._openrouter_model_ids():
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": self.site_url,
                        "X-Title": self.app_name,
                    },
                    json={
                        "model": model_name,
                        "max_tokens": self.MAX_TOKENS,
                        "temperature": 0,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_text},
                        ],
                    },
                    timeout=90,
                )
                if response.status_code >= 400:
                    message = response.text.lower()
                    if response.status_code in {400, 404} and "model" in message:
                        last_error = RuntimeError(response.text)
                        continue
                    response.raise_for_status()
                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError("OpenRouter returned no choices.")
                content = choices[0].get("message", {}).get("content", "")
                if isinstance(content, list):
                    text_parts = [
                        str(part.get("text", ""))
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    ]
                    content = "\n".join(text_parts).strip()
                return self._extract_json(str(content))
            except requests.HTTPError as exc:
                last_error = exc
                raise RuntimeError(f"OpenRouter API request failed: {exc}") from exc
            except requests.RequestException as exc:
                last_error = exc
                raise RuntimeError(f"OpenRouter network request failed: {exc}") from exc
            except ValueError as exc:
                message = str(exc).lower()
                if "model" in message:
                    continue
                raise

        raise RuntimeError(
            "No supported OpenRouter model was available. Tried: "
            + ", ".join(self._openrouter_model_ids())
        ) from last_error

    @staticmethod
    def _pdf_to_text(pdf_bytes: bytes) -> str:
        reader = PdfReader(BytesIO(pdf_bytes))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        text = "\n\n".join(page for page in pages if page)
        if not text.strip():
            raise ValueError("Unable to extract text from PDF.")
        return text

    def parse_from_pdf(self, pdf_bytes: bytes) -> dict[str, Any]:
        extracted_text = self._pdf_to_text(pdf_bytes)
        system_prompt = (
            "Extract structured resume data from the supplied PDF. "
            "Return ONLY a JSON object matching this schema exactly:\n"
            f"{self._resume_schema()}\n"
            "Use empty arrays when data is missing, infer experience level conservatively, "
            "and ensure years_experience is an integer."
        )
        user_text = (
            "Extract the resume into the requested JSON schema only.\n\n"
            "Resume text extracted from PDF:\n"
            f"{extracted_text}"
        )
        return self._create_message(user_text, system_prompt)

    def parse_from_text(self, text: str) -> dict[str, Any]:
        system_prompt = (
            "Extract structured resume data from the supplied text resume. "
            "Return ONLY a JSON object matching this schema exactly:\n"
            f"{self._resume_schema()}\n"
            "Use empty arrays when data is missing, infer experience level conservatively, "
            "and ensure years_experience is an integer."
        )
        return self._create_message(text, system_prompt)

    def parse_job_description(self, jd_text: str) -> dict[str, Any]:
        system_prompt = (
            "Extract hiring requirements from the supplied job description. "
            "Return ONLY a JSON object matching this schema exactly:\n"
            f"{self._job_schema()}\n"
            "Pick a single concise domain such as backend, frontend, fullstack, ML, data, mobile, security, or devops."
        )
        return self._create_message(jd_text, system_prompt)


class SkillMatcher:
    """Match resume claims against candidate code embeddings."""

    COMMON_TECH_TERMS = [
        "React",
        "Flask",
        "PostgreSQL",
        "Docker",
        "Redis",
        "JWT",
        "REST",
        "GraphQL",
        "TypeScript",
        "MongoDB",
        "FastAPI",
        "Django",
        "Node.js",
        "Express",
        "Next.js",
        "Vue",
        "Angular",
        "Svelte",
        "Python",
        "JavaScript",
        "Java",
        "C++",
        "Go",
        "Rust",
        "AWS",
        "GCP",
        "Azure",
        "Kubernetes",
        "Terraform",
        "CI/CD",
        "GitHub Actions",
        "Celery",
        "RabbitMQ",
        "Kafka",
        "SQLAlchemy",
        "Pandas",
        "NumPy",
        "PyTorch",
        "TensorFlow",
        "scikit-learn",
        "OAuth",
        "WebSockets",
        "Tailwind",
        "Bootstrap",
        "MySQL",
        "SQLite",
        "Elasticsearch",
        "Nginx",
        "Linux",
        "Microservices",
    ]

    def __init__(self, pinecone_store: PineconeStore) -> None:
        self.pinecone_store = pinecone_store

    @staticmethod
    def _verdict(score: float) -> str:
        if score > 0.72:
            return "confirmed"
        if score >= 0.40:
            return "partial"
        return "unsubstantiated"

    @staticmethod
    def _snippet_texts(matches: list[dict[str, Any]], limit: int = 2) -> list[str]:
        snippets: list[str] = []
        for match in matches[:limit]:
            text = (match.get("text") or "").strip()
            if text:
                snippets.append(text[:500])
        return snippets

    def match_skills_to_code(self, claims: list[str], candidate_namespace: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for skill in claims:
            query = f"code using {skill}: implementation example"
            matches = self.pinecone_store.query_similar(query, namespace=candidate_namespace, top_k=2)
            top_score = float(matches[0]["score"]) if matches else 0.0
            results.append(
                {
                    "skill": skill,
                    "match_score": top_score,
                    "supporting_snippets": self._snippet_texts(matches, limit=2),
                    "verdict": self._verdict(top_score),
                }
            )
        return results

    def match_project_claims(self, project: dict[str, Any], candidate_namespace: str) -> dict[str, Any]:
        features = project.get("claimed_features", []) or []
        technologies = project.get("technologies", []) or []
        evidence: list[dict[str, Any]] = []
        feature_scores: list[float] = []
        missing_features: list[str] = []

        for feature in features:
            query = f"implementation of {feature} in project code"
            matches = self.pinecone_store.query_similar(query, namespace=candidate_namespace, top_k=1)
            top_match = matches[0] if matches else None
            score = float(top_match["score"]) if top_match else 0.0
            found = score >= 0.45
            feature_scores.append(score)
            if not found:
                missing_features.append(feature)
            evidence.append(
                {
                    "feature": feature,
                    "found": found,
                    "evidence_snippet": (top_match.get("text", "")[:500] if top_match else None) or None,
                    "quality_note": self._quality_note(feature, score, top_match),
                }
            )

        technology_scores: list[float] = []
        for technology in technologies:
            matches = self.pinecone_store.query_similar(
                f"code using {technology}: implementation example",
                namespace=candidate_namespace,
                top_k=1,
            )
            technology_scores.append(float(matches[0]["score"]) if matches else 0.0)

        all_scores = feature_scores + technology_scores
        overall_match = round(sum(all_scores) / len(all_scores), 3) if all_scores else 0.0

        return {
            "project_name": project.get("name", ""),
            "overall_match": overall_match,
            "feature_evidence": evidence,
            "missing_features": missing_features,
        }

    @staticmethod
    def _quality_note(feature: str, score: float, match: dict[str, Any] | None) -> str:
        if not match or score < 0.40:
            return f"No strong code evidence found for {feature}."

        metadata = match.get("metadata", {})
        chunk_type = metadata.get("chunk_type") or "code chunk"
        if score > 0.72:
            return f"{feature} appears clearly implemented in a matching {chunk_type}."
        return f"{feature} may be partially implemented, but the evidence is not conclusive."

    def find_undeclared_skills(self, declared_skills: list[str], candidate_namespace: str) -> list[str]:
        declared_normalized = {skill.strip().lower() for skill in declared_skills if skill.strip()}
        discovered: list[str] = []

        for term in self.COMMON_TECH_TERMS:
            matches = self.pinecone_store.query_similar(
                f"code using {term}: implementation example",
                namespace=candidate_namespace,
                top_k=1,
            )
            score = float(matches[0]["score"]) if matches else 0.0
            if score > 0.65 and term.lower() not in declared_normalized:
                discovered.append(term)

        return discovered


if __name__ == "__main__":
    fake_resume = """
    Jane Doe
    GitHub: https://github.com/janedoe

    Computer Science student with 2 years of internship experience.
    Skills: Python, Flask, React, PostgreSQL, Docker.

    Projects:
    1. Campus Events Platform
       Built a Flask app with PostgreSQL and Docker deployment.
       Implemented authentication, event CRUD, and admin analytics dashboard.

    2. Portfolio Site
       Built a React frontend with reusable UI components and responsive layouts.

    Education:
    B.S. in Computer Science, State University
    """

    parser = ResumeParser()
    print("Parsed Resume:")
    print(json.dumps(parser.parse_from_text(fake_resume), indent=2))
