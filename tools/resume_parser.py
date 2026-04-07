"""Resume parsing and claim matching utilities for CodeLens."""

from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path
from typing import Any

# Repo root on sys.path so `python tools/resume_parser.py` can import `tools.*`
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import anthropic
from anthropic import APIError, BadRequestError

from tools.pinecone_tool import PineconeStore
from tools.project_env import load_project_env

load_project_env()


class ResumeParser:
    """Extract structured resume and job description data with Claude."""

    MODEL_CANDIDATES = ["claude-sonnet-4-5", "claude-sonnet-4-20250514"]
    MAX_TOKENS = 2000

    def __init__(self) -> None:
        self.client = anthropic.Anthropic()

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
                raise ValueError("Claude did not return valid JSON.")
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError as exc:
                raise ValueError("Claude returned malformed JSON.") from exc

    @staticmethod
    def _collect_text(response: Any) -> str:
        parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "\n".join(parts).strip()

    def _create_message(self, content: list[dict[str, Any]], system_prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for model_name in self.MODEL_CANDIDATES:
            try:
                response = self.client.messages.create(
                    model=model_name,
                    max_tokens=self.MAX_TOKENS,
                    system=system_prompt,
                    messages=[{"role": "user", "content": content}],
                )
                return self._extract_json(self._collect_text(response))
            except BadRequestError as exc:
                last_error = exc
                message = str(exc)
                if "model" in message.lower():
                    continue
                raise
            except APIError as exc:
                last_error = exc
                raise RuntimeError(f"Anthropic API request failed: {exc}") from exc

        raise RuntimeError(
            "No supported Claude Sonnet model was available. Tried: "
            + ", ".join(self.MODEL_CANDIDATES)
        ) from last_error

    def parse_from_pdf(self, pdf_bytes: bytes) -> dict[str, Any]:
        pdf_data = base64.b64encode(pdf_bytes).decode()
        system_prompt = (
            "Extract structured resume data from the supplied PDF. "
            "Return ONLY a JSON object matching this schema exactly:\n"
            f"{self._resume_schema()}\n"
            "Use empty arrays when data is missing, infer experience level conservatively, "
            "and ensure years_experience is an integer."
        )
        content = [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_data,
                },
            },
            {"type": "text", "text": "Extract the resume into the requested JSON schema only."},
        ]
        return self._create_message(content, system_prompt)

    def parse_from_text(self, text: str) -> dict[str, Any]:
        system_prompt = (
            "Extract structured resume data from the supplied text resume. "
            "Return ONLY a JSON object matching this schema exactly:\n"
            f"{self._resume_schema()}\n"
            "Use empty arrays when data is missing, infer experience level conservatively, "
            "and ensure years_experience is an integer."
        )
        content = [{"type": "text", "text": text}]
        return self._create_message(content, system_prompt)

    def parse_job_description(self, jd_text: str) -> dict[str, Any]:
        system_prompt = (
            "Extract hiring requirements from the supplied job description. "
            "Return ONLY a JSON object matching this schema exactly:\n"
            f"{self._job_schema()}\n"
            "Pick a single concise domain such as backend, frontend, fullstack, ML, data, mobile, security, or devops."
        )
        content = [{"type": "text", "text": jd_text}]
        return self._create_message(content, system_prompt)


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
