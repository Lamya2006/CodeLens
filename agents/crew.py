"""CrewAI multi-agent analysis pipeline for CodeLens."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Repo root on sys.path so `python agents/crew.py` can import `tools.*`
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from crewai import Agent, Crew, LLM, Process, Task
from pydantic import BaseModel

from tools.project_env import load_project_env

load_project_env()

llm = LLM(
    model="claude-sonnet-4-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=4096,
)


class SuspiciousPattern(BaseModel):
    pattern: str
    evidence: str


class StrengthObservation(BaseModel):
    observation: str
    location: str


class ConcernItem(BaseModel):
    issue: str
    location: str
    severity: str


class BugItem(BaseModel):
    description: str
    file: str
    line_hint: str


class AiEvidenceSignal(BaseModel):
    signal: str
    location: str
    note: str


class HallucinationFlag(BaseModel):
    description: str
    file: str


class ProjectVerdict(BaseModel):
    project_name: str
    claimed: str
    reality: str
    match_quality: str
    specific_notes: str


class SkillVerdict(BaseModel):
    skill: str
    verdict: str
    evidence: str


class CommitBehaviorOutput(BaseModel):
    commit_health_score: int
    velocity_assessment: str
    suspicious_patterns: list[SuspiciousPattern]
    positive_signals: list[str]
    summary: str


class CodeQualityOutput(BaseModel):
    quality_score: int
    readability_score: int
    strengths: list[StrengthObservation]
    concerns: list[ConcernItem]
    bugs_or_errors: list[BugItem]
    summary: str


class BaselineSimilarityOutput(BaseModel):
    human: float
    ai: float


class AIUsageOutput(BaseModel):
    ai_usage_score: int
    ai_evidence_signals: list[AiEvidenceSignal]
    hallucination_flags: list[HallucinationFlag]
    vibe_coding_flags: list[str]
    good_ai_usage: list[str]
    baseline_similarity: BaselineSimilarityOutput
    summary: str


class ResumeMatchOutput(BaseModel):
    resume_match_score: int
    project_verdicts: list[ProjectVerdict]
    skill_verdicts: list[SkillVerdict]
    undeclared_skills: list[str]
    inflation_flags: list[str]
    summary: str


class JudgeOutput(BaseModel):
    overall_quality_score: int
    ai_usage_score: int
    commit_health_score: int
    resume_match_score: int | None
    job_fit_score: int | None
    strengths: list[str]
    concerns: list[str]
    skill_map: dict[str, str]
    vibe_coding_flags: list[str]
    ai_usage_summary: str
    bugs_found: list[str]
    resume_inflation_flags: list[str]
    job_fit_analysis: str | None
    company_style_fit: int | None
    recommendation: str
    recommendation_reasoning: str
    summary: str
    disclaimer: str


class CodeLensCrew:
    """Run the multi-agent CodeLens evaluation workflow."""

    def __init__(self, analysis_data: dict[str, Any]) -> None:
        self.analysis_data = analysis_data
        self.analysis_json = json.dumps(analysis_data, indent=2, default=str)

        self.commit_behavior_agent = Agent(
            role="Senior Engineering Manager",
            goal="Analyze commit behavior and development patterns.",
            backstory=(
                "You have reviewed thousands of candidates and can read development patterns "
                "like a fingerprint. You are objective — messy commits are not automatically bad, "
                "and clean commits are not automatically AI."
            ),
            llm=llm,
            allow_delegation=False,
            verbose=False,
        )
        self.code_quality_agent = Agent(
            role="Principal Software Engineer",
            goal="Assess genuine code quality and engineering judgment.",
            backstory=(
                "You care about working software. You evaluate readability, structure, error "
                "handling appropriateness, naming clarity, and whether the code solves the problem "
                "elegantly. You have seen great AI-assisted code and terrible human code."
            ),
            llm=llm,
            allow_delegation=False,
            verbose=False,
        )
        self.ai_usage_agent = Agent(
            role="AI Engineering Specialist",
            goal="Assess how AI was used and whether that usage shows real understanding.",
            backstory=(
                "You believe AI is a tool. Using it well is a skill. Using it blindly is a red "
                "flag. You look for evidence of understanding versus evidence of cargo-cult copying."
            ),
            llm=llm,
            allow_delegation=False,
            verbose=False,
        )
        self.resume_match_agent = Agent(
            role="Technical Recruiter with Engineering Background",
            goal="Verify resume claims against actual code evidence.",
            backstory=(
                "You have seen every type of resume inflation. You look for specific implementations, "
                "not just keyword presence. If someone claims a feature, you check whether the code "
                "actually implements it in a credible way."
            ),
            llm=llm,
            allow_delegation=False,
            verbose=False,
        )
        self.judge_agent = Agent(
            role="VP of Engineering making a hiring decision",
            goal="Synthesize all prior reports into a fair, direct, evidence-based verdict.",
            backstory=(
                "You make practical hiring calls. You are specific, fair, and comfortable acknowledging "
                "uncertainty. You optimize for signal over theatrics."
            ),
            llm=llm,
            allow_delegation=False,
            verbose=False,
        )

    def run(self) -> dict[str, Any]:
        commit_task = Task(
            description=self._commit_task_description(),
            expected_output="A JSON object exactly matching the requested commit behavior schema.",
            agent=self.commit_behavior_agent,
            output_json=CommitBehaviorOutput,
        )
        code_quality_task = Task(
            description=self._code_quality_task_description(),
            expected_output="A JSON object exactly matching the requested code quality schema.",
            agent=self.code_quality_agent,
            context=[commit_task],
            output_json=CodeQualityOutput,
        )
        ai_usage_task = Task(
            description=self._ai_usage_task_description(),
            expected_output="A JSON object exactly matching the requested AI usage schema.",
            agent=self.ai_usage_agent,
            context=[commit_task, code_quality_task],
            output_json=AIUsageOutput,
        )

        tasks: list[Task] = [commit_task, code_quality_task, ai_usage_task]
        resume_task: Task | None = None

        if self.analysis_data.get("resume_data") is not None:
            resume_task = Task(
                description=self._resume_task_description(),
                expected_output="A JSON object exactly matching the requested resume match schema.",
                agent=self.resume_match_agent,
                context=[commit_task, code_quality_task, ai_usage_task],
                output_json=ResumeMatchOutput,
            )
            tasks.append(resume_task)

        judge_context = [commit_task, code_quality_task, ai_usage_task]
        if resume_task is not None:
            judge_context.append(resume_task)

        judge_task = Task(
            description=self._judge_task_description(resume_enabled=resume_task is not None),
            expected_output="A JSON object exactly matching the requested final verdict schema.",
            agent=self.judge_agent,
            context=judge_context,
            output_json=JudgeOutput,
        )
        tasks.append(judge_task)

        agents = [
            self.commit_behavior_agent,
            self.code_quality_agent,
            self.ai_usage_agent,
            self.judge_agent,
        ]
        if resume_task is not None:
            agents.insert(3, self.resume_match_agent)

        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()

        if getattr(result, "json_dict", None):
            return result.json_dict

        if hasattr(result, "to_dict"):
            result_dict = result.to_dict()
            if isinstance(result_dict, dict):
                return result_dict

        raw_output = getattr(result, "raw", None) or str(result)
        return json.loads(raw_output)

    def _commit_task_description(self) -> str:
        return (
            "Analyze the candidate's commit behavior using the provided repository analysis data.\n\n"
            "Focus inputs:\n"
            "- commit_patterns\n"
            "- commits\n"
            "- repo_metadata\n\n"
            "Assess:\n"
            "- Development velocity and consistency\n"
            "- Whether commit sizes suggest copy-paste drops versus iterative development\n"
            "- Commit message quality and evolution over time\n"
            "- Branch usage and collaboration signals\n"
            "- Specific suspicious patterns, each with the exact triggering data\n\n"
            "Be balanced: messy commits are not automatically bad and clean commits are not automatically AI.\n"
            "Return only JSON matching the schema.\n\n"
            f"Analysis data:\n{self.analysis_json}"
        )

    def _code_quality_task_description(self) -> str:
        return (
            "Assess actual code quality using the repository file data and GitNexus knowledge graph.\n\n"
            "Focus inputs:\n"
            "- files\n"
            "- knowledge_graph\n\n"
            "Assess:\n"
            "- Readability and naming conventions\n"
            "- Whether abstractions are meaningful or over-engineered\n"
            "- Error handling: appropriate, missing, or excessive\n"
            "- Organization and modularity\n"
            "- Obvious bugs, anti-patterns, or dangerous code\n"
            "- Cite specific files or functions as examples when possible\n\n"
            "Use prior task context if it adds helpful interpretation, but judge code quality on the code itself.\n"
            "Return only JSON matching the schema.\n\n"
            f"Analysis data:\n{self.analysis_json}"
        )

    def _ai_usage_task_description(self) -> str:
        baseline = self.analysis_data.get("baseline_comparison", {})
        return (
            "Assess HOW WELL AI appears to have been used in this codebase, not whether AI was used at all.\n\n"
            "Focus inputs:\n"
            "- files\n"
            "- baseline_comparison\n"
            "- commit_patterns\n\n"
            "Assess:\n"
            "- Stylistic signals of AI generation without making unsupported accusations\n"
            "- Hallucinated imports, APIs, or calls to non-existent interfaces\n"
            "- Boilerplate that appears unadapted\n"
            "- Good AI usage where scaffolding was thoughtfully customized\n"
            "- Vibe-coded sections versus intentional AI collaboration\n\n"
            "Include the provided baseline similarity numbers in your reasoning.\n"
            "Return only JSON matching the schema.\n\n"
            f"Baseline comparison snapshot:\n{json.dumps(baseline, indent=2, default=str)}\n\n"
            f"Analysis data:\n{self.analysis_json}"
        )

    def _resume_task_description(self) -> str:
        return (
            "Verify resume claims against code evidence.\n\n"
            "Focus inputs:\n"
            "- resume_data\n"
            "- skill_matches\n"
            "- project_matches\n\n"
            "Assess:\n"
            "- For each project claim, whether the claimed implementation is actually supported\n"
            "- For each skill claim, the strength of code evidence\n"
            "- Skills clearly evident in code but not declared on the resume\n"
            "- Specific inflated or misleading descriptions\n\n"
            "Be specific and evidence-based rather than punitive.\n"
            "Return only JSON matching the schema.\n\n"
            f"Analysis data:\n{self.analysis_json}"
        )

    def _judge_task_description(self, resume_enabled: bool) -> str:
        resume_instruction = (
            "Resume task output is available in context. Use it to set resume_match_score, resume_inflation_flags, "
            "skill_map, and recruiter-facing guidance.\n"
            if resume_enabled
            else "No resume_data was provided. Set resume_match_score to null and resume_inflation_flags to an empty list.\n"
        )
        job_instruction = (
            "If job_description is missing, set job_fit_score and job_fit_analysis to null.\n"
            if self.analysis_data.get("job_description") is None
            else "Use the provided job_description to estimate job_fit_score and explain fit gaps or strengths.\n"
        )
        company_style_instruction = (
            "If company_style_summary is missing, set company_style_fit to null.\n"
            if self.analysis_data.get("company_style_summary") is None
            else "Use company_style_summary to estimate company_style_fit from 0-100.\n"
        )
        return (
            "Synthesize all previous agent outputs into a final hiring verdict a recruiter can act on.\n\n"
            "Requirements:\n"
            "- Be direct, specific, and fair\n"
            "- Acknowledge uncertainty where it exists\n"
            "- Keep strengths and concerns specific and evidenced\n"
            "- Cap strengths and concerns at 5 items each\n"
            "- Recommendation must be one of: strong_hire, hire, maybe, pass\n"
            "- Disclaimer must exactly be: "
            "\"CodeLens provides probabilistic signals to assist human judgment. All findings should be verified in a technical interview.\"\n\n"
            f"{resume_instruction}"
            f"{job_instruction}"
            f"{company_style_instruction}"
            "For skill_map, map skills to confirmed, partial, or not_found based on the available evidence.\n"
            "Return only JSON matching the schema.\n\n"
            f"Analysis data:\n{self.analysis_json}"
        )
