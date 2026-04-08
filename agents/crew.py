"""CrewAI multi-agent analysis pipeline for CodeLens."""

from __future__ import annotations

import copy
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


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def _json_utf8_byte_length(obj: Any) -> int:
    return len(json.dumps(obj, indent=2, default=str).encode("utf-8"))


def _truncate_str(s: str, max_chars: int, label: str = "") -> str:
    if len(s) <= max_chars:
        return s
    suffix = f"\n\n[... truncated ({label}) ...]"
    return s[: max(0, max_chars - len(suffix))] + suffix


def _compact_files_for_llm(
    files: list[Any],
    max_files: int,
    max_content: int,
) -> tuple[list[Any], str | None]:
    if not isinstance(files, list) or not files:
        return files, None
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for i, f in enumerate(files):
        if not isinstance(f, dict):
            continue
        lc = int(f.get("line_count") or 0)
        scored.append((lc, i, f))
    if not scored:
        return files, None
    scored.sort(key=lambda x: -x[0])
    picked = [x[2] for x in scored[:max_files]]
    note = None
    if len(files) > max_files:
        note = f"files reduced from {len(files)} to {max_files} (by line_count)."
    for f in picked:
        for key in ("content", "text", "raw"):
            v = f.get(key)
            if isinstance(v, str) and len(v) > max_content:
                f[key] = _truncate_str(v, max_content, key)
        syms = f.get("symbols")
        if isinstance(syms, list):
            slim: list[Any] = []
            for s in syms[:40]:
                if not isinstance(s, dict):
                    slim.append(s)
                    continue
                s2 = dict(s)
                code = s2.get("code")
                if isinstance(code, str) and len(code) > 3500:
                    s2["code"] = _truncate_str(code, 3500, "symbol code")
                slim.append(s2)
            f["symbols"] = slim
        elif isinstance(syms, dict):
            items = syms.get("items")
            if isinstance(items, list):
                syms = dict(syms)
                syms["items"] = items[:40]
                f["symbols"] = syms
    return picked, note


def _compact_knowledge_graph(kg: dict[str, Any], list_cap: int) -> dict[str, Any]:
    out = dict(kg)
    for list_key in (
        "file_tree",
        "function_list",
        "class_list",
        "import_relationships",
        "call_chains",
    ):
        v = out.get(list_key)
        if isinstance(v, list) and len(v) > list_cap:
            out[list_key] = v[:list_cap]
    if _json_utf8_byte_length(out) > 400_000:
        return {
            "fallback_mode": kg.get("fallback_mode"),
            "warning": kg.get("warning"),
            "file_tree_sample": (kg.get("file_tree") or [])[: min(120, list_cap)],
            "function_list_sample": (kg.get("function_list") or [])[: min(80, list_cap)],
            "class_list_sample": (kg.get("class_list") or [])[: min(80, list_cap)],
            "note": "knowledge_graph lists truncated for LLM payload size",
        }
    return out


def _truncate_commits(commits: Any, max_msg: int) -> None:
    if not isinstance(commits, list):
        return
    for c in commits:
        if not isinstance(c, dict):
            continue
        msg = c.get("message")
        if isinstance(msg, str) and len(msg) > max_msg:
            c["message"] = _truncate_str(msg, max_msg, "commit message")


def _truncate_resume_and_matches(data: dict[str, Any], max_resume: int, max_blob: int) -> None:
    rd = data.get("resume_data")
    if isinstance(rd, dict):
        for k in ("raw_text", "text", "full_text"):
            v = rd.get(k)
            if isinstance(v, str) and len(v) > max_resume:
                rd[k] = _truncate_str(v, max_resume, k)
    for key in ("skill_matches", "project_matches"):
        block = data.get(key)
        if not isinstance(block, list):
            continue
        for item in block:
            if not isinstance(item, dict):
                continue
            for ik, iv in list(item.items()):
                if isinstance(iv, str) and len(iv) > max_blob:
                    item[ik] = _truncate_str(iv, max_blob, ik)


def _crew_mode() -> str:
    m = (os.getenv("CREWAI_MODE") or "efficient").strip().lower()
    if m == "full":
        return "full"
    return "efficient"


def _crew_int_env(name: str, *, full_default: str, efficient_default: str) -> int:
    """Use tighter defaults in efficient mode unless the user set the variable explicitly."""
    raw = os.getenv(name)
    if raw is not None and str(raw).strip() != "":
        return int(raw)
    return int(efficient_default if _crew_mode() == "efficient" else full_default)


def compact_analysis_data_for_llm(data: dict[str, Any]) -> dict[str, Any]:
    """Shrink repo analysis dict so CrewAI prompts stay under provider input limits (~8MB)."""
    budget = _crew_int_env("CREW_MAX_ANALYSIS_JSON_BYTES", full_default="5500000", efficient_default="4200000")
    max_files = _crew_int_env("CREW_MAX_FILES", full_default="48", efficient_default="28")
    max_file_chars = _crew_int_env("CREW_MAX_FILE_CONTENT_CHARS", full_default="9500", efficient_default="6500")
    max_commit_msg = _crew_int_env("CREW_MAX_COMMIT_MESSAGE_CHARS", full_default="1500", efficient_default="1200")
    kg_list_cap = _crew_int_env("CREW_MAX_KG_LIST_ITEMS", full_default="220", efficient_default="140")
    max_resume = _crew_int_env("CREW_MAX_RESUME_TEXT_CHARS", full_default="48000", efficient_default="32000")
    max_match_blob = _crew_int_env("CREW_MAX_MATCH_FIELD_CHARS", full_default="12000", efficient_default="8000")

    last_compact: dict[str, Any] | None = None
    for attempt in range(8):
        note_parts: list[str] = []
        out = copy.deepcopy(data)
        files = out.get("files")
        if isinstance(files, list):
            compacted, fnote = _compact_files_for_llm(files, max_files, max_file_chars)
            out["files"] = compacted
            if fnote:
                note_parts.append(fnote)
        kg = out.get("knowledge_graph")
        if isinstance(kg, dict):
            out["knowledge_graph"] = _compact_knowledge_graph(kg, kg_list_cap)
        _truncate_commits(out.get("commits"), max_commit_msg)
        _truncate_resume_and_matches(out, max_resume, max_match_blob)
        jd = out.get("job_description")
        if isinstance(jd, dict):
            dumped = json.dumps(jd, default=str)
            if len(dumped) > 80_000:
                out["job_description"] = {"note": "job description truncated", "preview": dumped[:80_000]}
        css = out.get("company_style_summary")
        if isinstance(css, str) and len(css) > 100_000:
            out["company_style_summary"] = _truncate_str(css, 100_000, "company_style_summary")

        last_compact = out
        n = _json_utf8_byte_length(out)
        if n <= budget:
            if note_parts or attempt > 0:
                out["_llm_payload_note"] = " ".join(note_parts).strip()
                if attempt > 0:
                    out["_llm_payload_note"] = (
                        (out.get("_llm_payload_note") or "") + f" Iteration {attempt + 1} shrink."
                    ).strip()
            return out

        max_files = max(6, max_files // 2)
        max_file_chars = max(2500, max_file_chars // 2)
        kg_list_cap = max(40, kg_list_cap // 2)
        max_commit_msg = max(400, max_commit_msg // 2)

    if last_compact is not None:
        last_compact["_llm_payload_note"] = (
            (last_compact.get("_llm_payload_note") or "")
            + " Payload still large after max shrink iterations; consider lowering CREW_MAX_* env vars."
        ).strip()
        return last_compact
    return {}


llm = LLM(
    model=os.getenv("OPENROUTER_MODEL", "openrouter/google/gemma-4-26b-a4b-it"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    # Default 2048 reduces cost vs 4096; raise OPENROUTER_MAX_TOKENS if JSON gets truncated.
    max_tokens=_env_int("OPENROUTER_MAX_TOKENS", 2048),
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


class EfficientAnalystOutput(BaseModel):
    """Single-call bundle for CREWAI_MODE=efficient (fewer API round-trips)."""

    commit_behavior: CommitBehaviorOutput
    code_quality: CodeQualityOutput
    ai_usage: AIUsageOutput
    resume_match: ResumeMatchOutput | None = None


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
        self._crew_mode_flag = _crew_mode()
        self.analysis_data = compact_analysis_data_for_llm(analysis_data)
        self.analysis_json = json.dumps(self.analysis_data, indent=2, default=str)

        self.unified_analyst_agent = Agent(
            role="Principal Staff Engineer",
            goal="Deliver commit, code quality, AI-usage, and optional resume assessments in one structured pass.",
            backstory=(
                "You output concise, evidence-based JSON. Keep prose fields short; cap open-ended lists "
                "reasonably so the response stays compact."
            ),
            llm=llm,
            allow_delegation=False,
            verbose=False,
        )
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
        reports = self.run_with_reports()
        return reports["verdict"]

    def run_with_reports(self) -> dict[str, Any]:
        if self._crew_mode_flag == "efficient":
            return self._run_efficient_crew()
        tasks, agents, report_names = self._build_execution_plan()
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()

        task_outputs = getattr(result, "tasks_output", []) or []
        reports: dict[str, Any] = {}
        for name, task_output in zip(report_names, task_outputs, strict=False):
            reports[name] = self._coerce_task_output(task_output)

        final_verdict = reports.get("judge") or self._coerce_result(result)
        return {"verdict": final_verdict, "reports": reports}

    def _run_efficient_crew(self) -> dict[str, Any]:
        """Two LLM calls: unified analyst + judge (vs. 4–5 sequential specialist calls)."""
        resume_ok = self.analysis_data.get("resume_data") is not None
        analyst_task = Task(
            description=self._efficient_analyst_task_description(include_resume=resume_ok),
            expected_output="One JSON object with nested commit_behavior, code_quality, ai_usage, optional resume_match.",
            agent=self.unified_analyst_agent,
            output_json=EfficientAnalystOutput,
        )
        judge_task = Task(
            description=self._judge_task_description(resume_enabled=resume_ok),
            expected_output="A JSON object exactly matching the requested final verdict schema.",
            agent=self.judge_agent,
            context=[analyst_task],
            output_json=JudgeOutput,
        )
        crew = Crew(
            agents=[self.unified_analyst_agent, self.judge_agent],
            tasks=[analyst_task, judge_task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()
        task_outputs = getattr(result, "tasks_output", []) or []
        reports: dict[str, Any] = {}
        if len(task_outputs) >= 1:
            bundle = self._coerce_task_output(task_outputs[0])
            if isinstance(bundle, dict):
                reports["commit_behavior"] = bundle.get("commit_behavior") or {}
                reports["code_quality"] = bundle.get("code_quality") or {}
                reports["ai_usage"] = bundle.get("ai_usage") or {}
                rm = bundle.get("resume_match")
                if rm is not None:
                    reports["resume_match"] = rm
        final_verdict: dict[str, Any]
        if len(task_outputs) >= 2:
            final_verdict = self._coerce_task_output(task_outputs[1])
        else:
            final_verdict = reports.get("judge") or self._coerce_result(result)
        return {"verdict": final_verdict, "reports": reports}

    def _efficient_analyst_task_description(self, *, include_resume: bool) -> str:
        resume_line = (
            "Include a populated `resume_match` object using resume_data, skill_matches, and project_matches.\n"
            if include_resume
            else "Set `resume_match` to null (no resume was provided).\n"
        )
        return (
            "Produce ONE JSON object matching the schema with nested fields: "
            "commit_behavior, code_quality, ai_usage"
            f"{', resume_match' if include_resume else ' (resume_match null)'}.\n\n"
            "Cover in order: (1) commits and commit_patterns vs repo_metadata, (2) code quality from files + "
            "knowledge_graph, (3) AI usage vs baseline_comparison and code, "
            f"{'(4) resume claims vs code evidence. ' if include_resume else ''}"
            "Keep string fields concise; limit open-ended lists to what is most informative.\n\n"
            f"{resume_line}\n"
            "Analysis data:\n"
            f"{self.analysis_json}"
        )

    def _analysis_summary_for_judge(self) -> str:
        """Metadata and job/resume context for the judge — not full file bodies (saves input tokens)."""
        d = self.analysis_data
        files = d.get("files") or []
        overview: list[dict[str, Any]] = []
        if isinstance(files, list):
            for f in files[: min(60, len(files))]:
                if not isinstance(f, dict):
                    continue
                overview.append(
                    {
                        "path": f.get("path") or f.get("file") or "",
                        "line_count": f.get("line_count", 0),
                    }
                )
        kg = d.get("knowledge_graph")
        kg_note = None
        if isinstance(kg, dict):
            kg_note = kg.get("note") or kg.get("warning") or kg.get("fallback_mode")
        summary: dict[str, Any] = {
            "repo_metadata": d.get("repo_metadata"),
            "commit_patterns": d.get("commit_patterns"),
            "baseline_comparison": d.get("baseline_comparison"),
            "resume_data": d.get("resume_data"),
            "skill_matches": d.get("skill_matches"),
            "project_matches": d.get("project_matches"),
            "job_description": d.get("job_description"),
            "company_style_summary": d.get("company_style_summary"),
            "files_overview": overview,
            "knowledge_graph_hint": kg_note,
            "_llm_payload_note": d.get("_llm_payload_note"),
        }
        return json.dumps(summary, indent=2, default=str)

    def _build_execution_plan(self) -> tuple[list[Task], list[Agent], list[str]]:
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
        report_names = ["commit_behavior", "code_quality", "ai_usage"]
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
            report_names.append("resume_match")

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
        report_names.append("judge")

        agents = [
            self.commit_behavior_agent,
            self.code_quality_agent,
            self.ai_usage_agent,
            self.judge_agent,
        ]
        if resume_task is not None:
            agents.insert(3, self.resume_match_agent)
        return tasks, agents, report_names

    @staticmethod
    def _coerce_task_output(task_output: Any) -> dict[str, Any]:
        if getattr(task_output, "json_dict", None):
            return task_output.json_dict
        raw = getattr(task_output, "raw", None) or str(task_output)
        return json.loads(raw)

    @staticmethod
    def _coerce_result(result: Any) -> dict[str, Any]:
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
            "Prior task JSON outputs are your primary evidence; use the supplementary block only for job/resume/metadata context.\n"
            "Return only JSON matching the schema.\n\n"
            "Supplementary context (metadata, job, resume, file overview — not full source):\n"
            f"{self._analysis_summary_for_judge()}"
        )
