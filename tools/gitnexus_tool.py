"""GitNexus-backed repository analysis helpers."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import httpx


def git_clone_shallow(github_url: str, repo_dir: Path, *, timeout: int = 180) -> None:
    """Clone a GitHub repo with a shallow, blob-filtered history (shared by analyzers)."""
    depth = os.getenv("GITNEXUS_CLONE_DEPTH", "1").strip()
    clone_cmd = ["git", "clone", "--filter=blob:none"]
    if depth.isdigit() and int(depth) > 0:
        clone_cmd.extend(["--depth", depth])
    clone_cmd.extend([github_url.strip(), str(repo_dir)])
    subprocess.run(clone_cmd, check=True, capture_output=True, text=True, timeout=timeout)


class GitNexusAnalyzer:
    """Analyze a GitHub repository via GitNexus, with a local fallback when MCP is unavailable."""

    filesystem_only = False

    TEXT_EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".sql": "sql",
        ".md": "markdown",
        ".json": "json",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".toml": "toml",
        ".html": "html",
        ".css": "css",
    }
    IGNORE_DIRS = {".git", "node_modules", "dist", "build", "__pycache__", ".venv", "venv"}
    IGNORE_SUFFIXES = {
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
        ".mp3",
        ".mp4",
        ".mov",
        ".webm",
        ".bin",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".lock",
    }

    def __init__(
        self,
        github_url: str,
        base_url: str = "http://localhost:3000",
        on_phase: Callable[[str], None] | None = None,
    ) -> None:
        self.github_url = github_url.strip()
        self.base_url = base_url.rstrip("/")
        self.repo_name = self._infer_repo_name(self.github_url)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="gitnexus-"))
        self.repo_dir = self.temp_dir / self.repo_name
        self.mcp_process: subprocess.Popen[bytes] | None = None
        self._http = httpx.Client(timeout=60.0)
        self._session_id: str | None = None
        self._request_id = 0
        self._mcp_available = False
        self._startup_warning: str | None = None
        self._analysis_warning: str | None = None
        self._cached_fallback_files: list[dict[str, Any]] | None = None
        self._on_phase = on_phase or (lambda _msg: None)

        self._on_phase("Checking Node.js, npx, and git…")
        self._verify_prerequisites()
        self._on_phase("Cloning repository…")
        self._clone_repo()
        self._on_phase("Running GitNexus analyze (set GITNEXUS_USE_EMBEDDINGS=0 to skip embeddings)…")
        self._run_analysis()
        self._on_phase("Starting GitNexus MCP (may fall back to disk scan)…")
        try:
            self._start_mcp_server()
        except Exception as exc:
            # GitNexus MCP often runs over stdio instead of an HTTP bridge.
            # Keep the analysis usable by falling back to direct repo scanning.
            self._startup_warning = str(exc)
            self.cleanup_mcp_only()

    def __enter__(self) -> "GitNexusAnalyzer":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.cleanup()

    @staticmethod
    def scan_repo_directory(repo_dir: Path, repo_name: str) -> list[dict[str, Any]]:
        """Read text-like source files from a cloned repository (no GitNexus required)."""
        cls = GitNexusAnalyzer
        files: list[dict[str, Any]] = []
        for path in sorted(repo_dir.rglob("*")):
            if not path.is_file():
                continue
            if any(part in cls.IGNORE_DIRS for part in path.parts):
                continue
            if path.suffix.lower() in cls.IGNORE_SUFFIXES:
                continue
            language = cls.TEXT_EXTENSIONS.get(path.suffix.lower())
            if language is None and path.suffix:
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            relative_path = path.relative_to(repo_dir).as_posix()
            lang = language or "text"
            symbols = cls._extract_symbols(content, lang)
            comment_density = cls._comment_density(content, lang)
            files.append(
                {
                    "file_path": relative_path,
                    "path": relative_path,
                    "language": lang,
                    "content": content,
                    "symbols": symbols,
                    "comment_density": comment_density,
                    "has_todos": "todo" in content.lower(),
                    "has_debug_prints": cls._has_debug_artifacts(content, lang),
                    "has_commented_out_code": cls._has_commented_out_code(content),
                    "line_count": len(content.splitlines()),
                    "repo": repo_name,
                }
            )
        return files

    @staticmethod
    def build_fallback_knowledge_graph(
        files: list[dict[str, Any]],
        *,
        warning: str | None = None,
        filesystem_only: bool = False,
    ) -> dict[str, Any]:
        function_list: list[dict[str, Any]] = []
        class_list: list[dict[str, Any]] = []
        import_relationships: list[dict[str, Any]] = []
        for file_data in files:
            for symbol in file_data.get("symbols", []):
                if symbol.get("type") in {"function", "method"}:
                    function_list.append({"file_path": file_data["file_path"], "name": symbol.get("name")})
                if symbol.get("type") == "class":
                    class_list.append({"file_path": file_data["file_path"], "name": symbol.get("name")})
                if symbol.get("type") == "import":
                    import_relationships.append(
                        {
                            "source": file_data["file_path"],
                            "import": symbol.get("name"),
                        }
                    )
        return {
            "file_tree": [file_data["file_path"] for file_data in files],
            "function_list": function_list,
            "class_list": class_list,
            "import_relationships": import_relationships,
            "call_chains": [],
            "fallback_mode": True,
            "filesystem_only": filesystem_only,
            "warning": warning,
        }

    @staticmethod
    def _infer_repo_name(github_url: str) -> str:
        parsed = urlparse(github_url)
        if parsed.netloc not in {"github.com", "www.github.com"}:
            raise ValueError("Please provide a valid GitHub repository URL.")

        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if len(parts) < 2:
            raise ValueError("GitHub repository URL must include both owner and repository name.")
        return parts[1].removesuffix(".git")

    @staticmethod
    def _check_binary(command: list[str], error_message: str) -> None:
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(error_message) from exc

    def _verify_prerequisites(self) -> None:
        node_error = "GitNexus requires Node.js. Install from nodejs.org."
        self._check_binary(["node", "--version"], node_error)
        self._check_binary(["npx", "--version"], node_error)
        self._check_binary(["git", "--version"], "Git is required to clone repositories. Install git and try again.")

    @staticmethod
    def _gitnexus_cmd(*args: str) -> list[str]:
        if shutil.which("gitnexus"):
            return ["gitnexus", *args]
        return ["npx", "--yes", "gitnexus", *args]

    def _clone_repo(self) -> None:
        git_clone_shallow(self.github_url, self.repo_dir, timeout=180)

    def _run_analysis(self) -> None:
        use_embeddings = os.getenv("GITNEXUS_USE_EMBEDDINGS", "1").strip().lower() not in {"0", "false", "no"}
        cmd = self._gitnexus_cmd("analyze")
        if use_embeddings:
            cmd.append("--embeddings")
        try:
            subprocess.run(
                cmd,
                check=True,
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or exc.stdout or "").strip()
            self._analysis_warning = f"GitNexus analyze failed (continuing without CLI graph): {stderr or exc}"
        except subprocess.TimeoutExpired as exc:
            self._analysis_warning = (
                "GitNexus analyze timed out (continuing without CLI graph). "
                "Try GITNEXUS_USE_EMBEDDINGS=0 or npm install -g gitnexus."
            )
            _ = exc

    def _start_mcp_server(self) -> None:
        self.mcp_process = subprocess.Popen(
            self._gitnexus_cmd("mcp"),
            cwd=self.repo_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_for_server()

    def _wait_for_server(self, timeout_seconds: int = 8) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if self.mcp_process is not None and self.mcp_process.poll() is not None:
                raise RuntimeError("GitNexus MCP server exited unexpectedly during startup.")
            try:
                self._ensure_initialized()
                self._mcp_available = True
                return
            except Exception:
                time.sleep(0.75)

        raise RuntimeError(
            "GitNexus MCP did not become reachable over HTTP. Falling back to direct repo scanning."
        )

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _mcp_urls(self) -> list[str]:
        return [f"{self.base_url}/mcp", self.base_url]

    def _post_mcp(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        last_error: Exception | None = None
        for url in self._mcp_urls():
            try:
                response = self._http.post(url, headers=headers, json=payload)
                response.raise_for_status()
                session_id = response.headers.get("Mcp-Session-Id") or response.headers.get("mcp-session-id")
                if session_id:
                    self._session_id = session_id
                return self._parse_mcp_response(response)
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"Unable to reach GitNexus MCP server at {self.base_url}.") from last_error

    @staticmethod
    def _parse_mcp_response(response: httpx.Response) -> dict[str, Any]:
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            event_payloads: list[dict[str, Any]] = []
            for line in response.text.splitlines():
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data:
                        event_payloads.append(json.loads(data))
            if not event_payloads:
                raise RuntimeError("GitNexus MCP server returned an empty event stream response.")
            return event_payloads[-1]
        return response.json()

    def _ensure_initialized(self) -> None:
        if self._session_id is not None:
            return

        initialize_payload = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "CodeLens", "version": "0.1.0"},
            },
        }
        self._post_mcp(initialize_payload)
        self._post_mcp({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    def _call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        self._ensure_initialized()
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments or {}},
        }
        result = self._post_mcp(payload)
        if "error" in result:
            raise RuntimeError(f"GitNexus MCP tool '{tool_name}' failed: {result['error']}")
        return self._unwrap_result(result.get("result"))

    def _read_resource(self, resource_uri: str) -> Any:
        self._ensure_initialized()
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "resources/read",
            "params": {"uri": resource_uri},
        }
        result = self._post_mcp(payload)
        if "error" in result:
            raise RuntimeError(f"GitNexus MCP resource '{resource_uri}' failed: {result['error']}")
        return self._unwrap_result(result.get("result"))

    @staticmethod
    def _unwrap_result(result: Any) -> Any:
        if not isinstance(result, dict):
            return result
        content = result.get("content")
        if isinstance(content, list) and len(content) == 1:
            item = content[0]
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    text = item["text"]
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return text
                if item.get("type") == "json" and "json" in item:
                    return item["json"]
        return result

    def _call_with_fallbacks(
        self,
        tool_candidates: list[tuple[str, dict[str, Any]]],
        resource_candidates: list[str] | None = None,
    ) -> Any:
        last_error: Exception | None = None

        for tool_name, arguments in tool_candidates:
            try:
                return self._call_tool(tool_name, arguments)
            except Exception as exc:
                last_error = exc
        for resource_uri in resource_candidates or []:
            try:
                return self._read_resource(resource_uri)
            except Exception as exc:
                last_error = exc
        raise RuntimeError("Unable to retrieve data from GitNexus MCP.") from last_error

    def _repo_context(self) -> dict[str, Any]:
        if not self._mcp_available:
            return self._fallback_knowledge_graph()
        result = self._call_with_fallbacks(
            tool_candidates=[("graph_summary", {"repo": self.repo_name}), ("context", {"repo": self.repo_name})],
            resource_candidates=[f"gitnexus://repo/{self.repo_name}/context"],
        )
        return result if isinstance(result, dict) else {"context": result}

    def get_knowledge_graph(self) -> dict[str, Any]:
        self._on_phase("Building knowledge graph…")
        return self._repo_context()

    def get_file_contents(self) -> list[dict[str, Any]]:
        self._on_phase("Loading file list (MCP or disk scan; large repos take longer)…")
        if not self._mcp_available:
            return self._fallback_file_contents()
        try:
            file_limit = max(1, int(os.getenv("GITNEXUS_FILE_LIMIT", "2000")))
        except ValueError:
            file_limit = 2000
        result = self._call_with_fallbacks(
            tool_candidates=[
                ("search_files", {"repo": self.repo_name, "query": "", "limit": file_limit}),
                ("query", {"repo": self.repo_name, "query": "source files", "limit": 1000}),
            ]
        )
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            for key in ("files", "results", "matches", "items"):
                value = result.get(key)
                if isinstance(value, list):
                    return value
        return [{"raw": result}]

    def get_call_chains(self) -> list[dict[str, Any]]:
        if not self._mcp_available:
            return []
        context = self._repo_context()
        entry_points = context.get("entry_points") or context.get("processes") or []
        if isinstance(entry_points, dict):
            entry_points = entry_points.get("items", [])
        chains: list[dict[str, Any]] = []
        for entry in entry_points:
            symbol = entry.get("name") or entry.get("symbol") or entry.get("id") if isinstance(entry, dict) else str(entry)
            if not symbol:
                continue
            try:
                impact = self._call_tool("impact", {"repo": self.repo_name, "symbol": symbol})
                chains.append({"entry_point": symbol, "impact": impact})
            except Exception:
                continue
        return chains

    def get_import_graph(self) -> dict[str, Any]:
        if not self._mcp_available:
            return self._fallback_import_graph()
        result = self._call_with_fallbacks(
            tool_candidates=[
                ("import_graph", {"repo": self.repo_name}),
                (
                    "cypher",
                    {
                        "repo": self.repo_name,
                        "query": (
                            "MATCH (f:File)-[r:IMPORTS]->(t) "
                            "RETURN f.path AS source, collect({target: coalesce(t.path, t.name), type: type(r)}) AS imports"
                        ),
                    },
                ),
            ]
        )
        return result if isinstance(result, dict) else {"imports": result}

    def get_community_clusters(self) -> list[dict[str, Any]]:
        if not self._mcp_available:
            return self._fallback_community_clusters()
        result = self._call_with_fallbacks(
            tool_candidates=[("community_clusters", {"repo": self.repo_name}), ("clusters", {"repo": self.repo_name})],
            resource_candidates=[f"gitnexus://repo/{self.repo_name}/clusters"],
        )
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            for key in ("clusters", "communities", "items"):
                value = result.get(key)
                if isinstance(value, list):
                    return value
        return [{"raw": result}]

    def _fallback_file_contents(self) -> list[dict[str, Any]]:
        if self._cached_fallback_files is not None:
            return self._cached_fallback_files
        self._cached_fallback_files = GitNexusAnalyzer.scan_repo_directory(self.repo_dir, self.repo_name)
        return self._cached_fallback_files

    def _fallback_knowledge_graph(self) -> dict[str, Any]:
        warn_parts = [self._startup_warning, self._analysis_warning]
        warning = "; ".join(w for w in warn_parts if w) or None
        return GitNexusAnalyzer.build_fallback_knowledge_graph(
            self._fallback_file_contents(),
            warning=warning,
            filesystem_only=False,
        )

    def _fallback_import_graph(self) -> dict[str, Any]:
        import_map: dict[str, list[str]] = {}
        for file_data in self._fallback_file_contents():
            imports = [symbol.get("name", "") for symbol in file_data.get("symbols", []) if symbol.get("type") == "import"]
            import_map[file_data["file_path"]] = [item for item in imports if item]
        return {"imports": import_map, "fallback_mode": True}

    def _fallback_community_clusters(self) -> list[dict[str, Any]]:
        buckets: dict[str, list[str]] = {}
        for file_data in self._fallback_file_contents():
            top_level = file_data["file_path"].split("/", 1)[0]
            buckets.setdefault(top_level, []).append(file_data["file_path"])
        return [{"community": name, "files": members} for name, members in buckets.items()]

    @staticmethod
    def _extract_symbols(content: str, language: str) -> list[dict[str, Any]]:
        symbols: list[dict[str, Any]] = []
        lines = content.splitlines()
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if language == "python":
                if stripped.startswith("def "):
                    name = stripped[4:].split("(", 1)[0].strip()
                    symbols.append({"type": "function", "name": name, "code": GitNexusAnalyzer._extract_block(lines, idx), "line_start": idx})
                elif stripped.startswith("class "):
                    name = stripped[6:].split("(", 1)[0].split(":", 1)[0].strip()
                    symbols.append({"type": "class", "name": name, "code": GitNexusAnalyzer._extract_block(lines, idx), "line_start": idx})
                elif stripped.startswith("import ") or stripped.startswith("from "):
                    symbols.append({"type": "import", "name": stripped, "line_start": idx})
            elif language in {"javascript", "typescript"}:
                func_match = re.match(r"(export\s+)?(async\s+)?function\s+([A-Za-z0-9_]+)", stripped)
                class_match = re.match(r"(export\s+)?class\s+([A-Za-z0-9_]+)", stripped)
                const_func_match = re.match(r"(export\s+)?const\s+([A-Za-z0-9_]+)\s*=\s*(async\s*)?\(", stripped)
                import_match = stripped.startswith("import ")
                if func_match:
                    symbols.append({"type": "function", "name": func_match.group(3), "code": GitNexusAnalyzer._extract_js_block(lines, idx), "line_start": idx})
                elif class_match:
                    symbols.append({"type": "class", "name": class_match.group(2), "code": GitNexusAnalyzer._extract_js_block(lines, idx), "line_start": idx})
                elif const_func_match:
                    symbols.append({"type": "function", "name": const_func_match.group(2), "code": GitNexusAnalyzer._extract_js_block(lines, idx), "line_start": idx})
                elif import_match:
                    symbols.append({"type": "import", "name": stripped, "line_start": idx})
        return symbols

    @staticmethod
    def _extract_block(lines: list[str], start_line: int) -> str:
        start_index = start_line - 1
        base_indent = len(lines[start_index]) - len(lines[start_index].lstrip(" "))
        block = [lines[start_index]]
        for line in lines[start_index + 1 :]:
            stripped = line.strip()
            current_indent = len(line) - len(line.lstrip(" "))
            if stripped and current_indent <= base_indent:
                break
            block.append(line)
        return "\n".join(block).strip()

    @staticmethod
    def _extract_js_block(lines: list[str], start_line: int) -> str:
        start_index = start_line - 1
        block = [lines[start_index]]
        brace_balance = lines[start_index].count("{") - lines[start_index].count("}")
        for line in lines[start_index + 1 :]:
            block.append(line)
            brace_balance += line.count("{") - line.count("}")
            if brace_balance <= 0 and "{" in lines[start_index]:
                break
        return "\n".join(block).strip()

    @staticmethod
    def _comment_density(content: str, language: str) -> float:
        lines = [line for line in content.splitlines() if line.strip()]
        if not lines:
            return 0.0
        comment_count = 0
        for line in lines:
            stripped = line.strip()
            if language == "python" and stripped.startswith("#"):
                comment_count += 1
            elif language in {"javascript", "typescript", "java", "go", "rust", "c", "cpp", "csharp", "php", "swift", "kotlin"} and (
                stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*")
            ):
                comment_count += 1
        return comment_count / len(lines)

    @staticmethod
    def _has_debug_artifacts(content: str, language: str) -> bool:
        lowered = content.lower()
        if language == "python":
            return "print(" in content or "pdb.set_trace" in lowered
        if language in {"javascript", "typescript"}:
            return "console.log" in lowered or "debugger" in lowered
        return False

    @staticmethod
    def _has_commented_out_code(content: str) -> bool:
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("//"):
                if any(token in stripped for token in ("def ", "class ", "return ", "if ", "for ", "while ", "function ", "const ", "let ", "var ")):
                    return True
        return False

    def cleanup_mcp_only(self) -> None:
        if self.mcp_process is not None and self.mcp_process.poll() is None:
            self.mcp_process.terminate()
            try:
                self.mcp_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.mcp_process.kill()
                self.mcp_process.wait(timeout=5)
        self.mcp_process = None

    def cleanup(self) -> None:
        self.cleanup_mcp_only()
        self._http.close()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


class FilesystemRepoAnalyzer:
    """Clone a GitHub repo and scan source files without GitNexus (full pipeline fallback)."""

    filesystem_only = True

    def __init__(
        self,
        github_url: str,
        *,
        prior_error: str | None = None,
        on_phase: Callable[[str], None] | None = None,
    ) -> None:
        self.github_url = github_url.strip()
        self.repo_name = GitNexusAnalyzer._infer_repo_name(self.github_url)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="codelens-fs-"))
        self.repo_dir = self.temp_dir / self.repo_name
        self._cached_files: list[dict[str, Any]] | None = None
        self.prior_error = prior_error
        self._on_phase = on_phase or (lambda _msg: None)
        GitNexusAnalyzer._check_binary(
            ["git", "--version"],
            "Git is required to clone repositories. Install git and try again.",
        )
        try:
            self._on_phase("Cloning repository (filesystem-only; GitNexus unavailable)…")
            git_clone_shallow(self.github_url, self.repo_dir, timeout=180)
        except Exception:
            self.cleanup()
            raise

    def __enter__(self) -> "FilesystemRepoAnalyzer":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.cleanup()

    def get_file_contents(self) -> list[dict[str, Any]]:
        self._on_phase("Scanning files on disk…")
        if self._cached_files is None:
            self._cached_files = GitNexusAnalyzer.scan_repo_directory(self.repo_dir, self.repo_name)
        return self._cached_files

    def get_knowledge_graph(self) -> dict[str, Any]:
        self._on_phase("Building knowledge graph from disk scan…")
        detail = self.prior_error or "Filesystem-only mode: GitNexus was skipped or could not start."
        return GitNexusAnalyzer.build_fallback_knowledge_graph(
            self.get_file_contents(),
            warning=detail,
            filesystem_only=True,
        )

    def cleanup(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


@contextmanager
def open_repo_analyzer(
    github_url: str,
    on_phase: Callable[[str], None] | None = None,
) -> Iterator[GitNexusAnalyzer | FilesystemRepoAnalyzer]:
    """Prefer GitNexus; if initialization fails entirely, clone and scan the repo on disk."""
    analyzer: GitNexusAnalyzer | FilesystemRepoAnalyzer | None = None
    try:
        try:
            analyzer = GitNexusAnalyzer(github_url, on_phase=on_phase)
        except Exception as exc:
            analyzer = FilesystemRepoAnalyzer(
                github_url,
                prior_error=f"GitNexus unavailable: {exc}",
                on_phase=on_phase,
            )
        yield analyzer
    finally:
        if analyzer is not None:
            analyzer.cleanup()


if __name__ == "__main__":
    TEST_REPO_URL = "https://github.com/pallets/flask"
    with GitNexusAnalyzer(TEST_REPO_URL) as analyzer:
        print("Knowledge Graph Summary:")
        print(analyzer.get_knowledge_graph())
        print("\nFirst 3 File Entries:")
        print(analyzer.get_file_contents()[:3])
