"""GitNexus-backed repository analysis helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx


class GitNexusAnalyzer:
    """Analyze a GitHub repository via the GitNexus CLI and MCP server."""

    def __init__(self, github_url: str, base_url: str = "http://localhost:3000") -> None:
        self.github_url = github_url.strip()
        self.base_url = base_url.rstrip("/")
        self.repo_name = self._infer_repo_name(self.github_url)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="gitnexus-"))
        self.repo_dir = self.temp_dir / self.repo_name
        self.mcp_process: subprocess.Popen[bytes] | None = None
        self._http = httpx.Client(timeout=60.0)
        self._session_id: str | None = None
        self._request_id = 0

        self._verify_prerequisites()
        try:
            self._clone_repo()
            self._run_analysis()
            self._start_mcp_server()
        except Exception:
            self.cleanup()
            raise

    def __enter__(self) -> "GitNexusAnalyzer":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.cleanup()

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
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(error_message) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(error_message) from exc

    def _verify_prerequisites(self) -> None:
        node_error = "GitNexus requires Node.js. Install from nodejs.org."
        self._check_binary(["node", "--version"], node_error)
        self._check_binary(["npx", "--version"], node_error)
        self._check_binary(["git", "--version"], "Git is required to clone repositories. Install git and try again.")

    def _clone_repo(self) -> None:
        subprocess.run(
            ["git", "clone", self.github_url, str(self.repo_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

    def _run_analysis(self) -> None:
        subprocess.run(
            ["npx", "gitnexus", "analyze", "--embeddings"],
            check=True,
            cwd=self.repo_dir,
            capture_output=True,
            text=True,
        )

    def _start_mcp_server(self) -> None:
        self.mcp_process = subprocess.Popen(
            ["npx", "gitnexus", "mcp"],
            cwd=self.repo_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_for_server()

    def _wait_for_server(self, timeout_seconds: int = 20) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if self.mcp_process is not None and self.mcp_process.poll() is not None:
                raise RuntimeError("GitNexus MCP server exited unexpectedly during startup.")
            try:
                self._ensure_initialized()
                return
            except Exception:
                time.sleep(1)

        raise RuntimeError(
            "GitNexus MCP server did not become reachable at "
            f"{self.base_url}. Confirm your local GitNexus MCP HTTP bridge is available."
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
        self._post_mcp(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }
        )

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
            error = result["error"]
            raise RuntimeError(f"GitNexus MCP tool '{tool_name}' failed: {error}")
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
            error = result["error"]
            raise RuntimeError(f"GitNexus MCP resource '{resource_uri}' failed: {error}")
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
        result = self._call_with_fallbacks(
            tool_candidates=[
                ("graph_summary", {"repo": self.repo_name}),
                ("context", {"repo": self.repo_name}),
            ],
            resource_candidates=[f"gitnexus://repo/{self.repo_name}/context"],
        )
        return result if isinstance(result, dict) else {"context": result}

    def get_knowledge_graph(self) -> dict[str, Any]:
        return self._repo_context()

    def get_file_contents(self) -> list[dict[str, Any]]:
        result = self._call_with_fallbacks(
            tool_candidates=[
                ("search_files", {"repo": self.repo_name, "query": "", "limit": 10000}),
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
        context = self._repo_context()
        entry_points = context.get("entry_points") or context.get("processes") or []
        if isinstance(entry_points, dict):
            entry_points = entry_points.get("items", [])

        chains: list[dict[str, Any]] = []
        for entry in entry_points:
            if isinstance(entry, dict):
                symbol = entry.get("name") or entry.get("symbol") or entry.get("id")
            else:
                symbol = str(entry)
            if not symbol:
                continue
            try:
                impact = self._call_tool("impact", {"repo": self.repo_name, "symbol": symbol})
                chains.append({"entry_point": symbol, "impact": impact})
            except Exception:
                continue
        return chains

    def get_import_graph(self) -> dict[str, Any]:
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
        result = self._call_with_fallbacks(
            tool_candidates=[
                ("community_clusters", {"repo": self.repo_name}),
                ("clusters", {"repo": self.repo_name}),
            ],
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

    def cleanup(self) -> None:
        if self.mcp_process is not None and self.mcp_process.poll() is None:
            self.mcp_process.terminate()
            try:
                self.mcp_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.mcp_process.kill()
                self.mcp_process.wait(timeout=5)
        self.mcp_process = None

        self._http.close()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    TEST_REPO_URL = "https://github.com/pallets/flask"

    with GitNexusAnalyzer(TEST_REPO_URL) as analyzer:
        print("Knowledge Graph Summary:")
        print(analyzer.get_knowledge_graph())
        print("\nFirst 3 File Entries:")
        print(analyzer.get_file_contents()[:3])
