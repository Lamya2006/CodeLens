"""Integration smoke tests for repo indexing (needs network + git + optional Node/npx).

Run from repo root:
  ./venv/bin/pip install pytest -q
  GITNEXUS_USE_EMBEDDINGS=0 ./venv/bin/pytest tests/test_open_repo_analyzer_smoke.py -v --tb=short
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _fast_gitnexus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITNEXUS_USE_EMBEDDINGS", "0")


def test_open_repo_analyzer_loads_werkzeug() -> None:
    from tools.gitnexus_tool import open_repo_analyzer

    phases: list[str] = []
    url = "https://github.com/pallets/werkzeug"
    with open_repo_analyzer(url, on_phase=phases.append) as analyzer:
        files = analyzer.get_file_contents()
        assert isinstance(files, list)
        assert len(files) >= 3
        kg = analyzer.get_knowledge_graph()
        assert isinstance(kg, dict)
        assert kg.get("fallback_mode") is True or "file_tree" in kg or "function_list" in kg

    assert phases, "expected progress callbacks"
