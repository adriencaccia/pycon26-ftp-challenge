"""Local CodSpeed walltime benchmarks for iterating on our submission.

Run with:
    .venv/bin/pytest benchmarks/ --codspeed --codspeed-mode=walltime

Walltime mode is required because we rely on real multi-threaded parallelism
under Python 3.14t — instrumentation mode would serialize and mislead.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "challenge"))

from graph import BuildGraph  # noqa: E402


def _load_submission():
    path = ROOT / "submissions" / "adriencaccia.py"
    spec = importlib.util.spec_from_file_location("submission", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SUBMISSION = _load_submission()
GRAPH_NAMES = ["chain", "diamond", "realistic", "tree", "wide"]


@pytest.mark.parametrize("graph_name", GRAPH_NAMES)
def test_build_all(benchmark, graph_name: str):
    graph_path = ROOT / "graphs" / f"{graph_name}.json"
    graph = BuildGraph.load(str(graph_path))
    benchmark(SUBMISSION.build_all, graph)
