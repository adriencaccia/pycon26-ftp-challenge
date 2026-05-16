"""Parallel build scheduler for the PyCon 2026 free-threading challenge.

Approach
--------
Standard parallel topological scheduler:

1. Pre-compute, for each target, how many of its deps are still unbuilt
   (`pending_deps`) and the list of targets that depend on it (`dependents`).
2. Submit every target with `pending_deps == 0` to a thread pool.
3. When a target finishes, decrement `pending_deps` for each of its dependents
   under a single lock; any dependent that hits 0 is now ready and gets
   submitted to the pool.
4. We're done when every target has completed.

Why this works under free-threading
-----------------------------------
`target.build()` is pure CPU work in Python (sha256 plus an FNV-style loop over
the source bytes). On a 3.14t (no-GIL) build, multiple threads execute that
Python bytecode in parallel, so a thread pool actually scales.

Tuning notes
------------
* One global lock protects `pending_deps`, `results`, and the "remaining"
  counter. The critical section is tiny (a few integer decrements and a dict
  write) compared to the build work, so contention is low even with many
  workers.
* We build each dep_results dict *inside* the lock so the submitting thread
  observes a consistent snapshot of results. We hand it off and submit
  *outside* the lock to keep the critical section minimal.
* Chain inlining: when a target's completion frees one or more new targets,
  we execute the first one inline (in the worker thread that just finished)
  and submit the rest to the pool. On a pure chain graph this collapses the
  whole walk into a single worker with zero re-dispatch overhead; on graphs
  with real parallelism, the "submit the rest" path keeps the pool saturated.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from graph import BuildGraph, Target


def build_all(graph: BuildGraph) -> dict[str, bytes]:
    targets = graph.targets

    pending_deps: dict[str, int] = {name: len(t.deps) for name, t in targets.items()}
    dependents: dict[str, list[Target]] = {name: [] for name in targets}
    for t in targets.values():
        for d in t.deps:
            dependents[d.name].append(t)

    results: dict[str, bytes] = {}
    lock = threading.Lock()
    remaining = len(targets)
    done = threading.Event()

    n_workers = max(1, (os.cpu_count() or 1))
    executor = ThreadPoolExecutor(max_workers=n_workers)

    def submit(target: Target, dep_results: dict[str, bytes]) -> None:
        fut = executor.submit(target.build, dep_results)
        fut.add_done_callback(lambda f, n=target.name: on_done(n, f))

    def on_done(name: str, fut: Future) -> None:
        nonlocal remaining
        cur_name = name
        cur_result = fut.result()
        while True:
            new_ready: list[tuple[Target, dict[str, bytes]]] = []
            with lock:
                results[cur_name] = cur_result
                remaining -= 1
                if remaining == 0:
                    done.set()
                for dep_target in dependents[cur_name]:
                    pending_deps[dep_target.name] -= 1
                    if pending_deps[dep_target.name] == 0:
                        dep_results = {d.name: results[d.name] for d in dep_target.deps}
                        new_ready.append((dep_target, dep_results))
            if not new_ready:
                return
            # Submit all but the first; run the first inline in this thread.
            for target, dep_results in new_ready[1:]:
                submit(target, dep_results)
            inline_target, inline_deps = new_ready[0]
            cur_name = inline_target.name
            cur_result = inline_target.build(inline_deps)

    initial: list[Target] = [targets[n] for n, p in pending_deps.items() if p == 0]
    for target in initial:
        submit(target, {})

    done.wait()
    executor.shutdown(wait=False)
    return results
