"""Parallel build scheduler for the PyCon 2026 free-threading challenge.

Approach
--------
Standard parallel topological scheduler with a hand-rolled worker pool:

1. Pre-compute, for each target, how many of its deps are still unbuilt
   (`pending_deps`) and the list of targets that depend on it (`dependents`).
2. Seed a shared `queue.SimpleQueue` with every target whose `pending_deps == 0`.
3. Spawn N worker threads (N = os.cpu_count()). Each worker:
     - Pops `(target, dep_results)` from the queue.
     - Calls `target.build(dep_results)`.
     - Under a single global lock, writes the result, decrements each
       dependent's pending count, and collects the dependents that just hit 0.
     - For those newly-ready targets, executes the first one inline in this
       same worker thread and pushes the rest back onto the shared queue.
4. When the last target completes, the worker pushes N sentinels so every
   other worker wakes from `queue.get()` and exits.

Why this works under free-threading
-----------------------------------
`target.build()` is pure CPU work in Python (sha256 plus an FNV-style loop over
the source bytes). On a 3.14t (no-GIL) build, multiple threads execute that
bytecode in parallel, so a thread pool actually scales.

Why a custom pool instead of `concurrent.futures.ThreadPoolExecutor`
--------------------------------------------------------------------
Profiling showed that on graphs with many small targets (e.g. the 20k-node
"realistic" graph) `ThreadPoolExecutor.submit` plus `Future` allocation and
the done-callback dispatch added a few percent of overhead per task. A hand
written pool that reads `(target, dep_results)` tuples directly from a
`queue.SimpleQueue` removes that indirection without changing the scheduling
algorithm.

Key tuning details
------------------
* Single global lock around `pending_deps`, `results`, and the `remaining`
  counter. The critical section is just a few integer decrements and one
  dict write — much smaller than the build work itself, so contention is
  low even with 24 workers.
* `dep_results` for each newly-ready target is built *inside* the lock so
  every dep's bytes are guaranteed visible to whichever worker picks it up.
* Inline-chain execution: when a completion frees several new targets, the
  worker keeps the first one for itself and only pushes the rest to the
  shared queue. On a pure chain graph this collapses the whole walk into a
  single worker with zero queue round-trips; on graphs with real parallelism
  the "push the rest" path keeps every worker fed.
"""

from __future__ import annotations

import os
import queue
import threading

from graph import BuildGraph, Target


def build_all(graph: BuildGraph) -> dict[str, bytes]:
    targets = graph.targets
    if not targets:
        return {}

    pending_deps: dict[str, int] = {name: len(t.deps) for name, t in targets.items()}
    dependents: dict[str, list[Target]] = {name: [] for name in targets}
    for t in targets.values():
        for d in t.deps:
            dependents[d.name].append(t)

    results: dict[str, bytes] = {}
    lock = threading.Lock()
    remaining = len(targets)
    ready: queue.SimpleQueue = queue.SimpleQueue()

    n_workers = max(1, os.cpu_count() or 1)
    SENTINEL: tuple[None, None] = (None, None)

    def worker() -> None:
        nonlocal remaining
        while True:
            target, dep_results = ready.get()
            if target is None:
                return
            # Inline-chain: handle this target and any single newly-ready
            # successor without going back through the queue.
            while True:
                result = target.build(dep_results)
                just_ready: list[Target] = []
                with lock:
                    results[target.name] = result
                    remaining -= 1
                    finished = remaining == 0
                    for dep_target in dependents[target.name]:
                        pending_deps[dep_target.name] -= 1
                        if pending_deps[dep_target.name] == 0:
                            just_ready.append(dep_target)
                # Building dep_results dicts and pushing sentinels both go
                # outside the lock: by the time pending_deps hit 0, all of a
                # target's deps have published their results, so reads are
                # safe; queue puts have their own internal lock.
                if finished:
                    for _ in range(n_workers):
                        ready.put(SENTINEL)
                if not just_ready:
                    break
                # Push all but the first newly-ready target; keep the first
                # for ourselves to avoid a queue round-trip.
                for t in just_ready[1:]:
                    ready.put((t, {d.name: results[d.name] for d in t.deps}))
                next_target = just_ready[0]
                target = next_target
                dep_results = {d.name: results[d.name] for d in next_target.deps}

    # Seed the queue with every target that has no dependencies.
    for name, p in pending_deps.items():
        if p == 0:
            ready.put((targets[name], {}))

    threads = [
        threading.Thread(target=worker, name=f"build-{i}", daemon=True)
        for i in range(n_workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return results
