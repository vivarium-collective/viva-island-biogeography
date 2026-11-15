"""Make a study's canonical run EARNABLE by the discovery-engine grader.

A canonical ``sims/run.py`` builds a composite document and runs it. By default
the document wires a ``local:RAMEmitter`` whose history lives only in memory, so
nothing is persisted and the discovery-engine grader (which opens a run store via
``viva_emitters.RunReader``) has no run to read — behavior tests SKIP as
``not resolvable``. These two helpers fix that with a 2-line change to a run:

1. :func:`attach_sqlite_emitter` repoints the document's emitter at a
   ``local:SQLiteEmitter`` that writes a ``history`` table to a store on disk
   (``gather_emitter_results`` still returns the in-memory rows, so a run's viz
   and prints are unchanged).
2. :func:`register_study_run` upserts the run into the study's ``runs.db`` with
   the store path, so ``discovery_engine.study.outcomes.sync`` finds it, opens
   the store, and grades the derived-scalar tests by CODE against the real run.

The store (``<study>/_runstore/``) and ``runs.db`` are regenerable artifacts
(git-ignored); the durable earned record is the ``computed_outcomes`` that
``sync`` writes back into ``study.yaml``.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

_DB_FILE = "history.db"
_STORE_SUBDIR = "_runstore"


def attach_sqlite_emitter(doc: dict, study_dir: Any, run_id: str,
                          *, emitter_key: str = "emitter") -> str:
    """Repoint ``doc[emitter_key]`` at a persisting SQLiteEmitter and return the
    store db path. Call it on the composite document before ``Composite(...)``."""
    store_dir = Path(study_dir) / _STORE_SUBDIR
    store_dir.mkdir(parents=True, exist_ok=True)
    em = doc[emitter_key]
    em["address"] = "local:SQLiteEmitter"
    cfg = em.setdefault("config", {})
    cfg["file_path"] = str(store_dir)
    cfg["db_file"] = _DB_FILE
    cfg["simulation_id"] = run_id
    return str(store_dir / _DB_FILE)


def register_study_run(study_dir: Any, run_id: str, *, spec_id: str,
                       params: dict, n_steps: int, emitter_path: str,
                       status: str = "complete") -> None:
    """Record the completed run in ``<study>/runs.db`` so the engine can grade it."""
    from discovery_engine.study.run_registry import register_run
    now = time.time()
    register_run(
        Path(study_dir) / "runs.db", run_id,
        spec_id=spec_id, params=params, n_steps=int(n_steps),
        status=status, completed_at=now, emitter_path=str(emitter_path),
    )
