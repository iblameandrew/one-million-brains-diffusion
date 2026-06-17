#!/usr/bin/env python3
"""
Minimal voter-pool worker entry — imports million_brains_dflash as a library
(never re-runs its __main__ / argparse). Spawned by MultiAgentEnginePool.
"""
from __future__ import annotations

import json
import os
import sys
import traceback


def main() -> None:
    os.environ["MBR_AGENT_WORKER"] = "1"
    worker_dir = os.path.dirname(os.path.abspath(__file__))
    if worker_dir not in sys.path:
        sys.path.insert(0, worker_dir)

    if len(sys.argv) < 3:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": "usage: mbr_voter_worker.py AGENT_ID GEN_PATH",
                }
            ),
            flush=True,
        )
        raise SystemExit(2)

    agent_id = int(sys.argv[1])
    gen_path = sys.argv[2]
    print(json.dumps({"status": "loading", "agent_id": agent_id}), flush=True)

    library_path = os.path.join(worker_dir, "million_brains_dflash.py")
    if not os.path.isfile(library_path):
        print(
            json.dumps(
                {
                    "status": "error",
                    "agent_id": agent_id,
                    "error": f"worker library missing: {library_path}",
                }
            ),
            flush=True,
        )
        raise SystemExit(1)

    try:
        lib_text = open(library_path, encoding="utf-8", errors="ignore").read()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "agent_id": agent_id,
                    "error": f"cannot read worker library: {exc}",
                }
            ),
            flush=True,
        )
        raise SystemExit(1) from exc

    if "_multi_agent_worker_main" not in lib_text:
        found_ver = "unknown"
        for line in lib_text.splitlines()[:400]:
            if "SCRIPT_VERSION" in line:
                found_ver = line.strip()
                break
        print(
            json.dumps(
                {
                    "status": "error",
                    "agent_id": agent_id,
                    "error": (
                        "stale worker library at "
                        f"{library_path} ({found_ver}); re-run parent notebook cell"
                    ),
                }
            ),
            flush=True,
        )
        raise SystemExit(1)

    try:
        import million_brains_dflash as mbr

        if not hasattr(mbr, "_multi_agent_worker_main"):
            os.execv(
                sys.executable,
                [
                    sys.executable,
                    "-u",
                    library_path,
                    "--mbr-agent-worker",
                    str(agent_id),
                    gen_path,
                ],
            )
        mbr._multi_agent_worker_main(agent_id, gen_path)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "agent_id": agent_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            ),
            flush=True,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()