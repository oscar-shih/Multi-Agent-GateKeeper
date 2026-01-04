# src/gatekeeper/cli.py
from __future__ import annotations

import argparse
import time
import json

from gatekeeper.graph.build import build_gatekeeper_graph


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--sim", required=True)
    p.add_argument("--past", required=True)
    p.add_argument("--formulas", required=True)
    args = p.parse_args()

    app = build_gatekeeper_graph()

    start = time.time()
    state = {
        "start_epoch_s": start,
        "deadline_epoch_s": start + 60.0,
        "input_paths": {
            "mesh_report": args.mesh,
            "sim_config": args.sim,
            "past_runs": args.past,
            "formulas": args.formulas,
        },
        "phase1_votes": [],
        "debate_messages": [],
        "trace": {},
    }

    out = app.invoke(state)
    # stdout must be final JSON only
    print(json.dumps(out["final_verdict"], ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    print("DERIVED_METRICS:", out.get("derived_metrics"))
    print("PHASE1_VOTES:", out.get("phase1_votes"))

if __name__ == "__main__":
    main()