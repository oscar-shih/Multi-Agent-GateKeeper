# src/gatekeeper/cli.py
from __future__ import annotations

import argparse
import time
import json
import sys
from dotenv import load_dotenv

from gatekeeper.graph.build import build_gatekeeper_graph


def main():
    load_dotenv()
    
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--sim", required=True)
    p.add_argument("--past", required=True)
    p.add_argument("--formulas", required=True)
    p.add_argument("--verbose", action="store_true", help="Print trace and intermediate outputs to stderr")
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
    
    # stdout must be final JSON only for the "Gatekeeper" contract
    final_json = json.dumps(out["final_verdict"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    print(final_json)

    if args.verbose:
        # Print debug info to stderr so it doesn't pollute the JSON stdout
        sys.stderr.write("\n--- PHASE 1 VOTES ---\n")
        sys.stderr.write(json.dumps(out.get("phase1_votes"), indent=2, ensure_ascii=False))
        sys.stderr.write("\n\n--- DEBATE MESSAGES ---\n")
        sys.stderr.write(json.dumps(out.get("debate_messages"), indent=2, ensure_ascii=False))
        sys.stderr.write("\n\n--- DERIVED METRICS ---\n")
        sys.stderr.write(json.dumps(out.get("derived_metrics"), indent=2, ensure_ascii=False))
        sys.stderr.write("\n")

if __name__ == "__main__":
    main()
