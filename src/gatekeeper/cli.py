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
    
    if args.verbose:
        # Print debug info to stderr
        def print_header(title):
            sys.stderr.write(f"\n{'='*60}\n")
            sys.stderr.write(f" {title}\n")
            sys.stderr.write(f"{'='*60}\n")

        # Phase 1
        print_header("PHASE 1: INITIAL ANALYSIS")
        votes = out.get("phase1_votes", [])
        for v in votes:
            sys.stderr.write(f"{v['vote']:<8} [{v['agent']}]\n")
            sys.stderr.write(f"   Reason: {v['reason']}\n")
            for m in v.get("modifications_required") or []:
                sys.stderr.write(f"   -> Suggestion: {m['field']} = {m['proposed_value']} ({m['priority']})\n")
            sys.stderr.write("\n")

        # Phase 2
        print_header("PHASE 2: DEBATE")
        msgs = out.get("debate_messages", [])
        if not msgs:
            sys.stderr.write("(No debate required - consensus reached)\n\n")
        else:
            for m in msgs:
                targets = ", ".join(m.get("targets", []))
                sys.stderr.write(f"{m['stance']:<8} [{m['agent']}] -> [{targets}]\n")
                sys.stderr.write(f"   \"{m['message']}\"\n\n")

        # Derived Metrics
        print_header("DERIVED METRICS")
        sys.stderr.write(json.dumps(out.get("derived_metrics"), indent=2, ensure_ascii=False))
        sys.stderr.write("\n")

        print_header("PHASE 3: FINAL VERDICT")

    # stdout must be final JSON only for the "Gatekeeper" contract
    # Using indent=2 for better readability as requested ("美觀")
    final_json = json.dumps(out["final_verdict"], ensure_ascii=False, sort_keys=True, indent=2)
    print(final_json)

if __name__ == "__main__":
    main()
