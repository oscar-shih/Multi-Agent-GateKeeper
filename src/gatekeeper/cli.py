from __future__ import annotations

import argparse
import time
import json
import sys
from dotenv import load_dotenv

from gatekeeper.graph.build import build_gatekeeper_graph
from gatekeeper.utils.logging import Logger

def main():
    load_dotenv()
    
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--sim", required=True)
    p.add_argument("--past", required=True)
    p.add_argument("--formulas", required=True)
    p.add_argument("--verbose", action="store_true", help="Print trace and intermediate outputs to stderr")
    p.add_argument("--log", help="Path to save execution log")
    args = p.parse_args()

    # Initialize Logger with file support
    log = Logger(verbose=args.verbose, log_file=args.log)

    try:
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
        
        # --- PHASE 1 ---
        log.header("PHASE 1: INITIAL ANALYSIS")
        votes = out.get("phase1_votes", [])
        for v in votes:
            log.vote(v['agent'], v['vote'], v['reason'], v.get('modifications_required'))

        # --- PHASE 2 ---
        log.header("PHASE 2: DEBATE")
        msgs = out.get("debate_messages", [])
        if not msgs:
            log.info("(No debate required - consensus reached)\n")
        else:
            for m in msgs:
                log.debate(m['agent'], m['stance'], m.get('targets', []), m['message'])

        # --- METRICS ---
        log.header("DERIVED METRICS")
        log.json(out.get("derived_metrics"))
        
        # --- FINAL ---
        log.header("PHASE 3: FINAL VERDICT")

        # stdout must be final JSON only for the "Gatekeeper" contract
        final_json = json.dumps(out["final_verdict"], ensure_ascii=False, sort_keys=True, indent=2)
        print(final_json)
        
        # Also log final JSON to file if enabled
        if args.log:
            log.info("\n=== FINAL OUTPUT (STDOUT) ===")
            log.info(final_json)

    finally:
        log.close()

if __name__ == "__main__":
    main()
