# Multi-Agent-GateKeeper
In Computational Fluid Dynamics (CFD) and Finite Element Analysis (FEA), there is no such thing as a "perfect" simulation. It is always a zero-sum trade-off:

**Accuracy vs. Cost**: To get 1% more accuracy, you often need 10x more compute power.

**Stability vs. Speed**: To make a simulation run faster (larger time steps), you risk the math "exploding" (diverging).

We cannot trust a single LLM to make these decisions because it will hallucinate a "perfect" answer that doesn't exist. We need a System of Checks and Balances. In other words, a council of experts.



## The Challenge: 
Build a Multi-Agent Orchestration System (using LangGraph, AutoGen, or CrewAI) that acts as a "Gatekeeper" for our simulation pipeline. It takes a raw job_config.json and a mesh_report.json as input, runs a structured debate, and outputs a deterministic Go/No-Go decision.

## Personas & Voting Policies
You must implement 5 distinct agents. We have provided their "Core Motivations" below. You will need to prompt them to act as adversarial stakeholders.

1. **The Geometer**: Obsessed with geometric topology.
   - **REJECT**: Skewness $\ge$ 0.90 (Hard constraint).
   - **REJECT**: Orthogonal Quality < 0.10 (Hard constraint).
   - **APPROVE**: Metrics within acceptable range.

2. **The Physicist**: Cares about the validity of the scientific model.
   - **REJECT**: Invalid turbulence model enum (Allowed: `k-epsilon`, `k-omega`, `Spalart-Allmaras`).
   - **REJECT**: "Laminar" flow regime requested with velocity $\ge$ 100 m/s (Physically impossible).
   - **APPROVE**: Valid physics configuration.

3. **The Resource Manager**: Cares only about ROI (Return on Investment) and simulation runtime.
   - **REJECT**: Estimated Cost > Budget * 1.2 (>20% overrun).
   - **MODIFY**: Budget < Estimated Cost $\le$ Budget * 1.2 (0-20% overrun).
   - **APPROVE**: Estimated Cost $\le$ Budget.

4. **The Historian**: Has access to a mock RAG database of past runs.
   - **REJECT**: High similarity ($\ge$ 0.95) to a past CRASHED run with **no** stabilization changes.
   - **MODIFY**: Moderate/High similarity ($\ge$ 0.85) to a past CRASHED run.
   - **APPROVE**: No similar crash patterns found (< 0.85 similarity).

5. **The Stabilizer**: Obsessed with the CFL Condition (Courant–Friedrichs–Lewy).
   - **REJECT**: CFL Number > Max_Allowed * 2.0 (Fundamental instability).
   - **MODIFY**: Max_Allowed < CFL Number $\le$ Max_Allowed * 2.0 (Risky, recommends smaller dt).
   - **APPROVE**: CFL Number $\le$ Max_Allowed.

6. **The Synthesizer (Final Verdict)**:
   - **UNANIMOUS APPROVE**: If all agents vote APPROVE.
   - **HARD REJECT**: If *any* agent triggers a hard constraint (REJECT).
   - **DEBATE RESOLUTION**: If agents vote MODIFY, the Synthesizer evaluates the debate trace. Strict policy enforces REJECT for unauthorized budget overruns.



## Orchestration:

Phase 1: Analysis. Each agent reviews the JSONs independently and casts a "Vote" (Approve/Reject/Modify) with a reasoning string.

Phase 2: The Debate. If votes are not unanimous, agents with opposing views must exchange exactly one round of arguments. (e.g., The Physicist argues for accuracy; The CFO argues for budget).

Phase 3: The Verdict. A "Synthesizer" node reviews the debate and produces the final JSON.



## Constraints:

Deterministic Output: The final output must be a valid JSON object: {"status": "APPROVED", "confidence": 0.0-1.0, "modifications_required": [...]}. No markdown, no chat logs.

The "Calculated" Constraint: The Stabilizer agent cannot hallucinate math. You must give it a Tool (Python function) to actually calculate the Courant Number (C=uΔt/Δx) based on the input JSON.

Hallucination Control: The Physicist must be constrained to a specific list of "Allowed Turbulence Models" (e.g., k-epsilon, k-omega, Spalart-Allmaras). If the JSON requests "Magic-Model-GPT," the agent must flag it as an invalid enum.

Timebox: The entire chain (from Input to JSON) must execute in under 60 seconds.


## Repo Architecture
```
Multi-Agent-GateKeeper/
  README.md
  pyproject.toml
  requirements.txt
  .gitignore

  data/
    inputs/
      mesh_report.json
      sim_config.json
      past_runs.json
      simulation_formulas.json
    test/
      test_hallucination_config.json
      test_impossible_config.json
      test_unstable_config.json

  src/gatekeeper/
    __init__.py
    cli.py                      # python -m gatekeeper run ...
    config.py                   # model, temperature=0, timeouts, thresholds
    schemas.py                  # Pydantic models: JobConfig, MeshReport, Votes, Verdict
    io/
      load_json.py              
      normalize.py              
    tools/
      cfl.py                    # compute_courant(u, dt, dx)
      cost.py                   # estimate_cost(mesh, sim, formulas)
      similarity.py             # historian mock RAG similarity (deterministic)
    prompts/
      geometer.system.txt
      physicist.system.txt
      resource_manager.system.txt
      historian.system.txt
      stabilizer.system.txt
      synthesizer.system.txt
    agents/
      base.py                   # invoke_llm_json(), schema validation
      geometer.py
      physicist.py
      resource_manager.py
      historian.py
      stabilizer.py
      synthesizer.py
    graph/
      state.py                  # LangGraph state type
      nodes.py                  # Function for each node
      build.py                  # compile LangGraph
    utils/
      timebox.py                # 60 seconds
      logging.py                # Clean trace log
```
## Environment
```
conda create -n cfd-gatekeeper python=3.11 -y
conda activate cfd-gatekeeper
pip install -e .
python -m gatekeeper.cli --help
```

## Testing

### Given Case
```
python -m gatekeeper.cli \
  --mesh data/inputs/mesh_report.json \
  --sim data/inputs/sim_config.json \
  --past data/inputs/past_runs.json \
  --formulas data/inputs/simulation_formulas.json \
  --log run_base.log \
  --verbose
```
### The Impossible Job (High Accuracy, Low Budget)
```
python -m gatekeeper.cli  \
  --mesh data/inputs/mesh_report.json \
  --sim data/test/test_impossible_config.json \
  --past data/inputs/past_runs.json \
  --formulas data/inputs/simulation_formulas.json \
  --log run_impossible.log \
  --verbose
```

### The Unstable Job (CFL Violation)
```
python -m gatekeeper.cli  \
  --mesh data/inputs/mesh_report.json \
  --sim data/test/test_unstable_config.json \
  --past data/inputs/past_runs.json \
  --formulas data/inputs/simulation_formulas.json \
  --log run_unstable.log \
  --verbose
```

### The Hallucination Job (Fake Model)
```
python -m gatekeeper.cli  \
  --mesh data/inputs/mesh_report.json \
  --sim data/test/test_hallucination_config.json \
  --past data/inputs/past_runs.json \
  --formulas data/inputs/simulation_formulas.json \
  --log run_hallucination.log \
  --verbose
```

## Note
“Agents are implemented as adversarial stakeholders; Phase 1 is deterministic policy evaluation with tool-verified evidence; Phase 2 uses Gemini to produce a structured, one-round debate trace grounded strictly in the Phase 1 evidence; final verdict remains deterministic per spec.”