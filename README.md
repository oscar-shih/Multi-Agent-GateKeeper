# Multi-Agent-GateKeeper
In Computational Fluid Dynamics (CFD) and Finite Element Analysis (FEA), there is no such thing as a "perfect" simulation. It is always a zero-sum trade-off:

**Accuracy vs. Cost**: To get 1% more accuracy, you often need 10x more compute power.

**Stability vs. Speed**: To make a simulation run faster (larger time steps), you risk the math "exploding" (diverging).

We cannot trust a single LLM to make these decisions because it will hallucinate a "perfect" answer that doesn't exist. We need a System of Checks and Balances. In other words, a council of experts.



## The Challenge: 
Build a Multi-Agent Orchestration System (using LangGraph, AutoGen, or CrewAI) that acts as a "Gatekeeper" for our simulation pipeline. It takes a raw job_config.json and a mesh_report.json as input, runs a structured debate, and outputs a deterministic Go/No-Go decision.

## Personas: 
You must implement 5 distinct agents. We have provided their "Core Motivations" below. You will need to prompt them to act as adversarial stakeholders.

1. The Geometer: Obsessed with geometric topology.

- Trigger: "This mesh has a skewness of 0.9. That is garbage. Reject."

2. The Physicist: Cares about the validity of the scientific model.

- Trigger: "You cannot use a 'Laminar' model for a flow moving at 100m/s. That is physically impossible. Reject."

3. The Resource Manager: Cares only about ROI (Return on Investment) and simulation runtime.

- Trigger: "This mesh has 50 million cells for a simple pipe. That will cost $400 and 5 days to run. The budget is $50. Reject."

4. The Historian: Has access to a mock RAG database of past runs.

- Trigger: "I am voting REJECT. This setup looks 95% similar to Job #402, which crashed. Unless you changed the relaxation factors, this will fail too."

5. The Stabilizer: Obsessed with the CFL Condition (Courant–Friedrichs–Lewy).

- Trigger: "The Physicist wants to run at dt=0.1s, but the Geometer made cells size dx=0.001m. At velocity u=50m/s, the Courant number is 5000. This is unstable. Reject."



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
  requirements.txt
  .gitignore

  data/
    inputs/
      mesh_report.json
      sim_config.json
      past_runs.json
      simulation_formulas.json

  src/gatekeeper/
    __init__.py
    cli.py                      # python -m gatekeeper run ...
    config.py                   # model, temperature=0, timeouts, thresholds
    schemas.py                  # Pydantic models: JobConfig, MeshReport, Votes, Verdict
    io/
      load_json.py              # Support JSON with // comments: strip or json5
      normalize.py              
    tools/
      cfl.py                    # compute_courant(u, dt, dx)
      cost.py                   # estimate_cost(mesh, sim, formulas)
      similarity.py             # historian mock RAG similarity (deterministic)
    rag/
      historian_store.py        # Read past_runs.json + top_k retrieval
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
      nodes.py                  # 每個 node 的純函數
      build.py                  # compile LangGraph
    utils/
      timebox.py                # 60 seconds
      logging.py                # Clean trace（for Loom demo）
      deterministic.py          # Fix seed / Sorting / Hashing

  tests/
    test_impossible_job.py
    test_unstable_job.py
    test_hallucination_job.py
```
