# Documentation Template: CHT Simulation Schema Definition

## 1. Overview
**Date:** 2026-01-13
**Status:** Completed
**Author:** Oscar Shih
This document describes the implementation of a rigorous Pydantic schema (`cht_sim.py`) for configuring OpenFOAM CHT simulations. It provides a structured, type-safe interface for generating valid OpenFOAM dictionaries (`controlDict`, `fvSchemes`, `fvSolution`, `thermophysicalProperties`).

## 2. Motivation
Generating OpenFOAM configuration files via LLMs or manual scripts is error-prone. Common issues include:
- Missing required solver entries in `fvSolution` (causing immediate crashes).
- Invalid time control settings (e.g., negative deltaT).
- Inconsistent region definitions between files.
- Incorrect data types for physical properties.

This schema solves these problems by:
1.  **Enforcing structural validity** through strict Pydantic models.
2.  **Preventing runtime crashes** by automatically injecting fail-safe defaults (e.g., required linear solvers).
3.  **Standardizing input** for the MVP phase (e.g., locking numerical schemes to a robust preset).

## 3. Technical Implementation

### 3.1. Schema Changes
The configuration is modeled as a hierarchical JSON-like structure with specific validation rules.

**Core Structure (`CHTSimulation`):**
- **System**:
    - `controlDict`: Global time settings.
    - `Regions`: Nested dictionaries for `fluids` and `solids`, containing `fvSchemes` and `fvSolution` per region.
- **Constant**:
    - `Regions`: Nested dictionaries for `fluids` and `solids`, containing `thermophysicalProperties`.
- **Time** (Optional/Injected):
    - `Regions`: Initial fields (`T`, `p_rgh`) for all regions.

**Key Models:**
- `ControlDict`: Validates `startTime`, `endTime`, `deltaT`.
- `FvSchemes`: Enforces `preset="robust/upwind"`.
- `FvSolution`: Auto-injects solvers for `T` and `p_rgh` if missing.
- `ThermophysicalProperties`: Validates physical consistency (e.g., positive molecular weight).

**Example JSON Payload:**
```json
{
  "System": {
    "controlDict": ["controlDict", { "solverName": "chtMultiRegionFoam", "endTime": 10.0, "deltaT": 0.01, "writeInterval": 100 }],
    "Regions": {
      "fluids": { "fluid_1": { "fvSchemes": ["fvSchemes", {}], "fvSolution": ["fvSolution", {}] } },
      "solids": { "solid_1": { "fvSchemes": ["fvSchemes", {}], "fvSolution": ["fvSolution", {}] } }
    }
  },
  "Constant": {
    "Regions": {
      "fluids": { "fluid_1": { "thermophysicalProperties": ["thermophysicalProperties", { ... }] } },
      "solids": { "solid_1": { "thermophysicalProperties": ["thermophysicalProperties", { ... }] } }
    }
  }
}
```

### 3.2. Logic & Algorithm Changes
- **Auto-Injection Mechanism**:
    - `_parse_and_inject` validator in `CHTSimulation`:
        - Iterates through all defined fluid and solid regions.
        - Checks `fvSolution`: If specific solvers (`T`, `p_rgh`) are missing, it injects robust defaults (`PBiCGStab`, `PCG`).
        - Checks/Creates `Time`: Generates the `0/` directory structure and injects default initial conditions (`T=298.15K`, `p_rgh=0Pa`).
- **Strict Validation**:
    - `_assert_node_shape`: Ensures all file nodes follow the `["TypeName", {kwargs}]` format.
    - `StrictModel`: Base class that forbids extra fields to catch typos.

### 3.3. Dependencies
- **pydantic**: Used for data validation and modeling (Core dependency).
- **typing**: Standard library for type hinting.

## 4. Usage / Integration

**Python Integration:**
```python
from cht_sim import CHTSimulation

# 1. Define configuration dictionary
config_data = { ... } # see Example JSON above

# 2. Validate and Instantiate
try:
    simulation = CHTSimulation.model_validate(config_data)
    simulation.check() # Run semantic checks
except ValueError as e:
    print(f"Invalid Configuration: {e}")
```

**CLI Usage (Test Script):**
```bash
python cht_sim.py
```
*Output will confirm successful injection of defaults and validation of the example case.*

## 5. Testing & Validation
- **Unit Validation**: The `cht_sim.py` script includes a `__main__` block that:
    1.  Constructs a minimal valid payload with empty solver settings.
    2.  Instantiates the model.
    3.  Asserts that `fvSolution` now contains the injected solver stanzas.
    4.  Asserts that `Time` dictionary has been populated with `T` and `p_rgh` fields.
    5.  Verifies global pressure reference values.
- **Negative Testing (Implicit)**: The strict schema definitions automatically reject invalid inputs (e.g., negative temperature defaults, invalid schemes) during Pydantic validation.

## 6. Screenshots / Visuals
N/A (Backend logic).

## 7. Future Work / Known Limitations
- **Boundary Conditions**: The current schema only models internal fields (`internalField`). Boundary conditions (`boundaryField`) are omitted for the MVP and must be handled by the writer or future schema extensions.
- **Scheme Flexibility**: `fvSchemes` is currently hardcoded. Future versions should allow users to specify custom schemes safely.
- **Solver Options**: `fvSolution` defaults are conservative. Exposing more solver parameters (e.g., smoother types, pre-sweeps) would be a logical next step.
