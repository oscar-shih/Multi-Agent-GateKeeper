# src/gatekeeper/tools/cfl.py
from __future__ import annotations
import math
from typing import Tuple

def compute_cfl_from_handbook(velocity_m_per_s: float, time_step_s: float, characteristic_length_m: float, cell_count: float) -> Tuple[float, float]:
    if cell_count <= 0:
        raise ValueError(f"cell_count must be > 0, got {cell_count}")
    if characteristic_length_m <= 0:
        raise ValueError(f"characteristic_length_m must be > 0, got {characteristic_length_m}")
    if time_step_s < 0:
        raise ValueError(f"time_step_s must be >= 0, got {time_step_s}")

    dx_eff = characteristic_length_m / (cell_count ** (1.0 / 3.0))
    if dx_eff <= 0:
        raise ValueError(f"dx_eff computed <= 0, got {dx_eff}")

    courant = (velocity_m_per_s * time_step_s) / dx_eff
    return courant, dx_eff

def compute_courant_number(u_m_per_s: float, dt_s: float, dx_m: float) -> float:
    if dx_m <= 0:
        raise ValueError(f"dx_m must be > 0, got {dx_m}")
    if dt_s < 0:
        raise ValueError(f"dt_s must be >= 0, got {dt_s}")
    # C = u * dt / dx
    return (u_m_per_s * dt_s) / dx_m