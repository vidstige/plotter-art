#!/usr/bin/env python3
"""
Generate flowy streamlines from a divergence-free 2D field
v = (∂ψ/∂y, −∂ψ/∂x), where ψ is classic 2D Perlin noise (single octave).
Outputs an A4 SVG with black strokes, uniform width — ideal for pen plotting.
"""

import math
import random
from pathlib import Path

from plotter_art.paper import Paper, save_svg
from plotter_art.perlin import grad_psi

# ----------------------- USER PARAMETERS -----------------------
SEED = 7                 # RNG seed for reproducibility
N_SEEDS = 350            # number of random seed points
A4_SIZE = (297.0, 210.0)
PAPER = Paper(A4_SIZE)
MARGIN_MM = 6.0          # keep lines away from edges
STEP_MM = 0.7            # arc-length step for integration (mm)
MAX_STEPS = 1000         # per direction
MIN_SPEED = 1e-4         # stop if |v| < MIN_SPEED

# Output simplification
DECIM_MM = 0.12          # min distance between consecutive output points
# ---------------------------------------------------------------

def vfield(x_mm: float, y_mm: float) -> tuple[float, float]:
    # Stream function velocity (divergence-free): v = (∂ψ/∂y, -∂ψ/∂x)
    gx, gy = grad_psi(x_mm, y_mm)
    return (gy, -gx)

# --- Integration domain (A4 with margin) ---
def in_bounds(x: float, y: float) -> bool:
    xmin = MARGIN_MM; ymin = MARGIN_MM
    xmax = PAPER.width - MARGIN_MM; ymax = PAPER.height - MARGIN_MM
    return (xmin <= x <= xmax) and (ymin <= y <= ymax)

def rk2_step(x: float, y: float, step: float, direction: float):
    # Midpoint (RK2) with unit-speed stepping
    vx, vy = vfield(x, y)
    s = math.hypot(vx, vy)
    if s < MIN_SPEED: return x, y, s
    nx, ny = vx / s, vy / s
    mx, my = x + 0.5 * step * direction * nx, y + 0.5 * step * direction * ny
    mvx, mvy = vfield(mx, my)
    ms = math.hypot(mvx, mvy)
    if ms < MIN_SPEED: return x, y, ms
    mnx, mny = mvx / ms, mvy / ms
    return x + step * direction * mnx, y + step * direction * mny, ms

def integrate_from_seed(x0: float, y0: float, paper: Paper):
    if not in_bounds(x0, y0): return []
    poly = [(x0, y0)]
    x, y = x0, y0
    for _ in range(MAX_STEPS):
        x, y, sp = rk2_step(x, y, STEP_MM, +1.0)
        if sp < MIN_SPEED or not in_bounds(x, y): break
        if math.hypot(x - poly[-1][0], y - poly[-1][1]) >= DECIM_MM:
            poly.append((x, y))
    return poly

def main():
    # Random seeds in the page interior
    rng = random.Random(SEED)
    xmin = MARGIN_MM; ymin = MARGIN_MM
    xmax = PAPER.width - MARGIN_MM; ymax = PAPER.height - MARGIN_MM
    seeds_x = [rng.uniform(xmin, xmax) for _ in range(N_SEEDS)]
    seeds_y = [rng.uniform(ymin, ymax) for _ in range(N_SEEDS)]

    paper = PAPER
    for sx, sy in zip(seeds_x, seeds_y):
        line = integrate_from_seed(sx, sy, paper)
        if len(line) >= 2:
            paper.add_polyline(line)

    save_svg(Path("output") / "streamlines.svg", paper)


if __name__ == "__main__":
    main()
