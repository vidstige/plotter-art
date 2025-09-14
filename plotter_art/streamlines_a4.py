#!/usr/bin/env python3
"""
Generate flowy streamlines from a divergence-free 2D field
v = (∂ψ/∂y, −∂ψ/∂x), where ψ is classic 2D Perlin noise (single octave).
Outputs an A4 SVG with black strokes, uniform width — ideal for pen plotting.
"""

import math
import numpy as np
from pathlib import Path

from plotter_art.perlin import grad_psi

# ----------------------- USER PARAMETERS -----------------------
SEED = 7                 # RNG seed for reproducibility
N_SEEDS = 350            # number of random seed points
WIDTH_MM = 210.0         # A4 size
HEIGHT_MM = 297.0
MARGIN_MM = 6.0          # keep lines away from edges
STEP_MM = 0.7            # arc-length step for integration (mm)
MAX_STEPS = 1000         # per direction
MIN_SPEED = 1e-4         # stop if |v| < MIN_SPEED
STROKE_WIDTH_MM = 0.25   # pen width in mm

# Output simplification
DECIM_MM = 0.12          # min distance between consecutive output points
# ---------------------------------------------------------------

def vfield(x_mm: float, y_mm: float) -> tuple[float, float]:
    # Stream function velocity (divergence-free): v = (∂ψ/∂y, -∂ψ/∂x)
    gx, gy = grad_psi(x_mm, y_mm)
    return (gy, -gx)

# --- Integration domain (A4 with margin) ---
xmin = MARGIN_MM; ymin = MARGIN_MM
xmax = WIDTH_MM - MARGIN_MM; ymax = HEIGHT_MM - MARGIN_MM

def in_bounds(x: float, y: float) -> bool:
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

def integrate_from_seed(x0: float, y0: float):
    if not in_bounds(x0, y0): return []
    poly = [(x0, y0)]
    # forward
    x, y = x0, y0
    for _ in range(MAX_STEPS):
        x, y, sp = rk2_step(x, y, STEP_MM, +1.0)
        if sp < MIN_SPEED or not in_bounds(x, y): break
        if math.hypot(x - poly[-1][0], y - poly[-1][1]) >= DECIM_MM:
            poly.append((x, y))
    # backward (prepend)
    x, y = x0, y0
    back = []
    for _ in range(MAX_STEPS):
        x, y, sp = rk2_step(x, y, STEP_MM, -1.0)
        if sp < MIN_SPEED or not in_bounds(x, y): break
        if not back or math.hypot(x - back[-1][0], y - back[-1][1]) >= DECIM_MM:
            back.append((x, y))
    back.reverse()
    return back + poly

def main() -> Path:
    # Random seeds in the page interior
    rng = np.random.default_rng(SEED)
    seeds_x = rng.uniform(xmin, xmax, size=N_SEEDS)
    seeds_y = rng.uniform(ymin, ymax, size=N_SEEDS)

    polylines = []
    for sx, sy in zip(seeds_x, seeds_y):
        line = integrate_from_seed(float(sx), float(sy))
        if len(line) >= 2:
            polylines.append(line)

    svg_path = Path("streamlines_a4.svg").absolute()
    with svg_path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_MM}mm" height="{HEIGHT_MM}mm" viewBox="0 0 {WIDTH_MM} {HEIGHT_MM}">\n')
        f.write(f'  <rect x="0" y="0" width="{WIDTH_MM}" height="{HEIGHT_MM}" fill="white"/>\n')
        f.write(f'  <g stroke="black" fill="none" stroke-width="{STROKE_WIDTH_MM}" stroke-linecap="round" stroke-linejoin="round">\n')
        for line in polylines:
            pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in line)
            f.write(f'    <polyline points="{pts}"/>\n')
        f.write('  </g>\n')
        f.write('</svg>\n')
    print(f"Generated {len(polylines)} streamlines.")
    print(f"Saved SVG to: {svg_path}")
    return svg_path

if __name__ == "__main__":
    main()
