#!/usr/bin/env python3
"""
Generate flowy streamlines from a divergence-free 2D field
v = (∂ψ/∂y, −∂ψ/∂x), where ψ is classic 2D Perlin noise (single octave).
Outputs an A4 SVG with black strokes, uniform width — ideal for pen plotting.
"""

import math
import numpy as np
from pathlib import Path

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

# Perlin noise scale (mm per lattice unit). Larger -> broader features.
NOISE_SCALE_MM = 9.0

# Output simplification
DECIM_MM = 0.12          # min distance between consecutive output points
# ---------------------------------------------------------------

# Permutation table for Perlin; repeat to avoid masking at edges
rng = np.random.default_rng(SEED)
_perm = list(map(int, rng.permutation(256)))
_perm512 = _perm + _perm  # length 512

def fade(t: float) -> float:
    # 6t^5 - 15t^4 + 10t^3
    return t*t*t*(t*(t*6.0 - 15.0) + 10.0)

def dfade(t: float) -> float:
    # derivative: 30 t^2 (t-1)^2
    return 30.0 * t*t * (t - 1.0) * (t - 1.0)

def grad2(h: int) -> tuple[float, float]:
    """
    Hash to one of 8 gradients: (±1,±1), (±1,0), (0,±1)
    Returns a vector (gx, gy) (not normalized).
    """
    h &= 7
    if   h == 0: return ( 1.0,  1.0)
    elif h == 1: return (-1.0,  1.0)
    elif h == 2: return ( 1.0, -1.0)
    elif h == 3: return (-1.0, -1.0)
    elif h == 4: return ( 1.0,  0.0)
    elif h == 5: return (-1.0,  0.0)
    elif h == 6: return ( 0.0,  1.0)
    else:        return ( 0.0, -1.0)

def perlin_with_grad_unit(X: float, Y: float) -> tuple[float, float, float]:
    """
    Classic improved Perlin noise at lattice coordinates (X, Y),
    returning (value, ∂/∂X, ∂/∂Y).
    """
    xi = math.floor(X); yi = math.floor(Y)
    xf = X - xi; yf = Y - yi
    u  = fade(xf); v  = fade(yf)
    du = dfade(xf); dv = dfade(yf)

    # Hash corners
    A  = _perm512[xi & 255] + (yi & 255)
    AA = _perm512[A]
    AB = _perm512[A + 1]
    B  = _perm512[(xi + 1) & 255] + (yi & 255)
    BA = _perm512[B]
    BB = _perm512[B + 1]

    # Corner gradient vectors
    g00 = grad2(AA)
    g10 = grad2(BA)
    g01 = grad2(AB)
    g11 = grad2(BB)

    # Corner dot products (distance to corners)
    n00 = g00[0]*xf       + g00[1]*yf
    n10 = g10[0]*(xf-1.0) + g10[1]*yf
    n01 = g01[0]*xf       + g01[1]*(yf-1.0)
    n11 = g11[0]*(xf-1.0) + g11[1]*(yf-1.0)

    # Lerp along x, then y
    nx0   = n00 + u*(n10 - n00)               # (1-u)*n00 + u*n10
    nx1   = n01 + u*(n11 - n01)
    value = nx0 + v*(nx1 - nx0)               # (1-v)*nx0 + v*nx1

    # Derivatives wrt X
    dn00_dx = g00[0]; dn10_dx = g10[0]; dn01_dx = g01[0]; dn11_dx = g11[0]
    dnx0_dx = (1.0 - u)*dn00_dx + u*dn10_dx + du*(n10 - n00)
    dnx1_dx = (1.0 - u)*dn01_dx + u*dn11_dx + du*(n11 - n01)
    dval_dX = (1.0 - v)*dnx0_dx + v*dnx1_dx

    # Derivatives wrt Y
    dn00_dy = g00[1]; dn10_dy = g10[1]; dn01_dy = g01[1]; dn11_dy = g11[1]
    dnx0_dy = (1.0 - u)*dn00_dy + u*dn10_dy
    dnx1_dy = (1.0 - u)*dn01_dy + u*dn11_dy
    dval_dY = (1.0 - v)*dnx0_dy + v*dnx1_dy + dv*(nx1 - nx0)

    return value, dval_dX, dval_dY

def psi_with_grad_mm(x_mm: float, y_mm: float) -> tuple[float, float, float]:
    """
    Stream function ψ(x,y) = Perlin(X,Y) with X=x/NOISE_SCALE_MM, Y=y/NOISE_SCALE_MM.
    Returns (ψ, ∂ψ/∂x_mm, ∂ψ/∂y_mm).
    """
    X = x_mm / NOISE_SCALE_MM
    Y = y_mm / NOISE_SCALE_MM
    val, dX, dY = perlin_with_grad_unit(X, Y)
    scale = 1.0 / NOISE_SCALE_MM
    return val, dX * scale, dY * scale

def grad_psi(x_mm: float, y_mm: float) -> tuple[float, float]:
    _, gx, gy = psi_with_grad_mm(x_mm, y_mm)
    return gx, gy

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
