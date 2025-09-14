import math
import random

# Perlin noise scale (mm per lattice unit). Larger -> broader features.
SEED = 7                 # RNG seed for reproducibility
NOISE_SCALE_MM = 9.0

# Permutation table for Perlin; repeat to avoid masking at edges
rng = random.Random(SEED)
_perm = list(range(256))
rng.shuffle(_perm)
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
