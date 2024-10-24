from dataclasses import dataclass
import sys
from typing import Iterable

import jax
import jax.numpy as jnp
from PIL import Image
import rerun as rr


class Geometry:
    def evaluate(self, uv: jnp.ndarray) -> jnp.ndarray:
        """Computes 3d-position for a uv position"""
        del uv
        raise NotImplemented
    
    def jacobian(self, uv: jnp.ndarray) -> jnp.ndarray:
        """Computes the jacobian (all partial differentials) at (u,v). It's 3x2
        matrix with one row for each output cordinate and one colmn for each input cordinate"""
        del uv
        raise NotImplemented

    def hessian(self, uv: jnp.ndarray) -> jnp.ndarray:
        """Evaluates the hessian (all 2nd order partial differentials) at (u, v)"""
        del uv
        raise NotImplemented


class AutoDiffMixin:
    """Provides automatically computed differentials"""
    def __init__(self) -> None:
        self.jacobian = jax.jacrev(self.evaluate)
        self.hessian = jax.jacfwd(self.jacobian)


#@dataclass
#class DepthImageGeometry(Geometry):
#    """Geometry where x=u, y=v, z=depth(u, v)"""
#    image: Image
#
#    def evaluate(self, position: np.ndarray) -> float:
#        return 

class ZGeometry(Geometry):
    def evaluate(self, uv: jnp.ndarray) -> jnp.ndarray:
        # x = u, y = v, z = f(u, v)
        return jnp.hstack([uv, jnp.expand_dims(self.z(uv), axis=-1)])
    
    def z(self, uv: jnp.ndarray) -> float:
        raise NotImplemented


class Hole(ZGeometry, AutoDiffMixin):
    def z(self, uv: jnp.ndarray) -> float:
        return -1.0 / jnp.linalg.norm(uv, axis=-1)


class Bump(ZGeometry, AutoDiffMixin):
    def z(self, uv: jnp.ndarray) -> float:
        r = jnp.linalg.norm(uv, axis=-1)
        return jnp.cos(r)


def evaluate_metric(geometry: Geometry, uv: jnp.ndarray) -> jnp.ndarray:
    # compute d/du and d/dv
    jacobian = geometry.jacobian(uv)
    # metric is just all the dot products of the three combinations
    # of the two partial derivatives
    return np.dot(jacobian.T, jacobian)


def compute_gamma(geometry: Geometry, p: jnp.ndarray) -> jnp.ndarray:
    """Evaluates Christoffer Symbol of geometry at position"""
    metric = evaluate_metric(geometry, p)
    inverse_metric = jnp.linalg.inv(metric)

    # compute all first order partial differentials at p, dx/du, dy/du, and so on
    d = geometry.jacobian(p)

    # compute all second order partial differentials, d²x/du², d²/dudv, d²x/du², d^2y/du^2 and so on
    d2 = geometry.hessian(p)

    # compute tensor product gamma^k_ij = (d²R/du^i du^j) * (dR/du^l) * (g^-1)^lk
    # the index l is thus summed over
    # TODO: use einsum instead
    tmp = np.zeros([2, 2, 2])
    for k in range(2):
        for i in range(2):
            for j in range(2):
                for l in range(2):
                    tmp[k][i][j] += np.dot(d2[:, i, j], d[:, l]) * inverse_metric[l, k]
                    #tmp[k][i][j] += d2[i][j].dot(&d[l]) * inverse_metric[(l, k)];
    return tmp

import numpy as np
def acceleration(geometry: Geometry, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Computes acceleration needed to stay on geodesic"""
    gamma = compute_gamma(geometry, position)
    return np.einsum("kij,i,j", gamma, velocity, velocity)


def trace(geometry: Geometry, p: np.ndarray, v: np.ndarray, length: float, dt: float) -> Iterable[np.ndarray]:
    # euler integration of geodesic equation
    for _ in range(int(length / dt)):
        yield p
        a = acceleration(geometry, p, v)
        p = p + v * dt
        v = v + a * dt


def main():
    rr.init("plotter-art", spawn=True)

    geometry = Bump()
    gw, gh = 20, 20
    w, h = 5, 5
    xx, yy = np.meshgrid(w*(1/gw * np.arange(gw) - 0.5), h*(1/gh * np.arange(gh) - 0.5))

    points = np.vstack([xx.ravel(), yy.ravel()]).T
    positions = geometry.evaluate(points)
    rr.log(
        "hole",
        rr.Points3D(positions, radii=0.01)
    )

    n = 20
    ps = np.vstack([-0.5*w*np.ones(n), np.linspace(-0.5*h, 0.5*h, n)]).T
    vs = np.vstack([np.ones(n), np.zeros(n)]).T
    vs = vs / np.expand_dims(np.linalg.norm(vs, axis=-1), axis=-1)

    line_strips = []
    for p, v in zip(ps, vs):
        curve = trace(geometry, p, v, w, 0.25)
        line_strip = [geometry.evaluate(uv) for uv in curve]
        line_strips.append(line_strip)
    rr.log("strip", rr.LineStrips3D(line_strips))


main()
