import sys
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
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


def grayscale(im: np.ndarray) -> np.ndarray:
    return np.mean(im, axis=-1)


class DepthImageGeometry(Geometry):
    """Geometry where x=u, y=v, z=depth(u, v)"""
    def __init__(self, image: Image) -> None:
        self.image = grayscale(np.array(image))
        # pre-compute compute jacobians
        self.dzdu, self.dzdv = np.gradient(self.image)
        # pre-compute compute hessians
        self.dzdu2, self.dzdudv = np.gradient(self.dzdu)
        self.dzdvdu, self.dzdv2 = np.gradient(self.dzdv)

    def evaluate(self, uv: np.ndarray) -> np.array:
        # x = u, y = v, z = f(u, v)
        u, v = np.hsplit(uv, 2)
        h, w = self.image.shape
        ui = np.clip(np.astype(u, np.int32), 0, w - 1)
        vi = np.clip(np.astype(v, np.int32), 0, h - 1)
        return np.hstack([uv, self.image[vi, ui]])

    def jacobian(self, uv: jnp.ndarray) -> jnp.ndarray:
        u, v = uv
        h, w = self.image.shape
        ui = np.clip(np.astype(u, np.int32), 0, w - 1)
        vi = np.clip(np.astype(v, np.int32), 0, h - 1)
        return np.array([            
            [1, 0],  # dx/du, dx/dv
            [0, 1],  # dy/du, dy/dv
            [
                self.dzdu[vi, ui],  # dz/du
                self.dzdv[vi, ui],  # dz/dv
             ],
        ])

    def hessian(self, uv: jnp.ndarray) -> jnp.ndarray:
        # d2: shape (3, 2, 2)
        u, v = uv
        h, w = self.image.shape
        ui = np.clip(int(u), 0, w - 1)
        vi = np.clip(int(v), 0, h - 1)
        return np.array([
            [
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
            ],
            [
                [self.dzdu2[vi, ui], self.dzdudv[vi, ui]],  # dz/du2, dz/dudv
                [self.dzdvdu[vi, ui], self.dzdv2[vi, ui]],  # dz/dvdu, dz/dv2
            ],
        ])


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
    return np.einsum('lij,lk,mk->mij', d2, d, inverse_metric)


def acceleration(geometry: Geometry, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Computes acceleration needed to stay on geodesic"""
    gamma = compute_gamma(geometry, position)
    return np.einsum("kij,i,j", gamma, velocity, velocity)


def trace(geometry: Geometry, p: np.ndarray, v: np.ndarray, length: float, dt: float) -> Iterable[np.ndarray]:
    # euler integration of geodesic equation
    for _ in range(int(length / dt)):
        yield p
        a = acceleration(geometry, p, v)
        if np.linalg.norm(a) > 10:
            return
        p = p + v * dt
        v = v + a * dt


def main():
    rr.init("plotter-art", spawn=True)

    #geometry = Bump()

    im = Image.open("depth/per.png")
    geometry = DepthImageGeometry(im)
    gw, gh = 256, 256
    w, h = im.width, im.height
    xx, yy = np.meshgrid(w*(1/gw * np.arange(gw)), h*(1/gh * np.arange(gh)))

    #gw, gh = 20, 20
    #w, h = 5, 5
    #xx, yy = np.meshgrid(w*(1/gw * np.arange(gw) - 0.5), h*(1/gh * np.arange(gh) - 0.5))

    points = np.vstack([xx.ravel(), yy.ravel()]).T
    positions = geometry.evaluate(points)
    rr.log(
        "hole",
        rr.Points3D(positions, radii=1)
    )

    n = 10
    ps = np.vstack([-0.5*w*np.ones(n), np.linspace(0, h, n)]).T
    vs = np.vstack([np.ones(n), np.zeros(n)]).T
    vs = vs / np.expand_dims(np.linalg.norm(vs, axis=-1), axis=-1)

    #print(geometry.jacobian(ps[0]))
    #print(geometry.hessian(ps[0]))

    line_strips = []
    for p, v in zip(ps, vs):
        curve = trace(geometry, p, v, w, 0.25)
        line_strip = [geometry.evaluate(uv) for uv in curve]
        line_strips.append(line_strip)
    rr.log("strip", rr.LineStrips3D(line_strips))


main()
