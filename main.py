from dataclasses import dataclass
import sys

import numpy as np
from PIL import Image
import rerun as rr


class Geometry:
    pass


#@dataclass
#class DepthImageGeometry(Geometry):
#    """Geometry where x=u, y=v, z=depth(u, v)"""
#    image: Image
#
#    def evaluate(self, position: np.ndarray) -> float:
#        return 

class ZGeometry(Geometry):
    def evaluate(self, uv: np.ndarray) -> np.ndarray:
        """Computes 3d-position for a uv position"""
        # x = u, y = v, z = f(u, v)
        return np.hstack([uv, np.expand_dims(self.z(uv), axis=-1)])
    
    def z(self, uv: np.ndarray) -> float:
        raise NotImplemented


class Hole(ZGeometry):
    def z(self, uv: np.ndarray) -> float:
        return -1.0 / np.linalg.norm(uv, axis=-1)


class Bump(ZGeometry):
    def z(self, uv: np.ndarray) -> float:
        return np.cos(np.linalg.norm(uv, axis=-1))


def compute_gamma(geometry: Geometry, position: np.ndarray) -> np.ndarray:
    """Evaluates Christoffer Symbol of geometry at position"""
    return np.array([
        [
            [1, 2],
            [3, 4],
        ],
        [
            [5, 6],
            [7, 8],
        ],
    ])


def acceleration(geometry: Geometry, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Computes acceleration needed to stay on geodesic"""
    gamma = compute_gamma(geometry, position);
    return np.einsum("kij,i,j", gamma, velocity, velocity)


def main():
    rr.init("plotter-art", spawn=True)

    geometry = Bump()
    #p = np.array([0.5, 0.5])
    #v = np.array([0.1, 0.1])
    gw, gh = 20, 20
    w, h = 5, 5
    xx, yy = np.meshgrid(w*(1/gw * np.arange(gw) - 0.5), h*(1/gh * np.arange(gh) - 0.5))

    points = np.vstack([xx.ravel(), yy.ravel()]).T
    positions = geometry.evaluate(points)
    rr.log(
        "hole",
        rr.Points3D(positions, radii=0.01)
    )

main()
