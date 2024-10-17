from dataclasses import dataclass
import sys

from PIL import Image
import numpy as np


class Geometry:
    pass


@dataclass
class ImageGeometry(Geometry):
    image: Image
    

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
    gamma = compute_gamma(geometry, position);
    return np.einsum("kij,i,j", gamma, velocity, velocity)


def main():
    for path in sys.argv[1:]:
        print(path)
        geometry = ImageGeometry(image=Image.open(path))
        p = np.array([0.5, 0.5])
        v = np.array([0.1, 0.1])
        for _ in range(10):
            p = np.add(p, v)


main()
