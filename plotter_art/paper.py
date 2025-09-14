from dataclasses import dataclass
from pathlib import Path


Polyline = list[tuple[float, float]]


class Paper:
    def __init__(self, size: tuple[float, float]):
        self.size = size
        self.polylines = []
    
    def add_polyline(self, polyline: Polyline):
        self.polylines.append(polyline)

    @property
    def width(self) -> float:
        return self.size[0]

    @property
    def height(self) -> float:
        return self.size[1]


def save_svg(path: Path, paper: Paper):
    STROKE_WIDTH_MM = 0.25   # pen width in mm
    with path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{paper.width}mm" height="{paper.height}mm" viewBox="0 0 {paper.width} {paper.height}">\n')
        f.write(f'  <rect x="0" y="0" width="{paper.width}" height="{paper.height}" fill="white"/>\n')
        f.write(f'  <g stroke="black" fill="none" stroke-width="{STROKE_WIDTH_MM}" stroke-linecap="round" stroke-linejoin="round">\n')
        for line in paper.polylines:
            pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in line)
            f.write(f'    <polyline points="{pts}"/>\n')
        f.write('  </g>\n')
        f.write('</svg>\n')
    print(f"Generated {len(paper.polylines)} streamlines.")
    print(f"Saved SVG to: {path}")
