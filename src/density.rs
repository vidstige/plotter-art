//! Random lines that bend toward dark areas, so that line density
//! follows the darkness of an image.
use crate::paper::Polyline;
use crate::raster::Raster;
use crate::rng::Rng;
use crate::vec2::{Vec2, vec2};

pub struct Params {
    pub lines: usize,
    /// Integration step in pixels.
    pub step: f64,
    /// Maximum line length in pixels.
    pub length: f64,
    /// Turning rate per unit of darkness gradient.
    pub pull: f64,
    /// Max random deviation from heading at the center, in radians.
    pub spread: f64,
}

/// Traces from p along unit v, turning toward increasing darkness and
/// stopping at the raster edge.
pub fn trace(darkness: &Raster, mut p: Vec2, mut v: Vec2, params: &Params) -> Polyline {
    let mut line = vec![p];
    for _ in 0..(params.length / params.step) as usize {
        let g = darkness.gradient(p);
        // only the part across the heading turns; the rest would change speed
        let turn = (g - v * g.dot(v)) * params.pull;
        v = (v + turn * params.step).normalized();
        p += v * params.step;
        if !darkness.contains(p) {
            break;
        }
        line.push(p);
    }
    line
}

/// Lines from uniformly random points, heading roughly toward the center.
pub fn lines(darkness: &Raster, params: &Params, rng: &mut Rng) -> Vec<Polyline> {
    let size = darkness.size();
    let center = size * 0.5;
    (0..params.lines)
        .map(|_| {
            let p = vec2(rng.range(0.0, size.x), rng.range(0.0, size.y));
            let to_center = center - p;
            let heading = to_center.y.atan2(to_center.x);
            let angle = heading + rng.range(-params.spread, params.spread);
            trace(darkness, p, Vec2::from_angle(angle), params)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Params {
        Params { lines: 10, step: 0.5, length: 100.0, pull: 50.0, spread: 0.5 }
    }

    fn raster(width: usize, height: usize, f: impl Fn(usize) -> f32) -> Raster {
        Raster { width, height, data: (0..width * height).map(|i| f(i % width)).collect() }
    }

    #[test]
    fn uniform_darkness_gives_straight_lines() {
        let flat = raster(50, 50, |_| 0.5);
        let line = trace(&flat, vec2(1.0, 10.0), vec2(1.0, 0.0), &params());
        assert!(line.iter().all(|p| (p.y - 10.0).abs() < 1e-9));
    }

    #[test]
    fn lines_turn_toward_dark_side() {
        let dark_right = raster(50, 50, |x| x as f32 / 50.0);
        let line = trace(&dark_right, vec2(25.0, 1.0), vec2(0.0, 1.0), &params());
        assert!(line.last().unwrap().x > 26.0);
    }

    #[test]
    fn same_seed_gives_same_lines() {
        let flat = raster(50, 50, |_| 0.5);
        let run = || lines(&flat, &params(), &mut Rng::new(7));
        assert_eq!(run(), run());
    }
}
