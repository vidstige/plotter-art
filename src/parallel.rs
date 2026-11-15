//! A family of evenly spaced parallel lines covering a raster.
use std::ops::Range;

use crate::vec2::Vec2;

pub struct Parallel {
    center: Vec2,
    /// Unit direction along the lines.
    pub dir: Vec2,
    /// Half extent of the raster along the lines.
    pub along: f64,
    /// Half extent of the raster across the lines.
    across: f64,
    pub n: usize,
}

impl Parallel {
    /// `n` lines heading along `angle` (radians) over a raster of `size`.
    pub fn new(size: Vec2, angle: f64, n: usize) -> Parallel {
        let dir = Vec2::from_angle(angle);
        let half = size * 0.5;
        let along = (half.x * dir.x).abs() + (half.y * dir.y).abs();
        let across = (half.x * dir.y).abs() + (half.y * dir.x).abs();
        Parallel { center: half, dir, along, across, n }
    }

    /// Unit vector across the lines; down the page at angle zero.
    pub fn down(&self) -> Vec2 {
        self.dir.perp()
    }

    /// Across coordinate of line i, spread evenly over the raster.
    pub fn offset(&self, i: usize) -> f64 {
        self.across * (2.0 * (i as f64 + 0.5) / self.n as f64 - 1.0)
    }

    /// Indices of the lines with offsets in [lo, hi].
    pub fn between(&self, lo: f64, hi: f64) -> Range<usize> {
        let index = |t: f64| (t / self.across + 1.0) * self.n as f64 / 2.0 - 0.5;
        let first = index(lo).ceil().max(0.0) as usize;
        let end = (index(hi).floor() + 1.0).clamp(0.0, self.n as f64) as usize;
        first..end
    }

    /// Point at coordinate `u` along and `t` across the lines.
    pub fn point(&self, u: f64, t: f64) -> Vec2 {
        self.center + self.dir * u + self.down() * t
    }

    /// Across coordinate of p.
    pub fn across(&self, p: Vec2) -> f64 {
        (p - self.center).dot(self.down())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec2::vec2;

    fn tilted() -> Parallel {
        Parallel::new(vec2(40.0, 30.0), 0.3, 10)
    }

    #[test]
    fn offsets_are_symmetric_around_center() {
        let f = tilted();
        assert!((f.offset(0) + f.offset(9)).abs() < 1e-12);
    }

    #[test]
    fn across_inverts_point() {
        let f = tilted();
        assert!((f.across(f.point(3.0, -7.0)) + 7.0).abs() < 1e-12);
    }

    #[test]
    fn between_selects_offsets_in_range() {
        let f = tilted();
        let (lo, hi) = (f.offset(3) - 0.1, f.offset(6));
        assert_eq!(f.between(lo, hi), 3..7);
        assert!(f.between(-1e9, f.offset(0) - 0.1).is_empty());
        assert!(f.between(f.offset(9) + 0.1, 1e9).is_empty());
    }
}
