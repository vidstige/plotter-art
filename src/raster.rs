//! Grayscale raster with bilinear sampling, blurring and downscaling.
use crate::vec2::{Vec2, vec2};

#[derive(Clone, Debug, PartialEq)]
pub struct Raster {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>, // width * height, row major
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

impl Raster {
    pub fn size(&self) -> Vec2 {
        vec2(self.width as f64, self.height as f64)
    }

    pub fn contains(&self, p: Vec2) -> bool {
        let s = self.size();
        p.x >= 0.0 && p.y >= 0.0 && p.x < s.x && p.y < s.y
    }

    fn at(&self, x: isize, y: isize) -> f64 {
        let x = x.clamp(0, self.width as isize - 1) as usize;
        let y = y.clamp(0, self.height as isize - 1) as usize;
        self.data[y * self.width + x] as f64
    }

    /// Bilinear sample with pixel centers at half-integer coordinates,
    /// clamped at the edges.
    pub fn sample(&self, p: Vec2) -> f64 {
        let (x, y) = (p.x - 0.5, p.y - 0.5);
        let (i, j) = (x.floor() as isize, y.floor() as isize);
        let (fx, fy) = (x - x.floor(), y - y.floor());
        let top = lerp(self.at(i, j), self.at(i + 1, j), fx);
        let bottom = lerp(self.at(i, j + 1), self.at(i + 1, j + 1), fx);
        lerp(top, bottom, fy)
    }

    /// Central difference gradient, one pixel each way.
    pub fn gradient(&self, p: Vec2) -> Vec2 {
        let f = |dx, dy| self.sample(p + vec2(dx, dy));
        vec2(f(1.0, 0.0) - f(-1.0, 0.0), f(0.0, 1.0) - f(0.0, -1.0)) * 0.5
    }

    pub fn map(&self, f: impl Fn(f32) -> f32) -> Raster {
        let data = self.data.iter().map(|&v| f(v)).collect();
        Raster { data, ..*self }
    }

    fn transpose(&self) -> Raster {
        let (w, h) = (self.width, self.height);
        let data = (0..w * h).map(|i| self.data[(i % h) * w + i / h]);
        Raster { width: h, height: w, data: data.collect() }
    }

    /// Box blur along rows; the window shrinks at the edges.
    fn box_rows(&self, radius: usize) -> Raster {
        let w = self.width;
        let mut data = Vec::with_capacity(self.data.len());
        for row in self.data.chunks(w) {
            let sums = row.iter().scan(0.0, |s, &v| {
                *s += v as f64;
                Some(*s)
            });
            let prefix: Vec<f64> = std::iter::once(0.0).chain(sums).collect();
            for x in 0..w {
                let (a, b) = (x.saturating_sub(radius), (x + radius + 1).min(w));
                data.push(((prefix[b] - prefix[a]) / (b - a) as f64) as f32);
            }
        }
        Raster { data, ..*self }
    }

    /// Approximate Gaussian blur from three box blurs per axis.
    pub fn blur(&self, radius: usize) -> Raster {
        let mut r = self.clone();
        for _ in 0..6 {
            r = r.box_rows(radius).transpose();
        }
        r
    }

    /// Averages n×n blocks so the longest side is at most `max_side`.
    pub fn shrink_to(&self, max_side: usize) -> Raster {
        let n = self.width.max(self.height).div_ceil(max_side);
        let (w, h) = (self.width / n, self.height / n);
        let block = |i: usize| {
            let (x, y) = (i % w * n, i / w * n);
            let rows = (y..y + n).map(|y| &self.data[y * self.width + x..][..n]);
            rows.flatten().sum::<f32>() / (n * n) as f32
        };
        Raster { width: w, height: h, data: (0..w * h).map(block).collect() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(width: usize, height: usize) -> Raster {
        let data = (0..width * height).map(|i| (i % 7) as f32).collect();
        Raster { width, height, data }
    }

    fn mean(r: &Raster) -> f32 {
        r.data.iter().sum::<f32>() / r.data.len() as f32
    }

    #[test]
    fn sample_at_pixel_center_is_pixel_value() {
        let r = ramp(5, 3);
        assert_eq!(r.sample(vec2(2.5, 1.5)), r.data[7] as f64);
    }

    #[test]
    fn sample_halfway_between_centers_is_average() {
        let r = ramp(5, 3);
        let expected = (r.data[0] + r.data[1]) as f64 / 2.0;
        assert_eq!(r.sample(vec2(1.0, 0.5)), expected);
    }

    #[test]
    fn transpose_twice_is_identity() {
        let r = ramp(5, 3);
        assert_eq!(r.transpose().transpose(), r);
    }

    #[test]
    fn blur_keeps_constant_raster_constant() {
        let r = ramp(6, 4).map(|_| 0.25);
        assert_eq!(r.blur(2), r);
    }

    #[test]
    fn blur_flattens_variation() {
        let r = ramp(40, 30);
        let spread = |r: &Raster| {
            r.data.iter().fold(0.0f32, |m, v| m.max((v - mean(r)).abs()))
        };
        assert!(spread(&r.blur(3)) < spread(&r) / 4.0);
    }

    #[test]
    fn shrink_preserves_mean_of_whole_blocks() {
        let r = ramp(12, 6);
        let small = r.shrink_to(4);
        assert_eq!((small.width, small.height), (4, 2));
        assert!((mean(&small) - mean(&r)).abs() < 1e-5);
    }
}
