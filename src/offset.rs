//! Parallel lines lifted by depth, drawn front to back with a floating
//! horizon so that lifted lines hide the lines behind them.
use crate::paper::Polyline;
use crate::parallel::Parallel;
use crate::raster::Raster;

/// Each line is lifted against `frame.down()` by `lift` · depth. Lines
/// further down are in front: a point is drawn only where it rises above
/// every line drawn before it.
pub fn lines(depth: &Raster, frame: &Parallel, lift: f64, step: f64) -> Vec<Polyline> {
    let samples = (2.0 * frame.along / step) as usize + 1;
    let mut horizon = vec![f64::INFINITY; samples];
    let mut lines = vec![];
    for i in (0..frame.n).rev() {
        let t = frame.offset(i);
        let mut line = vec![];
        for (k, h) in horizon.iter_mut().enumerate() {
            let u = k as f64 * step - frame.along;
            let base = frame.point(u, t);
            let lifted = t - lift * depth.sample(base);
            if depth.contains(base) && lifted < *h {
                *h = lifted;
                line.push(frame.point(u, lifted));
            } else if !line.is_empty() {
                lines.push(std::mem::take(&mut line));
            }
        }
        lines.push(line);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bump(x: f64, y: f64) -> f64 {
        (-((x - 32.0).powi(2) + (y - 32.0).powi(2)) / 50.0).exp()
    }

    fn drawn(lines: &[Polyline]) -> Vec<&Polyline> {
        lines.iter().filter(|l| l.len() > 1).collect()
    }

    #[test]
    fn constant_depth_shifts_lines_uniformly() {
        let flat = Raster::from_fn(64, 64, |_, _| 0.5);
        let frame = Parallel::new(flat.size(), 0.0, 8);
        let lines = lines(&flat, &frame, 4.0, 0.5);
        let lines = drawn(&lines);
        assert_eq!(lines.len(), 8);
        for (line, i) in lines.iter().zip((0..8).rev()) {
            let expected = frame.offset(i) - 2.0;
            assert!(line.iter().all(|&p| (frame.across(p) - expected).abs() < 1e-9));
        }
    }

    #[test]
    fn lines_return_to_base_offset_past_the_bump() {
        let f = Raster::from_fn(64, 64, bump);
        let frame = Parallel::new(f.size(), 0.0, 8);
        let lines = lines(&f, &frame, 6.0, 0.5);
        let middle = drawn(&lines)[3];
        let (first, last) = (middle[0], *middle.last().unwrap());
        assert!((first.y - last.y).abs() < 1e-6);
        assert!(middle.iter().any(|p| p.y < first.y - 3.0));
    }

    #[test]
    fn tall_bump_hides_part_of_the_line_behind() {
        let f = Raster::from_fn(64, 64, bump);
        let frame = Parallel::new(f.size(), 0.0, 8);
        let lines = lines(&f, &frame, 30.0, 0.5);
        assert!(drawn(&lines).len() > 8);
    }
}
