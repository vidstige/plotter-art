//! Parallel lines as level sets of across + lift · depth. Near objects
//! push the lines aside without them ever crossing. Traced with marching
//! squares over pixel centers, then chained into polylines.
use std::collections::{HashMap, HashSet};

use crate::paper::Polyline;
use crate::parallel::Parallel;
use crate::raster::Raster;
use crate::vec2::{Vec2, vec2};

/// A crossing of line `.0` with grid edge `.1`.
type Crossing = (usize, usize);

struct Node {
    point: Vec2,
    links: Vec<Crossing>,
}

/// Grid edge ids: 2·i for the edge right of node i, 2·i + 1 below it.
fn cell_edges(w: usize, x: usize, y: usize) -> [usize; 4] {
    let i = y * w + x;
    [2 * i, 2 * (i + 1) + 1, 2 * (i + w), 2 * i + 1]
}

/// Crossings of all lines through the grid, linked to their neighbors.
fn crossings(phi: &[f64], w: usize, h: usize, frame: &Parallel) -> HashMap<Crossing, Node> {
    let node = |x: usize, y: usize| vec2(x as f64 + 0.5, y as f64 + 0.5);
    let mut nodes: HashMap<Crossing, Node> = HashMap::new();
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            let corners = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)];
            let values = corners.map(|(x, y)| phi[y * w + x]);
            let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let edges = cell_edges(w, x, y);
            for line in frame.between(lo, hi) {
                let c = frame.offset(line);
                let mut found = vec![];
                for e in (0..4).filter(|&e| (values[e] < c) != (values[(e + 1) % 4] < c)) {
                    let ((ax, ay), (bx, by)) = (corners[e], corners[(e + 1) % 4]);
                    let t = (c - values[e]) / (values[(e + 1) % 4] - values[e]);
                    let (a, b) = (node(ax, ay), node(bx, by));
                    let key = (line, edges[e]);
                    let point = a + (b - a) * t;
                    nodes.entry(key).or_insert(Node { point, links: vec![] });
                    found.push(key);
                }
                // saddles have four crossings; pairing them in order is
                // one of the two consistent choices
                for pair in found.chunks(2) {
                    nodes.get_mut(&pair[0]).unwrap().links.push(pair[1]);
                    nodes.get_mut(&pair[1]).unwrap().links.push(pair[0]);
                }
            }
        }
    }
    nodes
}

/// Walks linked crossings into polylines, open ends first so that open
/// contours are not split; whatever remains is closed loops.
fn chain(nodes: &HashMap<Crossing, Node>) -> Vec<Polyline> {
    let mut starts: Vec<(bool, Crossing)> =
        nodes.iter().map(|(k, n)| (n.links.len() != 1, *k)).collect();
    starts.sort();
    let mut visited = HashSet::new();
    let mut lines = vec![];
    for (_, start) in starts {
        if visited.contains(&start) {
            continue;
        }
        let mut line = vec![];
        let mut current = start;
        loop {
            visited.insert(current);
            line.push(nodes[&current].point);
            match nodes[&current].links.iter().find(|k| !visited.contains(k)) {
                Some(&next) => current = next,
                None => break,
            }
        }
        if line.len() > 2 && nodes[&current].links.contains(&start) {
            line.push(nodes[&start].point);
        }
        lines.push(line);
    }
    lines
}

/// Contours of across + `lift` · depth at the line offsets of `frame`.
pub fn lines(depth: &Raster, frame: &Parallel, lift: f64) -> Vec<Polyline> {
    let (w, h) = (depth.width, depth.height);
    let center = |i: usize| vec2((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
    let phi: Vec<f64> =
        (0..w * h).map(|i| frame.across(center(i)) + lift * depth.data[i] as f64).collect();
    chain(&crossings(&phi, w, h, frame))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bump(x: f64, y: f64) -> f64 {
        (-((x - 32.0).powi(2) + (y - 32.0).powi(2)) / 50.0).exp()
    }

    #[test]
    fn constant_depth_gives_straight_lines_shifted_up() {
        let flat = Raster::from_fn(64, 64, |_, _| 0.5);
        let frame = Parallel::new(flat.size(), 0.3, 8);
        let lines = lines(&flat, &frame, 4.0);
        assert_eq!(lines.len(), 8);
        for line in &lines {
            let t = frame.across(line[0]);
            assert!(line.iter().all(|&p| (frame.across(p) - t).abs() < 1e-9));
            let offset = (0..8).map(|i| frame.offset(i) - 2.0);
            assert!(offset.into_iter().any(|o| (o - t).abs() < 1e-9));
        }
    }

    #[test]
    fn bump_bends_lines_that_return_to_parallel() {
        let f = Raster::from_fn(64, 64, bump);
        let frame = Parallel::new(f.size(), 0.0, 8);
        let lines = lines(&f, &frame, 6.0);
        assert_eq!(lines.len(), 8);
        let bent = lines.iter().find(|l| l.iter().any(|p| (p.y - l[0].y).abs() > 2.0));
        let bent = bent.expect("some line bends around the bump");
        assert!((bent[0].y - bent.last().unwrap().y).abs() < 1e-3);
    }
}
