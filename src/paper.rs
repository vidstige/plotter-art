//! Paper sizes and placing image space polylines on the page.
use crate::vec2::{Vec2, vec2};

pub type Polyline = Vec<Vec2>;

/// Portrait A4 in millimeters.
pub const A4: Vec2 = vec2(210.0, 297.0);

/// Turns `paper` to match the image orientation, then scales and
/// centers the lines from image pixels to millimeters within `margin`.
/// Returns the page size and the placed lines.
pub fn place(
    image: Vec2,
    paper: Vec2,
    margin: f64,
    lines: &[Polyline],
) -> (Vec2, Vec<Polyline>) {
    let turn = (image.x > image.y) != (paper.x > paper.y);
    let page = if turn { vec2(paper.y, paper.x) } else { paper };
    let fit = page - vec2(2.0, 2.0) * margin;
    let scale = (fit.x / image.x).min(fit.y / image.y);
    let offset = (page - image * scale) * 0.5;
    let placed = lines
        .iter()
        .filter(|l| l.len() > 1)
        .map(|l| l.iter().map(|&p| offset + p * scale).collect())
        .collect();
    (page, placed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn landscape_image_turns_paper_and_is_centered() {
        let image = vec2(200.0, 100.0);
        let corners = vec![vec![vec2(0.0, 0.0), image]];
        let (page, placed) = place(image, A4, 10.0, &corners);
        assert_eq!(page, vec2(297.0, 210.0));
        let [a, b] = placed[0][..] else { panic!() };
        assert_eq!(a.x, 10.0);
        assert_eq!(b.x, 287.0);
        assert_eq!(a.y + b.y, page.y);
    }

    #[test]
    fn single_points_are_dropped() {
        let lines = vec![vec![vec2(1.0, 1.0)]];
        let (_, placed) = place(vec2(10.0, 10.0), A4, 0.0, &lines);
        assert!(placed.is_empty());
    }
}
