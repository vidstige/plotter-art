//! SVG serialization of polylines in millimeters.
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::paper::Polyline;
use crate::vec2::Vec2;

pub fn write(path: &Path, page: Vec2, lines: &[Polyline], stroke: f64) -> io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    let (w, h) = (page.x, page.y);
    writeln!(f, r#"<svg xmlns="http://www.w3.org/2000/svg""#)?;
    writeln!(f, r#"  width="{w}mm" height="{h}mm" viewBox="0 0 {w} {h}">"#)?;
    writeln!(f, r#"<g fill="none" stroke="black" stroke-width="{stroke}""#)?;
    writeln!(f, r#"  stroke-linecap="round" stroke-linejoin="round">"#)?;
    for line in lines {
        write!(f, r#"<polyline points=""#)?;
        for p in line {
            write!(f, "{:.2},{:.2} ", p.x, p.y)?;
        }
        writeln!(f, r#""/>"#)?;
    }
    writeln!(f, "</g>\n</svg>")?;
    f.flush()
}
