//! Parallel lines lifted by a depth map, with hidden lines removed.
use std::io;
use std::path::Path;

use plotter_art::args::Args;
use plotter_art::parallel::Parallel;
use plotter_art::{offset, paper, png, svg};

const USAGE: &str = "relief <depth.png> <out.svg> \
[--lift 0.05] [--lines 120] [--angle 0] [--blur 4]";
const WORK_SIZE: usize = 1024;
const MARGIN_MM: f64 = 15.0;
const STROKE_MM: f64 = 0.3;
const STEP: f64 = 0.5;

fn main() -> io::Result<()> {
    let mut args = Args::parse(std::env::args().skip(1));
    let input = args.positional(0).expect(USAGE).to_string();
    let output = args.positional(1).expect(USAGE).to_string();
    // displacement of white relative to black, as a fraction of image width
    let lift: f64 = args.flag("lift", 0.05);
    let n = args.flag("lines", 120);
    let angle: f64 = args.flag("angle", 0.0);
    let blur = args.flag("blur", 4);
    args.finish();

    let depth = png::read_gray(Path::new(&input))?.shrink_to(WORK_SIZE).blur(blur);
    let frame = Parallel::new(depth.size(), angle.to_radians(), n);
    let lift = lift * depth.width as f64;
    let lines = offset::lines(&depth, &frame, lift, STEP);
    let (page, placed) = paper::place(depth.size(), paper::A4, MARGIN_MM, &lines);
    svg::write(Path::new(&output), page, &placed, STROKE_MM)
}
