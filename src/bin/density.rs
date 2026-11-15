//! Random lines whose density follows the darkness of an image.
use std::io;
use std::path::Path;

use plotter_art::args::Args;
use plotter_art::density::{self, Params};
use plotter_art::rng::Rng;
use plotter_art::{paper, png, svg};

const USAGE: &str = "density <image.png> <out.svg> [--lines 800] \
[--length 300] [--pull 15] [--spread 30] [--blur 20] [--seed 1]";
const WORK_SIZE: usize = 1024;
const MARGIN_MM: f64 = 15.0;
const STROKE_MM: f64 = 0.3;
const STEP: f64 = 0.5;

fn main() -> io::Result<()> {
    let mut args = Args::parse(std::env::args().skip(1));
    let input = args.positional(0).expect(USAGE).to_string();
    let output = args.positional(1).expect(USAGE).to_string();
    let params = Params {
        lines: args.flag("lines", 800),
        step: STEP,
        length: args.flag("length", 300.0),
        pull: args.flag("pull", 15.0),
        spread: args.flag::<f64>("spread", 30.0).to_radians(),
    };
    let blur = args.flag("blur", 20);
    let mut rng = Rng::new(args.flag("seed", 1));
    args.finish();

    let gray = png::read_gray(Path::new(&input))?.shrink_to(WORK_SIZE);
    let darkness = gray.map(|v| 1.0 - v).blur(blur);
    let lines = density::lines(&darkness, &params, &mut rng);
    let (page, placed) = paper::place(darkness.size(), paper::A4, MARGIN_MM, &lines);
    svg::write(Path::new(&output), page, &placed, STROKE_MM)
}
