//! PNG decoding to a grayscale raster. Supports non-interlaced 8 and 16
//! bit gray, gray+alpha, RGB and RGBA; color is averaged, alpha ignored.
use std::{fs, io, path::Path};

use crate::inflate;
use crate::raster::Raster;

const SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

fn u32_be(b: &[u8]) -> usize {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = a as i16 + b as i16 - c as i16;
    let (pa, pb, pc) = ((p - a as i16).abs(), (p - b as i16).abs(), (p - c as i16).abs());
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Reverses the per-row filters. `bpp` is bytes per pixel.
fn unfilter(data: &[u8], stride: usize, bpp: usize) -> io::Result<Vec<u8>> {
    let mut out = vec![0u8; data.len() / (stride + 1) * stride];
    let mut up = vec![0u8; stride];
    for (y, row) in data.chunks_exact(stride + 1).enumerate() {
        let cur = &mut out[y * stride..(y + 1) * stride];
        for x in 0..stride {
            let a = if x >= bpp { cur[x - bpp] } else { 0 };
            let c = if x >= bpp { up[x - bpp] } else { 0 };
            let b = up[x];
            let predicted = match row[0] {
                0 => 0,
                1 => a,
                2 => b,
                3 => ((a as u16 + b as u16) / 2) as u8,
                4 => paeth(a, b, c),
                _ => return Err(invalid("bad filter type")),
            };
            cur[x] = row[1 + x].wrapping_add(predicted);
        }
        up.copy_from_slice(cur);
    }
    Ok(out)
}

/// Decodes PNG bytes into a raster with values in [0, 1].
pub fn decode_gray(bytes: &[u8]) -> io::Result<Raster> {
    let mut chunks = bytes.strip_prefix(&SIGNATURE).ok_or(invalid("not a png"))?;
    let mut header = None;
    let mut idat = Vec::new();
    while chunks.len() >= 12 {
        let len = u32_be(chunks);
        let body = chunks.get(8..8 + len).ok_or(invalid("truncated chunk"))?;
        match &chunks[4..8] {
            b"IHDR" => header = Some(body),
            b"IDAT" => idat.extend_from_slice(body),
            _ => {}
        }
        chunks = &chunks[12 + len..];
    }
    let h = header.ok_or(invalid("missing IHDR"))?;
    let (width, height, depth, color) = (u32_be(h), u32_be(&h[4..]), h[8], h[9]);
    let channels = match color {
        0 => 1,
        2 => 3,
        4 => 2,
        6 => 4,
        _ => return Err(invalid("unsupported color type")),
    };
    if !(depth == 8 || depth == 16) || h[12] != 0 {
        return Err(invalid("unsupported bit depth or interlacing"));
    }
    let sample_bytes = depth as usize / 8;
    let bpp = channels * sample_bytes;
    let pixels = unfilter(&inflate::zlib(&idat)?, width * bpp, bpp)?;
    if pixels.len() != width * height * bpp {
        return Err(invalid("image data size mismatch"));
    }
    let colors = if channels >= 3 { 3 } else { 1 };
    let max = ((1u32 << depth) - 1) as f32;
    let gray = |px: &[u8]| {
        let samples = px.chunks(sample_bytes).take(colors);
        let sum: u32 = samples.map(|s| s.iter().fold(0, |v, &b| v << 8 | b as u32)).sum();
        sum as f32 / (colors as f32 * max)
    };
    let data = pixels.chunks(bpp).map(gray).collect();
    Ok(Raster { width, height, data })
}

pub fn read_gray(path: &Path) -> io::Result<Raster> {
    decode_gray(&fs::read(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inflate::tests::stored_zlib;

    fn chunk(kind: &[u8], body: &[u8]) -> Vec<u8> {
        let len = (body.len() as u32).to_be_bytes();
        [&len, kind, body, &[0; 4]].concat()
    }

    /// PNG with the given color type, 8 bits, built from filtered rows.
    fn png(width: u32, height: u32, color: u8, rows: &[u8]) -> Vec<u8> {
        let mut ihdr = [width.to_be_bytes(), height.to_be_bytes()].concat();
        ihdr.extend([8, color, 0, 0, 0]);
        let parts = [
            SIGNATURE.to_vec(),
            chunk(b"IHDR", &ihdr),
            chunk(b"IDAT", &stored_zlib(rows)),
            chunk(b"IEND", &[]),
        ];
        parts.concat()
    }

    #[test]
    fn sub_and_up_filters_reconstruct_gray_pixels() {
        // row 0 Sub: 10, 10+20; row 1 Up: 10+5, 30+5
        let rows = [1, 10, 20, 2, 5, 5];
        let r = decode_gray(&png(2, 2, 0, &rows)).unwrap();
        let expected: Vec<f32> = [10, 30, 15, 35].iter().map(|&v| v as f32 / 255.0).collect();
        assert_eq!(r.data, expected);
    }

    #[test]
    fn rgb_is_averaged_to_gray() {
        let rows = [0, 255, 0, 0, 0, 255, 255];
        let r = decode_gray(&png(2, 1, 2, &rows)).unwrap();
        assert_eq!(r.data, [1.0 / 3.0, 2.0 / 3.0]);
    }

    #[test]
    fn palette_images_are_rejected() {
        assert!(decode_gray(&png(1, 1, 3, &[0, 0])).is_err());
    }
}
