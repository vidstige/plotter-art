//! DEFLATE (RFC 1951) decompression and its zlib (RFC 1950) wrapper.
use std::io;

const LENGTH_BASE: [usize; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51,
    59, 67, 83, 99, 115, 131, 163, 195, 227, 258,
];
const LENGTH_EXTRA: [u32; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
    5, 5, 5, 5, 0,
];
const DIST_BASE: [usize; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385,
    513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385,
    24577,
];
const DIST_EXTRA: [u32; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10,
    10, 11, 11, 12, 12, 13, 13,
];
const CODE_LENGTH_ORDER: [usize; 19] =
    [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Little-endian bit reader.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Bits<'_> {
    fn bit(&mut self) -> io::Result<u32> {
        let byte = self.data.get(self.pos / 8).ok_or(invalid("truncated"))?;
        let b = (byte >> (self.pos % 8)) & 1;
        self.pos += 1;
        Ok(b as u32)
    }

    fn bits(&mut self, n: u32) -> io::Result<usize> {
        let mut v = 0;
        for i in 0..n {
            v |= (self.bit()? as usize) << i;
        }
        Ok(v)
    }

    fn align(&mut self) {
        self.pos = self.pos.div_ceil(8) * 8;
    }
}

/// Canonical Huffman code: symbol count per code length, and the
/// symbols sorted by code.
struct Huffman {
    counts: [usize; 16],
    symbols: Vec<usize>,
}

fn huffman(lengths: &[usize]) -> Huffman {
    let mut counts = [0; 16];
    for &l in lengths {
        counts[l] += 1;
    }
    counts[0] = 0;
    let mut offsets = [0; 16];
    for l in 1..15 {
        offsets[l + 1] = offsets[l] + counts[l];
    }
    let mut symbols = vec![0; lengths.len()];
    for (s, &l) in lengths.iter().enumerate().filter(|(_, l)| **l > 0) {
        symbols[offsets[l]] = s;
        offsets[l] += 1;
    }
    Huffman { counts, symbols }
}

fn decode(bits: &mut Bits, h: &Huffman) -> io::Result<usize> {
    // first: first code of this length, index: its position in symbols
    let (mut code, mut first, mut index) = (0, 0, 0);
    for &count in &h.counts[1..] {
        code |= bits.bit()? as usize;
        if code < first + count {
            return Ok(h.symbols[index + code - first]);
        }
        index += count;
        first = (first + count) << 1;
        code <<= 1;
    }
    Err(invalid("bad huffman code"))
}

fn stored(bits: &mut Bits, out: &mut Vec<u8>) -> io::Result<()> {
    bits.align();
    let len = bits.bits(16)?;
    bits.bits(16)?;
    let start = bits.pos / 8;
    let block = bits.data.get(start..start + len);
    out.extend_from_slice(block.ok_or(invalid("truncated"))?);
    bits.pos += 8 * len;
    Ok(())
}

fn fixed() -> (Huffman, Huffman) {
    let lit: Vec<usize> = (0..288)
        .map(|s| match s {
            0..144 => 8,
            144..256 => 9,
            256..280 => 7,
            _ => 8,
        })
        .collect();
    (huffman(&lit), huffman(&[5; 30]))
}

fn dynamic(bits: &mut Bits) -> io::Result<(Huffman, Huffman)> {
    let nlit = bits.bits(5)? + 257;
    let ndist = bits.bits(5)? + 1;
    let ncode = bits.bits(4)? + 4;
    let mut code_lengths = [0; 19];
    for &i in &CODE_LENGTH_ORDER[..ncode] {
        code_lengths[i] = bits.bits(3)?;
    }
    let code = huffman(&code_lengths);
    let mut lengths = Vec::with_capacity(nlit + ndist);
    while lengths.len() < nlit + ndist {
        let (value, repeat) = match decode(bits, &code)? {
            16 => {
                let prev = lengths.last().ok_or(invalid("no length to repeat"))?;
                (*prev, 3 + bits.bits(2)?)
            }
            17 => (0, 3 + bits.bits(3)?),
            18 => (0, 11 + bits.bits(7)?),
            l => (l, 1),
        };
        lengths.extend(std::iter::repeat_n(value, repeat));
    }
    Ok((huffman(&lengths[..nlit]), huffman(&lengths[nlit..nlit + ndist])))
}

fn codes(
    bits: &mut Bits,
    out: &mut Vec<u8>,
    lit: &Huffman,
    dist: &Huffman,
) -> io::Result<()> {
    loop {
        let symbol = decode(bits, lit)?;
        if symbol < 256 {
            out.push(symbol as u8);
            continue;
        }
        if symbol == 256 {
            return Ok(());
        }
        let s = symbol - 257;
        let len = LENGTH_BASE[s] + bits.bits(LENGTH_EXTRA[s])?;
        let d = decode(bits, dist)?;
        let distance = DIST_BASE[d] + bits.bits(DIST_EXTRA[d])?;
        if distance > out.len() {
            return Err(invalid("distance too far back"));
        }
        for _ in 0..len {
            out.push(out[out.len() - distance]);
        }
    }
}

/// Decompresses a raw DEFLATE stream.
pub fn inflate(data: &[u8]) -> io::Result<Vec<u8>> {
    let mut bits = Bits { data, pos: 0 };
    let mut out = Vec::new();
    loop {
        let last = bits.bit()? == 1;
        match bits.bits(2)? {
            0 => stored(&mut bits, &mut out)?,
            1 => {
                let (lit, dist) = fixed();
                codes(&mut bits, &mut out, &lit, &dist)?;
            }
            2 => {
                let (lit, dist) = dynamic(&mut bits)?;
                codes(&mut bits, &mut out, &lit, &dist)?;
            }
            _ => return Err(invalid("bad block type")),
        }
        if last {
            return Ok(out);
        }
    }
}

/// Decompresses a zlib stream. The Adler-32 checksum is not verified.
pub fn zlib(data: &[u8]) -> io::Result<Vec<u8>> {
    match data {
        [cmf, _, rest @ ..] if cmf & 0x0f == 8 => inflate(rest),
        _ => Err(invalid("not a zlib stream")),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// zlib stream with a single uncompressed block.
    pub(crate) fn stored_zlib(bytes: &[u8]) -> Vec<u8> {
        let len = bytes.len() as u16;
        let mut z = vec![0x78, 0x01, 1];
        z.extend(len.to_le_bytes());
        z.extend((!len).to_le_bytes());
        z.extend(bytes);
        z
    }

    #[test]
    fn stored_block_is_copied_verbatim() {
        let z = stored_zlib(b"plotter");
        assert_eq!(zlib(&z).unwrap(), b"plotter");
    }

    #[test]
    fn fixed_huffman_block_expands_back_references() {
        // zlib.compress(b"hello hello hello hello") from Python
        let z = [
            120, 156, 203, 72, 205, 201, 201, 87, 200, 64, 39, 1, 104, 3,
            8, 177,
        ];
        assert_eq!(zlib(&z).unwrap(), b"hello hello hello hello");
    }

    #[test]
    fn truncated_stream_is_an_error() {
        let z = stored_zlib(b"plotter");
        assert!(zlib(&z[..z.len() - 1]).is_err());
    }
}
