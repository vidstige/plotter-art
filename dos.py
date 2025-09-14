from pathlib import Path
import random

import cairo
from freetype import Face

#FONT_PATH = "fonts/DeluxefontRegular-xoOO.ttf"
FONT_PATH = "fonts/perfect_dos_vga_437.ttf"


def clear(surface: cairo.Surface, color: tuple[float, float, float]):
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(*color)
    ctx.paint()


def as_surface(bitmap, format: cairo.Format) -> cairo.ImageSurface:
    stride = cairo.Format.A8.stride_for_width(bitmap.width)
    buffer = bitmap.buffer
    padding = [0] * (stride - bitmap.width)
    rows = [buffer[i*bitmap.width:(i+1)*bitmap.width] + padding for i in range(bitmap.rows)]
    pixels = [pixel for row in rows for pixel in row]
    return cairo.ImageSurface.create_for_data(
        bytearray(pixels),
        cairo.Format.A8,
        bitmap.width,
        bitmap.rows,
        stride,
    )


def conic_to(ctx, control, dest):
    # Convert quadratic Bézier to cubic Bézier
    x0, y0 = ctx.get_current_point()
    cx, cy = control
    dx, dy = dest
    c1x = x0 + 2.0 / 3.0 * (cx - x0)
    c1y = y0 + 2.0 / 3.0 * (cy - y0)
    c2x = dx + 2.0 / 3.0 * (cx - dx)
    c2y = dy + 2.0 / 3.0 * (cy - dy)
    ctx.curve_to(c1x, c1y, c2x, c2y, dx, dy)


def draw_glyph_outline(face: Face, char: str, context, x_offset, y_offset):
    face.load_char(char)
    slot = face.glyph
    outline = slot.outline

    points = outline.points
    tags = outline.tags
    contours = outline.contours
    start = 0

    context.new_path()
    for contour_end in contours:
        contour_points = points[start:contour_end+1]
        contour_tags = tags[start:contour_end+1]
        n = len(contour_points)

        def get_point(i):
            x, y = contour_points[i % n]
            return x / 64.0 + x_offset, -y / 64.0 + y_offset

        i = 0
        context.move_to(*get_point(i))
        while i < n:
            tag = contour_tags[i % n]
            curr = get_point(i)
            if tag & 1:  # on-curve
                context.line_to(*curr)
                i += 1
            else:  # off-curve (conic)
                next_i = (i + 1) % n
                if contour_tags[next_i] & 1:
                    control = curr
                    dest = get_point(next_i)
                    conic_to(context, control, dest)
                    i += 2
                else:
                    # two off-curve points: insert midpoint
                    control = curr
                    mid = tuple((a + b) / 2 for a, b in zip(get_point(i), get_point(next_i)))
                    conic_to(context, control, mid)
                    i += 1
        context.close_path()
        start = contour_end + 1

    context.fill()

def render_text_to_svg(surface, text, face):
    context = cairo.Context(surface)

    # White background
    #context.set_source_rgb(1, 1, 1)
    #context.paint()

    # Text position
    x, y = 0, 48
    context.set_source_rgb(0, 0, 0)  # black

    for char in text:
        if char == '\n':
            x, y = 0, y + face.height / 64.0
        else:
            draw_glyph_outline(face, char, context, x, y)
            x += face.glyph.advance.x / 64.0

    surface.finish()


class Terminal:
    def __init__(self, face: Face, width: int, height: int):
        self.x, self.y = 0, 0
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        clear(self.surface, color=(1, 1, 1))
        self.face = face

    def print(self, text: str):
        ctx = cairo.Context(self.surface)
        ctx.set_source_rgb(0, 0, 0)  # black text
        for char in text:
            self.face.load_char(char)
            bitmap = self.face.glyph.bitmap
            top = self.face.glyph.bitmap_top
            left = self.face.glyph.bitmap_left

            if bitmap.width > 0:
                # Convert grayscale bitmap to image surface
                glyph_surface = as_surface(bitmap, cairo.Format.A8)

                # Composite it into the main surface
                ctx.save()
                ctx.translate(self.x + left, self.y - top + 64)
                ctx.set_source_surface(glyph_surface, 0, 0)
                ctx.paint()
                ctx.restore()

            self.x += self.face.glyph.advance.x >> 6
        
        self.y += self.face.height / 64
        self.x = 0

    def save(self, path: Path):
        self.surface.write_to_png(path)

def render_to(lines: list[str], path: Path):
    face = Face(FONT_PATH)
    w, h = max(len(line) for line in lines), len(lines)
    face.set_char_size(48 * 64)  # 48 pt

    terminal = Terminal(face, w*48, h*64)
    for line in lines:
        terminal.print(line)
    terminal.save(path)

def progress(i, n):
    return ('+' * i) + ('-' * (n - i))

def text(n):
    for _ in range(n):
        p = random.randint(0, 32)
        yield f"{progress(p, 32)}    {100*p/32:>3.0f}%    {p:>2} B"

# Hersey settings
# Hatch spacing: 0.35 mm
# Hatch angle: 30 degrees
# Tolerance: 1.4

if __name__ == "__main__":
    #text = Path("texts/nostalgia.txt").read_text()
    lines = list(text(8))
    #render_to(lines, path=Path("output.png"))
    
    face = Face(FONT_PATH)
    face.set_char_size(48 * 64)  # 72pt

    w, h = max(len(line) for line in lines), len(lines)

    output = "output.svg"
    surface = cairo.SVGSurface(output, w*32, h*64)
    render_text_to_svg(surface, '\n'.join(lines), face)
    print(f"Saved to {output}")
    
    #for line in lines:
    #    print(line.rstrip())
