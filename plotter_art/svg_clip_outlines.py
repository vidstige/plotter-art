from lxml import etree
from math import cos, sin, radians
from shapely.geometry import Polygon, LineString, LinearRing, MultiLineString, box
from shapely.ops import unary_union
import re, sys

# Slight inflation of occluders to avoid hairline show-through (tune to your units).
EPS = 0.03

SVG_NS = "http://www.w3.org/2000/svg"
NS = {"svg": SVG_NS}

# ----------------- Transforms -----------------
def mat_mul(a,b):
    return [
        [a[0][0]*b[0][0]+a[0][1]*b[1][0]+a[0][2]*b[2][0],
         a[0][0]*b[0][1]+a[0][1]*b[1][1]+a[0][2]*b[2][1],
         a[0][0]*b[0][2]+a[0][1]*b[1][2]+a[0][2]*b[2][2]],
        [a[1][0]*b[0][0]+a[1][1]*b[1][0]+a[1][2]*b[2][0],
         a[1][0]*b[0][1]+a[1][1]*b[1][1]+a[1][2]*b[2][1],
         a[1][0]*b[0][2]+a[1][1]*b[1][2]+a[1][2]*b[2][2]],
        [a[2][0]*b[0][0]+a[2][1]*b[1][0]+a[2][2]*b[2][0],
         a[2][0]*b[0][1]+a[2][1]*b[1][1]+a[2][2]*b[2][1]]
    ]

def mat_ident(): return [[1,0,0],[0,1,0],[0,0,1]]
def mat_translate(tx,ty): return [[1,0,tx],[0,1,ty],[0,0,1]]
def mat_scale(sx,sy): return [[sx,0,0],[0,sy,0],[0,0,1]]

def mat_rotate(deg, cx=0.0, cy=0.0):
    th = radians(deg)
    R = [[cos(th),-sin(th),0],[sin(th),cos(th),0],[0,0,1]]
    if cx or cy:
        return mat_mul(mat_mul(mat_translate(cx,cy), R), mat_translate(-cx,-cy))
    return R

def mat_matrix(a,b,c,d,e,f): return [[a,c,e],[b,d,f],[0,0,1]]

TRANS_RE = re.compile(r'(matrix|translate|scale|rotate)\s*\(([^)]*)\)')

def parse_transform_list(ts):
    M = mat_ident()
    if not ts: return M
    for fn, args in TRANS_RE.findall(ts):
        nums = [float(x) for x in re.split(r'[ ,]+', args.strip()) if x]
        if fn == 'matrix' and len(nums)==6: M = mat_mul(M, mat_matrix(*nums))
        elif fn == 'translate':
            if len(nums)==1: nums.append(0.0)
            M = mat_mul(M, mat_translate(nums[0], nums[1]))
        elif fn == 'scale':
            if len(nums)==1: nums.append(nums[0])
            M = mat_mul(M, mat_scale(nums[0], nums[1]))
        elif fn == 'rotate':
            if len(nums)==1: M = mat_mul(M, mat_rotate(nums[0]))
            elif len(nums)==3: M = mat_mul(M, mat_rotate(nums[0], nums[1], nums[2]))
    return M

def get_ctm(el):
    Ms = []
    cur = el
    while cur is not None and isinstance(cur.tag, str):
        t = cur.get('transform')
        if t: Ms.append(parse_transform_list(t))
        cur = cur.getparent()
    M = mat_ident()
    for m in reversed(Ms): M = mat_mul(M, m)
    return M

def apply_mat(M, pt):
    x,y = pt
    return (M[0][0]*x + M[0][1]*y + M[0][2],
            M[1][0]*x + M[1][1]*y + M[1][2])

# ----------------- ViewBox / clip rectangle -----------------
def parse_number(s):
    if s is None: return None
    try:
        match = re.match(r'[-+]?(\d+(\.\d+)?|\.\d+)([eE][-+]?\d+)?', s)
        return float(match.group()) if match else None
    except Exception:
        return None

def get_viewbox_clip(root):
    vb = root.get('viewBox')
    if vb:
        parts = [float(x) for x in re.split(r'[ ,]+', vb.strip()) if x]
        if len(parts) == 4:
            minx, miny, w, h = parts
            return box(minx, miny, minx + w, miny + h)
    # fallback to width/height if numeric (user units)
    w = parse_number(root.get('width'))
    h = parse_number(root.get('height'))
    if w is not None and h is not None:
        return box(0, 0, w, h)
    return None  # no clip

# ----------------- Geometry extraction -----------------
def parse_points_attr(points):
    nums = [float(t) for t in re.split(r'[ ,]+', points.strip()) if t]
    return [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)]

def polygon_from_polygon(el):
    pts_attr = el.get('points')
    if not pts_attr: return None
    pts = parse_points_attr(pts_attr)
    if len(pts) < 3: return None
    M = get_ctm(el)
    pts = [apply_mat(M,p) for p in pts]
    return Polygon(pts)

def polygon_from_rect(el):
    try:
        x = float(el.get('x','0')); y = float(el.get('y','0'))
        w = float(el.get('width')); h = float(el.get('height'))
    except: return None
    pts = [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
    M = get_ctm(el)
    pts = [apply_mat(M,p) for p in pts]
    return Polygon(pts)

# Straight-line path parser (M/L/H/V/Z only). Curves are ignored.
PATH_CMD_RE = re.compile(r'([MLHVZmlhvz])|(-?\d*\.?\d+(?:[eE][-+]?\d+)?)')

def polygon_from_path(el):
    d = el.get('d')
    if not d: return None
    tokens = PATH_CMD_RE.findall(d)
    if not tokens: return None
    pts = []
    x=y=0.0
    start=None
    cmd=None
    i=0
    def add_point(px,py): pts.append((px,py))
    while i < len(tokens):
        tcmd, tnum = tokens[i]
        if tcmd:
            cmd = tcmd; i += 1
            if cmd in 'Zz' and start is not None: add_point(*start); start=None
            continue
        if cmd in ('M','L'):
            x = float(tokens[i][1]); y = float(tokens[i+1][1]); i += 2; add_point(x,y)
            if cmd=='M' and start is None: start=(x,y)
        elif cmd in ('m','l'):
            dx = float(tokens[i][1]); dy = float(tokens[i+1][1]); i += 2
            x += dx; y += dy; add_point(x,y)
            if cmd=='m' and start is None: start=(x,y)
        elif cmd == 'H':
            x = float(tokens[i][1]); i += 1; add_point(x,y)
        elif cmd == 'h':
            dx = float(tokens[i][1]); i += 1; x += dx; add_point(x,y)
        elif cmd == 'V':
            y = float(tokens[i][1]); i += 1; add_point(x,y)
        elif cmd == 'v':
            dy = float(tokens[i][1]); i += 1; y += dy; add_point(x,y)
        else:
            return None  # curves not supported here
    if len(pts) < 3: return None
    M = get_ctm(el)
    pts = [apply_mat(M,p) for p in pts]
    poly = Polygon(pts)
    if not poly.is_valid or poly.is_empty: return None
    return poly

def outline_from_polygon(poly):
    return LineString(list(poly.exterior.coords))

def multiline_to_paths(geom):
    if geom.is_empty: return []
    if isinstance(geom, (LineString, LinearRing)):
        geoms = [geom]
    elif isinstance(geom, MultiLineString):
        geoms = list(geom.geoms)
    else:
        geoms = []
        for g in getattr(geom, 'geoms', []):
            if isinstance(g, (LineString, LinearRing)):
                geoms.append(g)
    out=[]
    for g in geoms:
        if isinstance(g, LinearRing): g = LineString(g)
        coords = list(g.coords)
        if not coords: continue
        d = f"M {coords[0][0]} {coords[0][1]} " + " ".join(f"L {x} {y}" for x,y in coords[1:])
        out.append(d)
    return out

# ----------------- Main -----------------
def main(in_svg, out_svg):
    tree = etree.parse(in_svg)
    root = tree.getroot()

    # Build clip rectangle from viewBox/width/height
    clip_rect = get_viewbox_clip(root)
    if clip_rect is None:
        print("Note: No viewBox or numeric width/height found — skipping outer clipping.")
    else:
        print(f"Clipping to rect: {clip_rect.bounds}")

    # Collect elements in document order and extract polygons
    items = []
    for el in root.xpath('.//*'):
        if not isinstance(el.tag, str): continue
        tag = etree.QName(el).localname
        poly = None
        if tag == 'polygon':
            poly = polygon_from_polygon(el)
        elif tag == 'rect':
            poly = polygon_from_rect(el)
        elif tag == 'path':
            poly = polygon_from_path(el)  # straight edges only
        if poly is None or poly.is_empty: continue

        # Clip the fill polygon to viewBox early
        if clip_rect is not None:
            poly = poly.intersection(clip_rect)
            if poly.is_empty:  # completely out of bounds
                continue

        fill = (el.get('fill') or '').lower()
        occludes = (fill != 'none')
        outline = outline_from_polygon(poly)
        items.append((el, poly, occludes, outline))

    if not items:
        print("Parsed 0 simple polygonal shapes. Writing original SVG.")
        tree.write(out_svg, encoding='utf-8', xml_declaration=True, pretty_print=True)
        return

    # Output shell with size/viewBox
    out_svg_el = etree.Element(f'{{{SVG_NS}}}svg', nsmap=root.nsmap)
    for attr in ('width','height','viewBox'):
        v = root.get(attr)
        if v: out_svg_el.set(attr, v)
    grp = etree.SubElement(out_svg_el, f'{{{SVG_NS}}}g')
    grp.set('id', 'visible-outlines')

    occlusion = None
    kept_segments = 0

    for el, poly, occludes, outline in reversed(items):
        # Subtract later occlusions
        visible = outline if occlusion is None else outline.difference(occlusion)
        # Clip visible segments to viewBox
        if clip_rect is not None:
            visible = visible.intersection(clip_rect)

        for d in multiline_to_paths(visible):
            path = etree.SubElement(grp, f'{{{SVG_NS}}}path')
            path.set('d', d)
            path.set('fill', 'none')
            path.set('stroke', '#000')
            path.set('stroke-width', '0.5')  # tune for your pen/scale
            kept_segments += 1

        if occludes:
            grown = poly.buffer(EPS) if EPS>0 else poly
            if clip_rect is not None:
                grown = grown.intersection(clip_rect)
            occlusion = grown if occlusion is None else unary_union([occlusion, grown])

    etree.ElementTree(out_svg_el).write(out_svg, encoding='utf-8', xml_declaration=True, pretty_print=True)
    print(f"Parsed shapes: {len(items)}, output visible segments: {kept_segments}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python svg_clip_outlines.py input.svg output.svg")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

