"""Draw the README diagrams (docs/pipeline.svg, docs/step.svg) as plain SVG.

Hand-placed layout so nothing overlaps or crosses; box sizes are measured from the text so nothing overflows.
Run:  python docs/make_diagrams.py
"""
import os
from PIL import ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"
_fr = _fb = None


def _fonts():
    global _fr, _fb
    if _fr is None:   # DejaVu is wider than the fonts browsers use, so measured boxes are on the safe side
        for r, b in (('DejaVuSans.ttf', 'DejaVuSans-Bold.ttf'), ('LiberationSans-Regular.ttf', 'LiberationSans-Bold.ttf')):
            try: _fr, _fb = ImageFont.truetype(r, 100), ImageFont.truetype(b, 100); break
            except OSError: pass
    return _fr, _fb


def tw(text, size, bold=False):
    fr, fb = _fonts()
    if fr is None: return len(text) * size * (0.62 if bold else 0.56)
    return (fb if bold else fr).getlength(text) * size / 100


def esc(s): return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


# stage colours: (panel fill, panel stroke, header text, box stroke, number badge)
STAGE = {
    'in':     ('#eff6ff', '#bfdbfe', '#1e40af', '#93c5fd', '#3b82f6'),
    'prep':   ('#f5f3ff', '#ddd6fe', '#5b21b6', '#c4b5fd', '#7c3aed'),
    'solve':  ('#fff7ed', '#fed7aa', '#9a3412', '#fdba74', '#ea580c'),
    'refine': ('#ecfdf5', '#a7f3d0', '#065f46', '#6ee7b7', '#059669'),
}


class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.o = w, h, []
        self.o.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{FONT}">')
        self.o.append('<defs>'
                      '<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                      '<path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker>'
                      '<marker id="arrA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                      '<path d="M0,0 L10,5 L0,10 z" fill="#c2410c"/></marker>'
                      '<filter id="sh" x="-10%" y="-10%" width="120%" height="140%"><feGaussianBlur in="SourceAlpha" stdDeviation="2"/><feOffset dy="1.5" result="b"/><feComponentTransfer><feFuncA type="linear" slope="0.10"/></feComponentTransfer><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
                      '</defs>')
        self.o.append(f'<rect x="0" y="0" width="{w}" height="{h}" rx="18" fill="#ffffff"/>')

    def rect(self, x, y, w, h, fill, stroke, rx=12, dash=False, shadow=False, sw=1.5):
        d = ' stroke-dasharray="6 4"' if dash else ''
        f = ' filter="url(#sh)"' if shadow else ''
        self.o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{f}/>')

    def text(self, x, y, s, size=13, bold=False, color='#1f2937', anchor='middle', italic=False):
        st = (' font-weight="700"' if bold else '') + (' font-style="italic"' if italic else '')
        self.o.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}"{st}>{esc(s)}</text>')

    def path(self, d, color='#475569', dash=False, marker='arr', sw=1.8):
        ds = ' stroke-dasharray="5 4"' if dash else ''
        self.o.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{ds} marker-end="url(#{marker})"/>')

    def label(self, x, y, s, color='#475569', bg='#ffffff', size=11.5):
        w = tw(s, size) + 12
        self.o.append(f'<rect x="{x - w / 2:.1f}" y="{y - 10:.1f}" width="{w:.1f}" height="19" rx="9.5" fill="{bg}" stroke="{color}" stroke-width="1"/>')
        self.text(x, y + 4, s, size, color=color)

    def save(self, name):
        self.o.append('</svg>')
        with open(os.path.join(HERE, name), 'w') as f: f.write('\n'.join(self.o))


def box_size(title, lines, pad=16, tsize=14, lsize=12):
    w = max([tw(title, tsize, True) + 34] + [tw(l, lsize) for l in lines]) + 2 * pad
    h = 30 + 17 * len(lines) + 14
    return w, h


def box(s, x, y, w, h, num, title, lines, stage, optional=False, kind='step'):
    fill, _, head, stroke, badge = STAGE[stage]
    bg = '#ffffff' if kind != 'check' else '#fffbeb'
    s.rect(x, y, w, h, bg, stroke if kind != 'check' else '#f59e0b', rx=12, dash=optional, shadow=True, sw=1.6)
    ty = y + 26
    if num is not None:
        bw = tw(str(num), 12, True) + 14
        tx = x + w / 2 - (tw(title, 14, True) + bw + 8) / 2
        s.rect(tx, ty - 14, bw, 19, badge, badge, rx=9.5)
        s.text(tx + bw / 2, ty, str(num), 12, True, '#ffffff')
        s.text(tx + bw + 8, ty, title, 14, True, '#111827', anchor='start')
    else:
        s.text(x + w / 2, ty, title, 14, True, '#111827')
    for k, l in enumerate(lines):
        s.text(x + w / 2, ty + 20 + 17 * k, l, 12, False, '#4b5563', italic=(optional and k == 0 and l.startswith('optional')))


def lane(s, x, y, w, h, title, stage):
    fill, stroke, head, _, _ = STAGE[stage]
    s.rect(x, y, w, h, fill, stroke, rx=16)
    s.text(x + 18, y + 24, title.upper(), 12, True, head, anchor='start')


def pipeline():
    W = 1240; M = 24; LW = W - 2 * M
    s = SVG(W, 1010)
    s.text(W / 2, 40, 'FlexSfM Studio pipeline', 22, True, '#0f172a')
    s.text(W / 2, 62, 'Every step saves its outputs, runs sanity checks and draws figures. Dashed = optional.', 13, False, '#64748b')

    # ---------------- inputs
    y0 = 84; lane(s, M, y0, LW, 112, 'Inputs', 'in')
    ins = [('Video', ['any camera: .mp4 .mov .mkv']), ('Photos / frames', ['a folder of images']),
           ('GPS (optional)', ['EXIF · CSV · DJI .SRT']), ('Masks (optional)', ['black = ignore'])]
    iw = (LW - 40 - 3 * 24) / 4; centers_in = []
    for k, (t, l) in enumerate(ins):
        x = M + 20 + k * (iw + 24); box(s, x, y0 + 38, iw, 60, None, t, l, 'in'); centers_in.append(x + iw / 2)

    def row(y, items, stage, gap=46):
        sizes = [box_size(t, l) for _, t, l, *_ in items]
        h = max(hh for _, hh in sizes); ws = [ww for ww, _ in sizes]
        free = LW - 40 - sum(ws) - gap * (len(items) - 1)
        ws = [ww + free / len(items) for ww in ws]          # stretch boxes evenly to fill the lane
        xs = []; x = M + 20
        for (n, t, l, *opt), ww in zip(items, ws):
            box(s, x, y, ww, h, n, t, l, stage, optional=bool(opt and opt[0]), kind=(opt[1] if len(opt) > 1 else 'step')); xs.append((x, ww)); x += ww + gap
        for (xa, wa), (xb, _) in zip(xs, xs[1:]):
            s.path(f'M{xa + wa + 4:.1f},{y + h / 2:.1f} L{xb - 4:.1f},{y + h / 2:.1f}')
        return xs, h

    # ---------------- prepare
    y1 = 230; lane(s, M, y1, LW, 150, 'Prepare', 'prep')
    xs1, h1 = row(y1 + 40, [(1, 'Frames and GPS', ['extract frames, read GPS,', 'find passes over the same place']),
                            (2, 'Keypoints', ['SIFT / ALIKED keypoints,', 'FlexGlue + global descriptors']),
                            (3, 'Image pairs', ['sequence, GPS radius and', 'image similarity (revisits)']),
                            (4, 'Matching', ['FlexGlue matches + geometric', 'check → COLMAP database'])], 'prep')
    for c in centers_in:                                   # inputs feed step 1 through a bus line
        s.o.append(f'<path d="M{c:.1f},{y0 + 98:.1f} L{c:.1f},{y0 + 122:.1f}" stroke="#93c5fd" stroke-width="1.8" fill="none"/>')
    bx = xs1[0][0] + xs1[0][1] / 2
    s.o.append(f'<path d="M{centers_in[0]:.1f},{y0 + 122:.1f} L{centers_in[-1]:.1f},{y0 + 122:.1f}" stroke="#93c5fd" stroke-width="1.8" fill="none"/>')
    s.path(f'M{bx:.1f},{y0 + 122:.1f} L{bx:.1f},{y1 + 36:.1f}', color='#3b82f6')

    # ---------------- solve and check
    y2 = 414; lane(s, M, y2, LW, 286, 'Solve and check', 'solve')
    xs2, h2 = row(y2 + 40, [(5, 'Reconstruct cameras', ['GLOMAP (global) or COLMAP', '(incremental), align to GPS']),
                            (6, 'Markers', ['optional: ground control points,', 'scale bars, check-point errors'], True),
                            (7, 'Diagnose', ['do revisits line up?', 'do all cameras fit?'], False, 'check')], 'solve', gap=56)
    # repair below diagnose
    x7, w7 = xs2[2]; rw, rh = box_size('Repair revisits', ['optional: tie passes together,', 'bundle adjust, held-out test'])
    rw = w7; ry = y2 + 40 + h2 + 64
    box(s, x7, ry, rw, rh, 8, 'Repair revisits', ['optional: tie passes together,', 'bundle adjust, held-out test'], 'solve', optional=True)
    a, b = x7 + w7 * 0.32, x7 + w7 * 0.68
    s.path(f'M{a:.1f},{y2 + 40 + h2 + 4:.1f} L{a:.1f},{ry - 4:.1f}', color='#c2410c', marker='arrA')
    s.label(a, y2 + 40 + h2 + 32, 'doubled passes', color='#c2410c', bg='#fff7ed')
    s.path(f'M{b:.1f},{ry - 4:.1f} L{b:.1f},{y2 + 40 + h2 + 4:.1f}', color='#c2410c', marker='arrA', dash=True)
    s.label(b, y2 + 40 + h2 + 32, 're-check', color='#c2410c', bg='#fff7ed')
    # 4 -> 5 : down the right gutter, across the gap, into 5
    x4, w4 = xs1[3]; x5, w5 = xs2[0]; gy = (y1 + 150 + y2) / 2
    s.path(f'M{x4 + w4 / 2:.1f},{y1 + 40 + h1 + 4:.1f} L{x4 + w4 / 2:.1f},{gy:.1f} L{x5 + w5 / 2:.1f},{gy:.1f} L{x5 + w5 / 2:.1f},{y2 + 36:.1f}')
    # notes on the left of the repair box
    s.text(M + 24, ry + 22, 'If a check fails, the Results and Issues tabs say', 12.5, False, '#7c2d12', anchor='start')
    s.text(M + 24, ry + 41, 'which setting to change and which step to re-run;', 12.5, False, '#7c2d12', anchor='start')
    s.text(M + 24, ry + 60, 'Diagnose problems come with an Apply button.', 12.5, False, '#7c2d12', anchor='start')

    # ---------------- refine and deliver
    y3 = 734; lane(s, M, y3, LW, 150, 'Refine and deliver', 'refine')
    xs3, h3 = row(y3 + 40, [(9, 'Densify + refine', ['dense full-resolution tracks,', 'sub-pixel refinement']),
                            (10, 'Filter points', ['error / angle / track limits,', 'table of point counts']),
                            (11, 'Export', ['undistorted images + masks,', 'PLY, cameras, georeference'])], 'refine')
    # 7 -> 9 "all good": right of diagnose, down the right gutter, across, into 9
    x9, w9 = xs3[0]; rx_ = M + LW - 10; gy2 = (y2 + 286 + y3) / 2
    s.path(f'M{x7 + w7 + 4:.1f},{y2 + 40 + h2 / 2:.1f} L{rx_:.1f},{y2 + 40 + h2 / 2:.1f} L{rx_:.1f},{gy2:.1f} L{x9 + w9 / 2:.1f},{gy2:.1f} L{x9 + w9 / 2:.1f},{y3 + 36:.1f}', color='#059669')
    s.label((x9 + w9 / 2 + rx_) / 2, gy2, 'all good', color='#047857', bg='#ecfdf5')

    # ---------------- output
    yo = 918; ow = 560; ox = (W - ow) / 2
    s.rect(ox, yo, ow, 68, '#059669', '#047857', rx=34, shadow=True)
    s.text(W / 2, yo + 29, 'Ready for Gaussian splatting / NeRF / photogrammetry', 15, True, '#ffffff')
    s.text(W / 2, yo + 50, 'export/undistorted/  →  images · sparse/0 · masks', 12.5, False, '#d1fae5')
    x11, w11 = xs3[2]
    s.path(f'M{x11 + w11 / 2:.1f},{y3 + 40 + h3 + 4:.1f} L{x11 + w11 / 2:.1f},{yo - 14:.1f} L{ox + ow - 40:.1f},{yo - 14:.1f} L{ox + ow - 40:.1f},{yo - 4:.1f}', color='#059669')
    s.save('pipeline.svg')


def step_anatomy():
    W, H = 1240, 400; s = SVG(W, H)
    s.text(W / 2, 40, 'What happens in every step', 22, True, '#0f172a')
    s.text(W / 2, 62, 'Each step runs as its own process, so the app stays responsive and a step can be stopped at any time.', 13, False, '#64748b')
    def b(x, y, w, h, t, l, stage, kind='step'):
        box(s, x, y, w, h, None, t, l, stage, kind=kind); return x, y, w, h
    S = b(40, 104, 230, 66, 'Settings', ['project.yaml (edit in the app)'], 'in')
    P = b(40, 214, 230, 66, 'Earlier outputs', ['+ the current model'], 'in')
    R = b(330, 150, 200, 86, 'Step runs', ['logs → logs/<step>.log', 'progress → the app'], 'prep')
    O1 = b(590, 96, 260, 66, 'Outputs', ['frames, features, models/<name>'], 'refine')
    O2 = b(590, 184, 260, 66, 'Report', ['checks · metrics · figures'], 'refine')
    C = b(910, 184, 290, 66, 'Results / Issues tabs', ['each check: pass · warn · fail'], 'solve', kind='check')
    N = b(910, 96, 290, 66, 'Next step', ['or “Run from here to the end”'], 'refine')
    F = b(590, 296, 610, 66, 'Fix', ['change the suggested setting, or press Apply on a Diagnose issue'], 'solve')
    s.path(f'M{S[0] + S[2] + 4},{S[1] + 33} L300,{S[1] + 33} L300,{R[1] + 30} L{R[0] - 4},{R[1] + 30}')
    s.path(f'M{P[0] + P[2] + 4},{P[1] + 33} L300,{P[1] + 33} L300,{R[1] + 56} L{R[0] - 4},{R[1] + 56}')
    s.path(f'M{R[0] + R[2] + 4},{R[1] + 30} L560,{R[1] + 30} L560,{O1[1] + 33} L{O1[0] - 4},{O1[1] + 33}')
    s.path(f'M{R[0] + R[2] + 4},{R[1] + 56} L560,{R[1] + 56} L560,{O2[1] + 33} L{O2[0] - 4},{O2[1] + 33}')
    s.path(f'M{O2[0] + O2[2] + 4},{O2[1] + 33} L{C[0] - 4},{C[1] + 33}')
    s.path(f'M{C[0] + C[2] / 2},{C[1] - 4} L{N[0] + N[2] / 2},{N[1] + N[3] + 4}', color='#059669')
    s.label(C[0] + C[2] / 2 + 34, (C[1] + N[1] + N[3]) / 2, 'pass', color='#047857', bg='#ecfdf5')
    s.path(f'M{C[0] + C[2] / 2},{C[1] + C[3] + 4} L{C[0] + C[2] / 2},{F[1] - 4}', color='#c2410c', marker='arrA')
    s.label(C[0] + C[2] / 2 + 52, (C[1] + C[3] + F[1]) / 2, 'warn / fail', color='#c2410c', bg='#fff7ed')
    s.path(f'M{F[0] - 4},{F[1] + 33} L22,{F[1] + 33} L22,{S[1] + 33} L{S[0] - 4},{S[1] + 33}', color='#c2410c', marker='arrA', dash=True)
    s.label(300, F[1] + 33, 're-run', color='#c2410c', bg='#fff7ed')
    s.save('step.svg')


if __name__ == '__main__':
    pipeline(); step_anatomy(); print('wrote', os.path.join(HERE, 'pipeline.svg'), os.path.join(HERE, 'step.svg'))
