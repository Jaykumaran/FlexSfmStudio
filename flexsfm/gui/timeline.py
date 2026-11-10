"""Steps drawn as a vertical timeline: a connector line through status nodes, the running step pulsing, the selected step highlighted."""
import math, time
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

R_STATUS, R_CHECKS, R_SECS, R_OPT, R_NUM, R_TITLE, R_NEXT, R_PREV = (Qt.UserRole + k for k in range(1, 9))

COL = {'done': '#16a34a', 'stale': '#d97706', 'running': '#2563eb', 'failed': '#dc2626', 'interrupted': '#dc2626', 'not run': '#9ca3af'}
LABEL = {'done': 'done', 'stale': 'needs re-run', 'running': 'running', 'failed': 'failed', 'interrupted': 'interrupted', 'not run': 'not run'}


def _pen(color, w):
    pen = QtGui.QPen(QtGui.QColor(color)); pen.setWidthF(w); pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin); return pen


def _fmt_secs(s):
    if s is None: return ''
    s = int(s); return f'{s} s' if s < 60 else f'{s // 60} min {s % 60:02d} s' if s < 3600 else f'{s // 3600} h {(s % 3600) // 60:02d} min'


class TimelineDelegate(QtWidgets.QStyledItemDelegate):
    H = 66

    def sizeHint(self, opt, idx):
        return QtCore.QSize(opt.rect.width(), self.H)

    def paint(self, p, opt, idx):
        p.save(); p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = opt.rect; st = idx.data(R_STATUS) or 'not run'; chk = idx.data(R_CHECKS) or {}; opt_ = bool(idx.data(R_OPT))
        num = idx.data(R_NUM) or ''; title = idx.data(R_TITLE) or ''; secs = idx.data(R_SECS); nxt = idx.data(R_NEXT); prv = idx.data(R_PREV)
        selected = bool(opt.state & QtWidgets.QStyle.State_Selected); hover = bool(opt.state & QtWidgets.QStyle.State_MouseOver)
        running = st == 'running'; col = QtGui.QColor(COL.get(st, '#9ca3af'))
        # ---- card background
        card = QtCore.QRectF(r.left() + 40, r.top() + 4, r.width() - 46, r.height() - 8)
        if running or selected or hover:
            bg = QtGui.QColor('#eff6ff' if running else '#eef2ff' if selected else '#f8fafc')
            p.setPen(QtGui.QPen(QtGui.QColor('#93c5fd' if running else '#c7d2fe' if selected else '#e2e8f0'), 1.2)); p.setBrush(bg)
            p.drawRoundedRect(card, 10, 10)
            if selected or running:     # accent bar on the left edge of the card
                p.setPen(Qt.NoPen); p.setBrush(QtGui.QColor('#2563eb' if running else '#6366f1'))
                p.drawRoundedRect(QtCore.QRectF(card.left(), card.top() + 8, 4, card.height() - 16), 2, 2)
        # ---- connector line
        cx = r.left() + 20; cy = r.top() + r.height() / 2
        def seg(y0, y1, done):
            pen = QtGui.QPen(QtGui.QColor('#16a34a' if done else '#d1d5db'), 2.5); pen.setCapStyle(Qt.FlatCap)
            if not done and opt_: pen.setStyle(Qt.DashLine)
            p.setPen(pen); p.drawLine(QtCore.QPointF(cx, y0), QtCore.QPointF(cx, y1))
        if prv is not None: seg(r.top(), cy - 11, prv == 'done' and st in ('done', 'running'))
        if nxt is not None: seg(cy + 11, r.bottom() + 1, st == 'done' and nxt in ('done', 'running'))
        # ---- node
        if running:                      # pulsing halo
            ph = (time.time() * 1.6) % 1.0; rad = 11 + 7 * ph
            halo = QtGui.QColor('#2563eb'); halo.setAlphaF(0.35 * (1 - ph)); p.setPen(Qt.NoPen); p.setBrush(halo); p.drawEllipse(QtCore.QPointF(cx, cy), rad, rad)
        node = QtCore.QRectF(cx - 11, cy - 11, 22, 22)
        if st == 'not run':
            pen = QtGui.QPen(QtGui.QColor('#9ca3af'), 2); pen.setStyle(Qt.DashLine if opt_ else Qt.SolidLine)
            p.setPen(pen); p.setBrush(QtGui.QColor('#ffffff')); p.drawEllipse(node)
            p.setPen(QtGui.QColor('#6b7280')); f = p.font(); f.setPointSizeF(8.5); f.setBold(True); p.setFont(f); p.drawText(node, Qt.AlignCenter, str(num))
        else:
            p.setPen(Qt.NoPen); p.setBrush(col); p.drawEllipse(node); p.setPen(_pen('#ffffff', 2.2))
            if st == 'done': p.drawPolyline([QtCore.QPointF(cx - 5, cy + 0.5), QtCore.QPointF(cx - 1.5, cy + 4), QtCore.QPointF(cx + 5.5, cy - 4)])
            elif st in ('failed', 'interrupted'):
                p.drawLine(QtCore.QPointF(cx, cy - 6), QtCore.QPointF(cx, cy + 1.5)); p.drawPoint(QtCore.QPointF(cx, cy + 5))
            elif st == 'stale':
                p.setBrush(Qt.NoBrush); p.drawArc(QtCore.QRectF(cx - 5.5, cy - 5.5, 11, 11), 40 * 16, 280 * 16)
                p.drawLine(QtCore.QPointF(cx + 4.5, cy - 6.5), QtCore.QPointF(cx + 4.8, cy - 2.6))
            elif running:
                a = (time.time() * 360) % 360; p.setBrush(Qt.NoBrush); p.drawArc(QtCore.QRectF(cx - 6, cy - 6, 12, 12), int(-a * 16), 260 * 16)
        # ---- text
        x = card.left() + 14; w = card.width() - 20
        f = QtGui.QFont(opt.font); f.setPointSizeF(10.5); f.setBold(True); p.setFont(f); p.setPen(QtGui.QColor('#111827' if st != 'not run' or selected else '#374151'))
        t = f'{num}. {title}'; p.drawText(QtCore.QRectF(x, r.top() + 10, w, 22), Qt.AlignLeft | Qt.AlignVCenter, QtGui.QFontMetrics(f).elidedText(t, Qt.ElideRight, int(w)))
        # status pill + extras
        f2 = QtGui.QFont(opt.font); f2.setPointSizeF(8.5); f2.setBold(True); p.setFont(f2); fm = QtGui.QFontMetrics(f2)
        lab = LABEL.get(st, st) + (f' · {_fmt_secs(secs)}' if secs is not None and st in ('done', 'stale') else '')
        px, py = x, r.top() + 36
        def pill(text, fg, bg):
            nonlocal px
            tw = fm.horizontalAdvance(text) + 14; rr = QtCore.QRectF(px, py, tw, 18)
            p.setPen(Qt.NoPen); p.setBrush(QtGui.QColor(bg)); p.drawRoundedRect(rr, 9, 9); p.setPen(QtGui.QColor(fg)); p.drawText(rr, Qt.AlignCenter, text); px += tw + 6
        light = {'done': '#dcfce7', 'stale': '#fef3c7', 'running': '#dbeafe', 'failed': '#fee2e2', 'interrupted': '#fee2e2', 'not run': '#f3f4f6'}
        pill(lab, COL.get(st, '#6b7280') if st != 'not run' else '#6b7280', light.get(st, '#f3f4f6'))
        if opt_: pill('optional', '#6b7280', '#f3f4f6')
        if chk and st in ('done', 'stale'):
            if chk.get('fail'): pill(f'{chk["fail"]} fail', '#b91c1c', '#fee2e2')
            if chk.get('warn'): pill(f'{chk["warn"]} warn', '#b45309', '#fef3c7')
            if not chk.get('fail') and not chk.get('warn') and chk.get('ok'): pill(f'{chk["ok"]} checks ok', '#15803d', '#f0fdf4')
        p.restore()


def make_icon(kind, color='#ffffff', size=18):
    """Small vector icons drawn here (same look on every desktop theme)."""
    pm = QtGui.QPixmap(size * 2, size * 2); pm.fill(Qt.transparent); p = QtGui.QPainter(pm); p.setRenderHint(QtGui.QPainter.Antialiasing)
    s = size * 2; c = QtGui.QColor(color); p.setPen(_pen(color, s * 0.09)); p.setBrush(Qt.NoBrush)
    P = lambda *xy: [QtCore.QPointF(xy[i] * s, xy[i + 1] * s) for i in range(0, len(xy), 2)]
    if kind == 'play': p.setBrush(c); p.drawPolygon(P(.3, .2, .8, .5, .3, .8))
    elif kind == 'playall': p.setBrush(c); p.drawPolygon(P(.12, .22, .5, .5, .12, .78)); p.drawPolygon(P(.5, .22, .88, .5, .5, .78))
    elif kind == 'stop': p.setBrush(c); p.drawRoundedRect(QtCore.QRectF(.24 * s, .24 * s, .52 * s, .52 * s), s * .06, s * .06)
    elif kind == 'new':
        p.drawRoundedRect(QtCore.QRectF(.2 * s, .12 * s, .6 * s, .76 * s), s * .06, s * .06); p.drawLine(*P(.5, .35, .5, .65)); p.drawLine(*P(.35, .5, .65, .5))
    elif kind == 'open':
        p.drawPolyline(P(.1, .3, .1, .82, .82, .82, .9, .45, .28, .45, .2, .82)); p.drawPolyline(P(.1, .3, .38, .3, .45, .2, .78, .2, .78, .45))
    elif kind == 'folder': p.drawPolyline(P(.1, .25, .4, .25, .48, .35, .9, .35, .9, .8, .1, .8, .1, .25))
    elif kind == 'help':
        p.drawEllipse(QtCore.QRectF(.1 * s, .1 * s, .8 * s, .8 * s)); p.drawArc(QtCore.QRectF(.36 * s, .26 * s, .28 * s, .26 * s), 180 * 16, -250 * 16)
        p.drawLine(*P(.5, .52, .5, .6)); p.drawPoint(*P(.5, .72))
    p.end(); pm.setDevicePixelRatio(2); return QtGui.QIcon(pm)
