"""Inspection widgets: figure viewer, 3D model, frames, pairs/matches, marker clicking."""
import json, os, time
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from .flow import FlowLayout, group as fgroup, label as flabel
from .style import tidy
from PySide6.QtCore import Qt

_MODEL_CACHE = {}


def load_model(path):
    import pycolmap
    key = (path, os.path.getmtime(os.path.join(path, 'images.bin')) if os.path.exists(os.path.join(path, 'images.bin')) else 0)
    if key not in _MODEL_CACHE:
        if len(_MODEL_CACHE) > 2: _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
        _MODEL_CACHE[key] = pycolmap.Reconstruction(path)
    return _MODEL_CACHE[key]


def _safe_remove(gl, it):
    """Remove an item from a GL view if it is still there (a reload may already have cleared it)."""
    if it in gl.items: gl.removeItem(it)


def frame_label(name, k):
    """Number shown for a frame: the number in its file name (frame_000041.jpg -> 41), else its position + 1."""
    import re
    m = re.findall(r'\d+', os.path.splitext(os.path.basename(name))[0])
    return int(m[-1]) if m else k + 1


def models_of(p):
    d = p.path('models')
    return sorted(m for m in os.listdir(d) if not m.startswith('_') and os.path.exists(os.path.join(d, m, 'images.bin'))) if os.path.isdir(d) else []


def busy(fn):
    def w(*a, **k):
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try: return fn(*a, **k)
        except Exception as e: QtWidgets.QMessageBox.warning(None, 'FlexSfM Studio', f'{type(e).__name__}: {e}')
        finally: QtWidgets.QApplication.restoreOverrideCursor()
    return w


# ======================================================================================== zoomable image
ARROW_QSS = ('QToolButton { background: rgba(17,24,39,0.38); color: white; border: none; border-radius: 26px; font-size: 30px; font-weight: 700; }'
             'QToolButton:hover { background: rgba(37,99,235,0.85); }')


class ZoomView(QtWidgets.QGraphicsView):
    clicked = QtCore.Signal(float, float, int)   # image x, y, mouse button
    prev_requested = QtCore.Signal()
    next_requested = QtCore.Signal()

    def __init__(self, nav=False):
        super().__init__(); self.setScene(QtWidgets.QGraphicsScene(self)); self.item = None
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag); self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform); self._press = None
        self.nav = nav; self.arrows = []
        if nav:                                   # big round arrows at the left / right edge of the image
            for txt, sig, tip in (('‹', self.prev_requested, 'previous (← / Page Up)'), ('›', self.next_requested, 'next (→ / Page Down)')):
                b = QtWidgets.QToolButton(self); b.setText(txt); b.setFixedSize(52, 52); b.setStyleSheet(ARROW_QSS); b.setCursor(Qt.PointingHandCursor)
                b.setToolTip(tip); b.clicked.connect(sig.emit); b.hide(); self.arrows.append(b)
            self.setMouseTracking(True); self.setFocusPolicy(Qt.StrongFocus)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.arrows:
            y = self.height() // 2 - 26; self.arrows[0].move(12, y); self.arrows[1].move(self.width() - 64 - (self.verticalScrollBar().width() if self.verticalScrollBar().isVisible() else 0), y)

    def enterEvent(self, e):
        for b in self.arrows: b.show(); b.raise_()
        super().enterEvent(e)

    def leaveEvent(self, e):
        for b in self.arrows: b.hide()
        super().leaveEvent(e)

    def set_image(self, qimg, keep_view=False):
        had = self.item is not None; tr = self.transform(); c = self.mapToScene(self.viewport().rect().center())
        self.scene().clear(); self.item = self.scene().addPixmap(QtGui.QPixmap.fromImage(qimg)); self.setSceneRect(self.item.boundingRect())
        if keep_view and had: self.setTransform(tr); self.centerOn(c)
        else: self.fitInView(self.item, Qt.KeepAspectRatio)

    def wheelEvent(self, e):
        f = 1.25 if e.angleDelta().y() > 0 else 0.8; self.scale(f, f)

    def mousePressEvent(self, e):
        self._press = e.position(); super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        if self._press is not None and (e.position() - self._press).manhattanLength() < 4 and self.item:
            pt = self.mapToScene(e.position().toPoint()); self.clicked.emit(pt.x(), pt.y(), int(e.button().value))
        self._press = None

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_F and self.item: self.fitInView(self.item, Qt.KeepAspectRatio)
        elif self.nav and k in (Qt.Key_Left, Qt.Key_PageUp): self.prev_requested.emit()
        elif self.nav and k in (Qt.Key_Right, Qt.Key_PageDown, Qt.Key_Space): self.next_requested.emit()
        else: super().keyPressEvent(e)


class NavBar(QtWidgets.QWidget):
    """'◀ Prev   12 / 80   Next ▶' bar."""
    prev = QtCore.Signal()
    next = QtCore.Signal()

    def __init__(self, what='image'):
        super().__init__(); h = QtWidgets.QHBoxLayout(self); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
        self.bp = QtWidgets.QPushButton('◀  Prev'); self.bn = QtWidgets.QPushButton('Next  ▶'); self.lab = QtWidgets.QLabel('-'); self.lab.setObjectName('subtle')
        self.bp.setToolTip(f'previous {what} (← / Page Up)'); self.bn.setToolTip(f'next {what} (→ / Page Down)')
        self.bp.clicked.connect(self.prev.emit); self.bn.clicked.connect(self.next.emit)
        for b_ in (self.bp, self.bn): b_.setMinimumWidth(b_.fontMetrics().horizontalAdvance(b_.text()) + 40)
        self.lab.setMinimumWidth(self.lab.fontMetrics().horizontalAdvance('8888 / 88888') + 6); self.lab.setAlignment(Qt.AlignCenter)
        h.addWidget(self.bp); h.addWidget(self.lab); h.addWidget(self.bn)

    def set(self, i, n):
        self.lab.setText(f'{i + 1} / {n}' if n and i >= 0 else f'- / {n}'); self.bp.setEnabled(i > 0); self.bn.setEnabled(0 <= i < n - 1 or (i < 0 and n > 0))


def step_list(lw, d):
    """Move the current row of a QListWidget / QTableWidget by d (clamped)."""
    n = lw.count() if hasattr(lw, 'count') else lw.rowCount(); r = lw.currentRow()
    if n == 0: return
    r = min(n - 1, max(0, (r if r >= 0 else -1) + d))
    if isinstance(lw, QtWidgets.QTableWidget): lw.selectRow(r); lw.scrollToItem(lw.item(r, 0))
    else: lw.setCurrentRow(r)


class ImageView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__(); v = QtWidgets.QVBoxLayout(self); self.cap = QtWidgets.QLabel('Click a figure in Results to see it here. Wheel = zoom, drag = pan, F = fit.')
        self.cap.setWordWrap(True); v.addWidget(self.cap); self.view = ZoomView(); v.addWidget(self.view, 1); self.path = None
        b = QtWidgets.QPushButton('Open in system viewer'); b.clicked.connect(lambda: self.path and QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.path))); v.addWidget(b)

    def load(self, path, caption):
        self.path = path; self.cap.setText(f'<b>{caption}</b><br><span style="color:#666">{tidy(path)}</span>'); self.view.set_image(QtGui.QImage(path))


# ======================================================================================== 3D
def _make_flyview(gl_mod):
    """GLViewWidget with smooth orbit and first-person navigation for mouse and touchpad.

    Orbit:        left drag = rotate, right/middle drag or Shift+drag = pan, wheel / two-finger scroll / pinch = zoom towards the
                  cursor, double-click = rotate around that point, R = reset view.
    First person: W/A/S/D move, Q/E down/up, Shift = 4x faster, mouse drag = look, wheel / two-finger scroll / pinch = move
                  forward/back, Ctrl+wheel or +/- = speed. Movement accelerates and stops smoothly.
    """
    class FlyView(gl_mod.GLViewWidget):
        moved = QtCore.Signal()
        toggle_fly = QtCore.Signal()
        reset_view = QtCore.Signal()

        def __init__(self):
            super().__init__(); self.fly = False; self.speed = 5.0; self.keys = set(); self.vel = np.zeros(3); self.t_last = time.time()
            self.look_sens = 0.22; self.orbit_sens = 0.35; self.pick_pts = None; self.anim = None
            self.setFocusPolicy(Qt.StrongFocus); self.setMouseTracking(False); self.setAttribute(Qt.WA_AcceptTouchEvents, True)
            self.grabGesture(Qt.PinchGesture)
            self.tick = QtCore.QTimer(self); self.tick.setTimerType(Qt.PreciseTimer); self.tick.timeout.connect(self._step); self.tick.start(12)

        # ---------------------------------------------------------- camera helpers
        def _c(self):
            c = self.opts['center']; return np.array([c.x(), c.y(), c.z()])

        def _fwd(self):
            az, el = np.radians(self.opts['azimuth']), np.radians(self.opts['elevation'])
            return -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])   # eye -> centre

        def eye(self):
            return self._c() - self._fwd() * self.opts['distance']

        scene_ext = 100.0      # size of what is shown (m); sets the clipping planes

        def projectionMatrix(self, region=None, viewport=None):
            """pyqtgraph clips at 0.001x / 1000x the orbit distance - in first person (distance ~0) that hid everything beyond ~10 m.
            Here the near / far planes follow the scene size instead. Same frustum maths as pyqtgraph (fov = horizontal)."""
            if region is None: region = (0, 0, self.deviceWidth(), self.deviceHeight())
            if viewport is None: viewport = region
            x0, y0, w, h = viewport; dist = self.opts['distance']; ext = max(self.scene_ext, 1e-3)
            near = max(ext * 2e-5, min(dist * 0.001, ext * 1e-3)) if not self.fly else max(ext * 2e-5, 1e-3)
            far = max(dist * 1000, ext * 50)
            r = near * np.tan(0.5 * np.radians(self.opts['fov'])); t = r * h / w
            left = r * ((region[0] - x0) * (2.0 / w) - 1); right = r * ((region[0] + region[2] - x0) * (2.0 / w) - 1)
            bottom = t * ((region[1] - y0) * (2.0 / h) - 1); top = t * ((region[1] + region[3] - y0) * (2.0 / h) - 1)
            tr = QtGui.QMatrix4x4(); tr.frustum(left, right, bottom, top, near, far); return tr

        def _set(self, center=None, az=None, el=None, dist=None):
            if center is not None: self.opts['center'] = QtGui.QVector3D(*map(float, center))
            if az is not None: self.opts['azimuth'] = float(az)
            if el is not None: self.opts['elevation'] = float(np.clip(el, -89.5, 89.5))
            if dist is not None: self.opts['distance'] = float(max(dist, 1e-3))
            self.update(); self.moved.emit()

        def look_from(self, pos, direction, fov=None):
            """Put the eye at pos looking along direction (first-person)."""
            d = np.asarray(direction, float); d /= np.linalg.norm(d) + 1e-12
            dist = 0.01 if self.fly else self.opts['distance']
            if fov: self.opts['fov'] = float(fov)
            self._set(center=np.asarray(pos, float) + d * dist, az=np.degrees(np.arctan2(-d[1], -d[0])), el=np.degrees(np.arcsin(np.clip(-d[2], -1, 1))), dist=dist)

        def set_fly(self, on):
            e, f = self.eye(), self._fwd(); self.fly = on; self.vel[:] = 0
            dist = 0.01 if on else max(self.speed * 20, 10.0)
            self._set(center=e + f * dist, dist=dist)

        def _right_up(self):
            f = self._fwd(); r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r) + 1e-12; u = np.cross(r, f); return r, u

        # ---------------------------------------------------------- picking (double-click = new rotation centre)
        def pick(self, x, y, radius=12):
            if self.pick_pts is None or not len(self.pick_pts): return None
            dpr = self.devicePixelRatioF(); w, h = self.width() * dpr, self.height() * dpr
            vp = (0, 0, int(w), int(h))
            m = self.projectionMatrix(vp, vp) * self.viewMatrix(); M = np.array(m.data(), dtype=np.float64).reshape(4, 4).T
            P = self.pick_pts; Q = P @ M[:3, :3].T + M[:3, 3]; wq = P @ M[3, :3] + M[3, 3]
            ok = wq > 1e-6; sx = (Q[:, 0] / wq * 0.5 + 0.5) * w; sy = (1 - (Q[:, 1] / wq * 0.5 + 0.5)) * h
            d2 = (sx - x * dpr) ** 2 + (sy - y * dpr) ** 2; d2[~ok] = np.inf
            near = np.flatnonzero(d2 < (radius * dpr) ** 2)
            if not len(near): return None
            e = self.eye(); depth = np.linalg.norm(P[near] - e, axis=1)
            return P[near[np.argmin(depth + 0.02 * np.sqrt(d2[near]) * depth.min() / max(radius, 1))]]

        def _animate_to(self, center, dist, ms=280):
            self.anim = (time.time(), ms / 1000, self._c(), self.opts['distance'], np.asarray(center, float), float(dist))

        # ---------------------------------------------------------- per-frame update (movement with acceleration / damping)
        def _step(self):
            now = time.time(); dt = min(0.05, now - self.t_last); self.t_last = now
            if self.anim:
                t0, T, c0, d0, c1, d1 = self.anim; a = min(1.0, (now - t0) / T); a = a * a * (3 - 2 * a)
                self._set(center=c0 + (c1 - c0) * a, dist=d0 * (d1 / d0) ** a)
                if a >= 1: self.anim = None
            if not self.fly: return
            f = self._fwd(); r, _ = self._right_up(); up = np.array([0, 0, 1.0]); k = self.keys; want = np.zeros(3)
            if Qt.Key_W in k or Qt.Key_Up in k: want += f
            if Qt.Key_S in k or Qt.Key_Down in k: want -= f
            if Qt.Key_D in k or Qt.Key_Right in k: want += r
            if Qt.Key_A in k or Qt.Key_Left in k: want -= r
            if Qt.Key_E in k or Qt.Key_Space in k: want += up
            if Qt.Key_Q in k or Qt.Key_C in k: want -= up
            n = np.linalg.norm(want)
            if n: want /= n
            boost = 4.0 if QtWidgets.QApplication.keyboardModifiers() & Qt.ShiftModifier else 1.0
            target = want * self.speed * boost
            self.vel += (target - self.vel) * min(1.0, dt * (9.0 if n else 7.0))      # ease in / ease out
            if np.linalg.norm(self.vel) < 1e-3 * max(self.speed, 1e-3): self.vel[:] = 0; return
            self._set(center=self._c() + self.vel * dt)

        # ---------------------------------------------------------- keyboard
        MOVE_KEYS = {Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_Q, Qt.Key_E, Qt.Key_C, Qt.Key_Space, Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right}

        def keyPressEvent(self, ev):
            if ev.isAutoRepeat(): return
            k = ev.key()
            if self.fly and k in self.MOVE_KEYS: self.keys.add(k); return
            if k in (Qt.Key_Plus, Qt.Key_Equal): self.speed = min(2000, self.speed * 1.25); self.moved.emit(); return
            if k == Qt.Key_Minus: self.speed = max(0.05, self.speed / 1.25); self.moved.emit(); return
            if k == Qt.Key_F: self.toggle_fly.emit(); return
            if k == Qt.Key_R: self.reset_view.emit(); return
            if not self.fly and k in self.MOVE_KEYS: return       # keep arrow keys from doing pyqtgraph's own jumps
            super().keyPressEvent(ev)

        def keyReleaseEvent(self, ev):
            if not ev.isAutoRepeat(): self.keys.discard(ev.key())

        def focusOutEvent(self, ev):
            self.keys.clear(); super().focusOutEvent(ev)

        # ---------------------------------------------------------- mouse
        def mousePressEvent(self, ev):
            self._lp = ev.position(); self.setFocus(); self.anim = None; ev.accept()

        def mouseMoveEvent(self, ev):
            pos = ev.position(); d = pos - getattr(self, '_lp', pos); self._lp = pos; b = ev.buttons(); mods = ev.modifiers()
            if self.fly:
                if b & (Qt.LeftButton | Qt.RightButton | Qt.MiddleButton):       # look around, eye stays where it is
                    e = self.eye(); az = self.opts['azimuth'] - d.x() * self.look_sens; el = self.opts['elevation'] + d.y() * self.look_sens
                    self.opts['azimuth'] = az; self.opts['elevation'] = float(np.clip(el, -89.5, 89.5))
                    self._set(center=e + self._fwd() * self.opts['distance'])
                return
            if (b & Qt.LeftButton) and not (mods & Qt.ShiftModifier):
                self._set(az=self.opts['azimuth'] - d.x() * self.orbit_sens, el=self.opts['elevation'] + d.y() * self.orbit_sens)
            elif b & (Qt.RightButton | Qt.MiddleButton) or ((b & Qt.LeftButton) and (mods & Qt.ShiftModifier)):
                r, u = self._right_up()
                k = 2 * self.opts['distance'] * np.tan(np.radians(self.opts['fov']) / 2) / max(1, self.height())   # world units per pixel
                self._set(center=self._c() - r * d.x() * k + u * d.y() * k)

        def mouseReleaseEvent(self, ev):
            ev.accept()

        def mouseDoubleClickEvent(self, ev):
            if self.fly: return
            P = self.pick(ev.position().x(), ev.position().y())
            if P is not None: self._animate_to(P, np.linalg.norm(self.eye() - P))

        def _zoom(self, factor, at=None):
            """factor < 1 = closer. In orbit mode zoom towards the point under the cursor; in first person move forward/back."""
            if self.fly:
                self._set(center=self._c() + self._fwd() * (1 - factor) * self.speed * 3); return
            dist = self.opts['distance']; new = dist * factor
            if at is not None:
                P = self.pick(at.x(), at.y(), radius=20)
                if P is not None:                                   # slide the centre towards the point so it stays under the cursor
                    c = self._c(); self._set(center=c + (P - c) * (1 - factor), dist=new); return
            self._set(dist=new)

        def wheelEvent(self, ev):
            pd, ad = ev.pixelDelta(), ev.angleDelta()
            if not pd.isNull(): steps = pd.y() / 60.0                  # touchpad: many small pixel steps, gentle
            else: steps = ad.y() / 120.0                                # mouse wheel: one notch = 1 step
            if ev.modifiers() & Qt.ControlModifier:
                self.speed = float(np.clip(self.speed * (1.2 ** steps), 0.05, 2000)); self.moved.emit(); return
            self._zoom(0.85 ** steps, ev.position()); ev.accept()

        def event(self, ev):
            if ev.type() == QtCore.QEvent.NativeGesture:              # touchpad pinch (Wayland / X11 with libinput)
                try:
                    if ev.gestureType() == Qt.ZoomNativeGesture: self._zoom(1 - ev.value(), ev.position()); return True
                except Exception: pass
            if ev.type() == QtCore.QEvent.Gesture:
                g = ev.gesture(Qt.PinchGesture)
                if g is not None:
                    sf = g.scaleFactor()
                    if sf > 0: self._zoom(1 / sf)
                    return True
            return super().event(ev)
    return FlyView


CONTROLS_HTML = '''<table cellspacing="0" cellpadding="3">
<tr><td colspan="2"><b>Orbit</b> (default)</td><td width="40"></td><td colspan="2"><b>First person</b> (tick it, or press F)</td></tr>
<tr><td style="color:#6b7280">left drag</td><td>rotate</td><td></td><td style="color:#6b7280">W A S D</td><td>forward / left / back / right</td></tr>
<tr><td style="color:#6b7280">right / middle drag, Shift+drag</td><td>pan</td><td></td><td style="color:#6b7280">Q / E</td><td>down / up</td></tr>
<tr><td style="color:#6b7280">wheel, two-finger scroll, pinch</td><td>zoom towards the cursor</td><td></td><td style="color:#6b7280">Shift</td><td>4x faster</td></tr>
<tr><td style="color:#6b7280">double-click a point</td><td>rotate around it</td><td></td><td style="color:#6b7280">drag</td><td>look around</td></tr>
<tr><td style="color:#6b7280">R</td><td>reset the view</td><td></td><td style="color:#6b7280">wheel</td><td>move forward / back</td></tr>
<tr><td style="color:#6b7280">[ &nbsp; ]</td><td>previous / next image</td><td></td><td style="color:#6b7280">Ctrl+wheel, + / -</td><td>speed</td></tr>
</table>'''


class Model3D(QtWidgets.QWidget):
    """Points + cameras (frustums) + GPS track. Orbit or first-person (WASD) navigation; jump into any camera / GPS position."""

    def __init__(self):
        super().__init__(); self.p = None; self.gl = None; self.rec = None; self.cams_c = None
        v = QtWidgets.QVBoxLayout(self); v.setSpacing(6)
        groups = FlowLayout(hspacing=10, vspacing=8); v.addLayout(groups)

        def group(title):
            g = QtWidgets.QGroupBox(title); lay = QtWidgets.QHBoxLayout(g); lay.setContentsMargins(10, 6, 10, 6); lay.setSpacing(6); groups.addWidget(g); return lay

        def lab(t):
            l = QtWidgets.QLabel(t); l.setStyleSheet('color:#4b5563'); return l
        # ---- model
        r = group('Model')
        self.model = QtWidgets.QComboBox(); self.model.setMinimumWidth(130); self.model.setToolTip('which reconstruction to show'); r.addWidget(self.model)
        self.other = QtWidgets.QPushButton('Other...'); self.other.setToolTip('open any COLMAP model folder (e.g. a Metashape export)'); self.other.clicked.connect(self.pick_other); r.addWidget(self.other)
        b = QtWidgets.QPushButton('Load'); b.setObjectName('primary'); b.setToolTip('load / reload the selected model'); b.clicked.connect(self.load); r.addWidget(b)
        self.model.activated.connect(lambda _: self.load())
        # ---- points
        r = group('Points')
        self.color = QtWidgets.QComboBox(); self.color.addItems(['image colour', 'reprojection error', 'track length', 'height']); self.color.setToolTip('colour the points by')
        r.addWidget(lab('colour')); r.addWidget(self.color)
        self.size = QtWidgets.QDoubleSpinBox(); self.size.setRange(0.5, 10); self.size.setValue(2.5); self.size.setSingleStep(0.5); self.size.setToolTip('point size on screen (pixels)')
        self.size.valueChanged.connect(self.resize_pts); r.addWidget(lab('size')); r.addWidget(self.size)
        self.maxp = QtWidgets.QSpinBox(); self.maxp.setRange(10000, 10_000_000); self.maxp.setSingleStep(250000); self.maxp.setValue(5_000_000); self.maxp.setGroupSeparatorShown(True)
        self.maxp.setToolTip('largest number of points drawn (a random sample is shown above this)'); r.addWidget(lab('max')); r.addWidget(self.maxp)
        self.bgbtn = QtWidgets.QToolButton(); self.bgbtn.setCheckable(True); self.bgbtn.setText('☀'); self.bgbtn.setToolTip('background: light (click for dark)')
        self.bgbtn.setStyleSheet('QToolButton { font-size: 16px; padding: 1px 8px; border: 1px solid #d1d5db; border-radius: 6px; background: #ffffff; }')
        self.bgbtn.toggled.connect(self.apply_bg); r.addWidget(self.bgbtn)
        # ---- cameras / overlays
        r = group('Show')
        self.cams = QtWidgets.QCheckBox('cameras'); self.cams.setChecked(True); self.cams.setToolTip('camera frustums'); self.cams.toggled.connect(self.draw_overlays); r.addWidget(self.cams)
        self.fsize = QtWidgets.QDoubleSpinBox(); self.fsize.setRange(0.01, 1000); self.fsize.setDecimals(2); self.fsize.setSingleStep(0.10); self.fsize.setValue(1.0); self.fsize.setSuffix(' m')
        self.fsize.setToolTip('size of the camera frustums (depth of the pyramid)'); self.fsize.valueChanged.connect(self.draw_overlays); r.addWidget(self.fsize)
        self.every = QtWidgets.QSpinBox(); self.every.setRange(1, 1000); self.every.setValue(1); self.every.setPrefix('every '); self.every.setToolTip('draw every Nth camera')
        self.every.valueChanged.connect(self.draw_overlays); r.addWidget(self.every)
        self.path = QtWidgets.QCheckBox('path'); self.path.setChecked(True); self.path.setToolTip('camera path, coloured along the sequence: violet = first frame, amber = last'); self.path.toggled.connect(self.draw_overlays); r.addWidget(self.path)
        self.gps = QtWidgets.QCheckBox('GPS'); self.gps.setChecked(True); self.gps.setToolTip('GPS track (blue dots and line)'); self.gps.toggled.connect(self.draw_overlays); r.addWidget(self.gps)
        # ---- navigation
        r = group('Navigate')
        self.fly = QtWidgets.QCheckBox('first person'); self.fly.setToolTip('fly with W/A/S/D (F key)'); self.fly.toggled.connect(self.toggle_fly); r.addWidget(self.fly)
        self.follow = QtWidgets.QCheckBox('look through camera'); self.follow.setToolTip('When stepping / playing through the images, put the view inside the camera. '
                                                                                     'Off: the current camera is highlighted and you keep looking from wherever you like.')
        r.addWidget(self.follow)
        self.src = QtWidgets.QComboBox(); self.src.addItems(['cameras', 'GPS positions']); self.src.setToolTip('step through the estimated cameras or the GPS positions'); r.addWidget(self.src)
        bp = QtWidgets.QToolButton(); bp.setText('◀'); bp.setToolTip('previous image  ( [ )'); bp.clicked.connect(lambda: self.frame.setValue(self.frame.value() - 1)); r.addWidget(bp)
        self.frame = QtWidgets.QSpinBox(); self.frame.setRange(1, 1); self.frame.setPrefix('image '); self.frame.setToolTip('position in the image list (1 = first image)')
        self.frame.valueChanged.connect(lambda v: self.goto(v - 1)); r.addWidget(self.frame)
        bn = QtWidgets.QToolButton(); bn.setText('▶'); bn.setToolTip('next image  ( ] )'); bn.clicked.connect(lambda: self.frame.setValue(self.frame.value() + 1)); r.addWidget(bn)
        self.play = QtWidgets.QPushButton('▶▶ play'); self.play.setCheckable(True); self.play.setToolTip('step through all images automatically'); self.play.toggled.connect(self.toggle_play); r.addWidget(self.play)
        # ---- photo overlay
        r = group('Photo overlay')
        self.opac = QtWidgets.QSlider(Qt.Horizontal); self.opac.setRange(0, 100); self.opac.setValue(50); self.opac.setFixedWidth(110)
        self.opac.setToolTip('when looking through a camera: how strongly its photo is shown over the points (0 % = only the points)')
        self.opac_lab = QtWidgets.QLabel('50 %'); self.opac_lab.setMinimumWidth(36); r.addWidget(self.opac); r.addWidget(self.opac_lab)
        # ---- help
        r = group('Help')
        self.helpbtn = QtWidgets.QToolButton(); self.helpbtn.setText('?  controls'); self.helpbtn.setCheckable(True); self.helpbtn.setToolTip('show / hide the mouse and keyboard controls')
        r.addWidget(self.helpbtn)
        self.help_panel = QtWidgets.QLabel(CONTROLS_HTML); self.help_panel.setTextFormat(Qt.RichText); self.help_panel.hide()
        self.help_panel.setStyleSheet('background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; padding:8px'); v.addWidget(self.help_panel)
        self.helpbtn.toggled.connect(self.help_panel.setVisible)
        self.info = QtWidgets.QLabel('Pick a model and press Load.'); self.info.setObjectName('subtle'); self.info.setWordWrap(True); v.addWidget(self.info)
        self.holder = QtWidgets.QVBoxLayout(); v.addLayout(self.holder, 1); self.other_path = None; self.overlay_items = []
        self.hud = QtWidgets.QLabel(''); self.hud.setObjectName('subtle'); v.addWidget(self.hud)
        self.pip = QtWidgets.QLabel(self); self.pip.setStyleSheet('background:#ffffff; border:2px solid #dc2626; border-radius:6px; padding:2px'); self.pip.hide()
        self.pip.setToolTip('image of the highlighted camera'); self.cur_k = None; self.hl_items = []; self.inside = False
        self.photo = QtWidgets.QLabel(self); self.photo.setAttribute(Qt.WA_TransparentForMouseEvents); self.photo.hide(); self.photo.setScaledContents(True)
        self.photo_fx = QtWidgets.QGraphicsOpacityEffect(self.photo); self.photo.setGraphicsEffect(self.photo_fx)
        self.opac.valueChanged.connect(lambda v_: (self.photo_fx.setOpacity(v_ / 100), self.opac_lab.setText(f'{v_} %'))); self.photo_fx.setOpacity(0.5)
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(lambda: self.frame.setValue(self.frame.value() + 1 if self.frame.value() < self.frame.maximum() else 1))
        QtGui.QShortcut(QtGui.QKeySequence('['), self, activated=lambda: self.frame.setValue(self.frame.value() - 1))
        QtGui.QShortcut(QtGui.QKeySequence(']'), self, activated=lambda: self.frame.setValue(self.frame.value() + 1))

    def set_project(self, p):
        new = self.p is None or p.root != self.p.root
        self.p = p; cur = self.model.currentText(); ms = models_of(p); self.model.clear(); self.model.addItems(ms)
        if cur and not new and self.model.findText(cur) >= 0: self.model.setCurrentText(cur)
        else:                                          # show the richest cloud by default (dense RGB points), else the current model
            pick = next((m for m in ('dense', 'dense_raw') if m in ms), p.state.get('model'))
            if pick: self.model.setCurrentText(pick)
        self.frame.setMaximum(max(1, len(p.names())))
        if new: self.rec = None; self.loaded_key = None
        if self.isVisible(): self.activate()

    def activate(self):
        """Called when the tab is shown: load the model automatically if one exists and nothing is shown yet."""
        if self.p and self.model.currentText() and getattr(self, 'loaded_key', None) != (self.p.root, self.model.currentText()):
            QtCore.QTimer.singleShot(50, self.load)

    def apply_bg(self, *_):
        dark = self.bgbtn.isChecked(); self.bgbtn.setText('☾' if dark else '☀'); self.bgbtn.setToolTip('background: dark (click for light)' if dark else 'background: light (click for dark)')
        if self.gl is not None: self.gl.setBackgroundColor('#111827' if dark else '#ffffff')
        self.draw_overlays()

    def resizeEvent(self, e):
        super().resizeEvent(e); self._place_pip()

    def _place_pip(self):
        self._place_photo()
        if self.gl is not None and self.pip.isVisible():
            g = self.gl.geometry(); self.pip.move(g.right() - self.pip.width() - 12, g.top() + 12); self.pip.raise_()

    def pick_other(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, 'COLMAP model folder (cameras/images/points3D)',
                                                       os.path.dirname(self.p.root) if getattr(self, 'p', None) else os.path.expanduser('~'))
        if d: self.other_path = d; self.model.addItem('other: ' + d); self.model.setCurrentIndex(self.model.count() - 1)

    @staticmethod
    def gl_available():
        """True when a desktop OpenGL >= 2.1 context can be created (not on headless / offscreen sessions)."""
        if QtWidgets.QApplication.platformName() in ('offscreen', 'minimal'): return False
        ctx = QtGui.QOpenGLContext()
        if not ctx.create(): return False
        v = ctx.format().version(); return ctx.isOpenGLES() or v >= (2, 1)

    def _gl(self):
        if self.gl is None:
            if not self.gl_available():
                raise RuntimeError('The 3D view needs OpenGL 2.1 or newer, which this session does not provide (remote / headless display?). '
                                   'Open export/points.ply or models/<name> in another viewer instead (e.g. CloudCompare, COLMAP GUI).')
            import pyqtgraph.opengl as gl
            self.glmod = gl; self.gl = _make_flyview(gl)(); self.gl.setBackgroundColor('#111827' if self.bgbtn.isChecked() else '#ffffff'); self.holder.addWidget(self.gl)
            self.gl.moved.connect(self.update_hud); self.gl.moved.connect(self._moved); self.gl.fly = self.fly.isChecked()
            self.photo.setParent(self.gl); self.photo.setGraphicsEffect(self.photo_fx)   # child of the 3D view: clipped to it
            self.gl.toggle_fly.connect(lambda: self.fly.setChecked(not self.fly.isChecked())); self.gl.reset_view.connect(self.reset_view)
        return self.gl

    @busy
    def load(self):
        if not self.p or not self.model.currentText(): return
        t = self.model.currentText(); path = t[7:] if t.startswith('other: ') else self.p.model_dir(t)
        rec = load_model(path); gl = self._gl(); g = self.glmod; self.rec = rec; self.is_other = t.startswith('other'); self.loaded_key = (self.p.root, t); self.cur_k = None; self.pip.hide(); self.inside = False; self.photo.hide()
        for it in list(gl.items): _safe_remove(gl, it)
        self.overlay_items = []; self.hl_items = []
        pts = list(rec.points3D.values()); n = len(pts)
        sel = np.random.default_rng(0).choice(n, min(n, self.maxp.value()), replace=False) if n > self.maxp.value() else np.arange(n)
        X = np.array([pts[i].xyz for i in sel], np.float32).reshape(-1, 3)
        mode = self.color.currentText()
        if mode == 'image colour': C = np.array([pts[i].color for i in sel], np.float32).reshape(-1, 3) / 255
        else:
            if mode == 'reprojection error': rec.update_point_3d_errors(); v = np.array([pts[i].error for i in sel]); v = np.clip(v / 3, 0, 1)
            elif mode == 'track length': v = np.clip((np.array([pts[i].track.length() for i in sel]) - 2) / 20, 0, 1)
            else: z = X[:, 2]; lo, hi = np.percentile(z, [2, 98]) if len(z) else (0, 1); v = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
            import matplotlib.cm as cm; C = cm.turbo(v)[:, :3].astype(np.float32)
        # cameras in project frame order (frame index = position in images.txt)
        names = self.p.names(); by = {im.name: im for im in rec.images.values() if im.has_pose}
        self.cam_list = [(k, by[nm]) for k, nm in enumerate(names) if nm in by] or [(k, im) for k, im in enumerate(sorted(by.values(), key=lambda i: i.name))]
        Cc = np.array([im.projection_center() for _, im in self.cam_list], np.float32).reshape(-1, 3)
        o = np.median(X, 0) if len(X) else (np.median(Cc, 0) if len(Cc) else np.zeros(3)); self.o = o.astype(np.float64)
        col = np.ascontiguousarray(np.c_[C, np.ones(len(C))], dtype=np.float32)
        self.sc = g.GLScatterPlotItem(pos=np.ascontiguousarray(X - o, dtype=np.float32), color=col, size=self.size.value(), pxMode=True)
        self.sc.setGLOptions('opaque'); gl.addItem(self.sc); gl.pick_pts = (X - o).astype(np.float64)[:: max(1, len(X) // 400000)]
        ext = np.ptp(Cc, 0).max() if len(Cc) > 1 else (np.ptp(X, 0).max() if len(X) else 10)
        self.fsize.blockSignals(True); self.fsize.setValue(round(max(0.02, float(ext) / 120), 2)); self.fsize.blockSignals(False)
        gl.speed = max(0.5, float(ext) / 40); gl.scene_ext = float(max(np.ptp(X, 0).max() if len(X) else 0, ext, 1.0))
        self.draw_overlays()
        lo, hi = (np.percentile(X, 2, 0), np.percentile(X, 98, 0)) if len(X) else (np.zeros(3), np.ones(3) * 10); self.home = float(np.max(hi - lo)) * 1.1
        if not self.fly.isChecked(): self.reset_view()
        else: self.goto(self.frame.value() - 1)
        gps = self.p.gps()
        self.info.setText(f'{tidy(path)}: {rec.num_reg_images()} cameras, {n:,} points (' + ('all shown' if len(sel) == n else f'{len(sel):,} shown - raise "max points" to see all') + '). Cameras and path coloured along the sequence (violet = first frame, amber = last)'
                          + ('; blue dots = GPS' if gps is not None and not self.is_other else '')
                          + '.   Controls: press "? controls".')

    def draw_overlays(self):
        if self.gl is None or self.rec is None: return
        gl, g, o = self.gl, self.glmod, self.o
        for it in self.overlay_items: _safe_remove(gl, it)
        self.overlay_items = []
        Cc = np.array([im.projection_center() for _, im in self.cam_list], np.float64).reshape(-1, 3)
        if getattr(self, 'inside', False): return              # looking through a camera: no frustums / path in front of the lens
        if self.path.isChecked() and len(Cc) > 1:
            it = g.GLLinePlotItem(pos=(Cc - o).astype(np.float32), color=self._seq_colors(len(Cc)), width=3.5, mode='line_strip', antialias=True)
            gl.addItem(it); self.overlay_items.append(it)
        if self.cams.isChecked() and len(Cc):
            s = self.fsize.value(); seg = []; scol = []; seqc = self._seq_colors(len(self.cam_list)); kpos = {k_: i_ for i_, (k_, _) in enumerate(self.cam_list)}
            skip = getattr(self, 'near', set()) if self.fly.isChecked() else set()
            for k_, im in self.cam_list[:: self.every.value()]:
                if k_ in skip: continue
                cam = self.rec.cameras[im.camera_id]; R = im.cam_from_world().rotation.matrix(); c = im.projection_center() - o
                fx = cam.focal_length_x if hasattr(cam, 'focal_length_x') else cam.mean_focal_length(); fy = cam.focal_length_y if hasattr(cam, 'focal_length_y') else fx
                hw, hh = cam.width / 2 / fx * s, cam.height / 2 / fy * s
                corners = [R.T @ np.array([x, y, s]) + c for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
                for q in corners: seg += [c, q]
                for a_, b_ in ((0, 1), (1, 2), (2, 3), (3, 0)): seg += [corners[a_], corners[b_]]
                up = R.T @ np.array([0, -hh * 1.35, s]) + c; seg += [corners[0], up, up, corners[1]]          # small "roof" marks the image top
                scol += [seqc[kpos[k_]]] * (len(seg) - len(scol))
            it = g.GLLinePlotItem(pos=np.array(seg, np.float32), color=np.array(scol, np.float32), width=1.6, mode='lines', antialias=True); gl.addItem(it); self.overlay_items.append(it)
        gps = self.p.gps() if self.p else None
        if self.gps.isChecked() and gps is not None and not getattr(self, 'is_other', False):
            gc = self._c('gps'); gl_ = (gc[0], gc[1], gc[2], 0.45)
            it = g.GLLinePlotItem(pos=(gps['enu'][gps['valid']] - o).astype(np.float32), color=gl_, width=1.5, antialias=True); it.setGLOptions('translucent'); gl.addItem(it); self.overlay_items.append(it)
            sel = np.flatnonzero(gps['valid'])
            if len(sel): it = g.GLScatterPlotItem(pos=(gps['enu'][sel] - o).astype(np.float32), color=self._c('gps'), size=7, pxMode=True); gl.addItem(it); self.overlay_items.append(it)

    def reset_view(self):
        if self.gl is None: return
        if self.fly.isChecked(): self.fly.setChecked(False)
        self.gl._set(center=np.zeros(3), az=-90, el=55, dist=getattr(self, 'home', 100.0))

    def _c(self, what):
        white = not self.bgbtn.isChecked()
        return {'path': (.85, .35, 0, 1) if white else (1, .6, .1, 1), 'frustum': (.75, .45, .0, 1) if white else (1, .78, .25, 1),
                'gps': (.1, .4, .95, 1) if white else (.3, .75, 1, 1)}[what]

    def _seq_colors(self, n):
        """One colour per camera along the sequence: violet (first) -> teal -> amber (last). Same colours for path and frustums."""
        stops = np.array([[0.42, 0.20, 0.75], [0.05, 0.55, 0.62], [0.95, 0.62, 0.05]]) if not self.bgbtn.isChecked() else np.array([[0.75, 0.55, 1.0], [0.3, 0.9, 0.9], [1.0, 0.8, 0.3]])
        t = np.linspace(0, 1, max(n, 1)) * 2; i = np.minimum(t.astype(int), 1); f = (t - i)[:, None]
        rgb = stops[i] * (1 - f) + stops[i + 1] * f; return np.c_[rgb, np.ones(len(rgb))].astype(np.float32)

    FPV_FRUSTUM = 0.10     # m - small frustums while flying through the scene

    def toggle_fly(self, on):
        if self.gl is None: return
        if on: self._orbit_fsize = self.fsize.value(); self.fsize.setValue(self.FPV_FRUSTUM)
        elif getattr(self, '_orbit_fsize', None): self.fsize.setValue(self._orbit_fsize); self._orbit_fsize = None
        self.gl.set_fly(on); self.gl.setFocus()
        if on and self.rec is not None and self.cur_k is None and self.cam_list:      # start inside the first camera
            self.frame.blockSignals(True); self.frame.setValue(self.cam_list[0][0] + 1); self.frame.blockSignals(False); self.enter_camera(self.cam_list[0][0])
        if not on: self.leave_camera()
        self.update_hud()

    def enter_camera(self, k):
        """Put the eye exactly where camera k was, with its field of view, hide frustums / path and show its photo over the view."""
        idx = {kk: im for kk, im in self.cam_list}
        if k not in idx: self.hud.setText(f'image {k + 1}: not placed in this model'); return False
        im = idx[k]; cam = self.rec.cameras[im.camera_id]; self.highlight(k)
        self.gl.vel[:] = 0; self.gl.keys.clear(); self.gl.anim = None            # no leftover motion
        fx = cam.focal_length_x if hasattr(cam, 'focal_length_x') else cam.mean_focal_length()
        fovx = 2 * np.degrees(np.arctan(cam.width / 2 / fx))                      # pyqtgraph's fov is horizontal
        if not self.fly.isChecked():
            self._orbit_fsize = self.fsize.value(); self.fsize.blockSignals(True); self.fsize.setValue(self.FPV_FRUSTUM); self.fsize.blockSignals(False)
            self.fly.blockSignals(True); self.fly.setChecked(True); self.fly.blockSignals(False); self.gl.set_fly(True)
        self._eye_in = None; self.inside = True; self.draw_overlays()
        for it in self.hl_items: _safe_remove(self.gl, it)
        self.hl_items = []; self.pip.hide()
        self.gl.look_from(im.projection_center() - self.o, im.viewing_direction(), fov=fovx)
        self._eye_in = self.gl.eye().copy(); self._az_in = (self.gl.opts['azimuth'], self.gl.opts['elevation'])
        names = self.p.names(); r = QtGui.QImageReader(os.path.join(self.p.images_dir, names[k])); img = r.read()
        if not img.isNull():
            self.photo.setPixmap(QtGui.QPixmap.fromImage(img)); self._photo_aspect = cam.height / cam.width; self.photo.show(); self._place_photo()
        return True

    def leave_camera(self):
        if self.inside: self.inside = False; self.photo.hide(); self.draw_overlays()

    def _place_photo(self):
        """The photo covers the same part of the view as the camera's image: full width, height from its aspect ratio, centred."""
        if self.gl is None or not self.photo.isVisible(): return
        gw, gh = self.gl.width(), self.gl.height(); h = int(gw * getattr(self, '_photo_aspect', 0.5625))
        self.photo.setGeometry(0, (gh - h) // 2, gw, h); self.photo.raise_()

    def _moved(self):
        """Leaving the camera's exact spot hides the photo (it would no longer line up) and brings the frustums back."""
        if self.inside and self.gl is not None and getattr(self, '_eye_in', None) is not None:
            if np.linalg.norm(self.gl.eye() - self._eye_in) > 1e-3 or abs(self.gl.opts['azimuth'] - self._az_in[0]) > 0.05 or abs(self.gl.opts['elevation'] - self._az_in[1]) > 0.05:
                self.leave_camera()

    def toggle_play(self, on):
        if on: self.timer.start(250 if not self.follow.isChecked() else 120)
        else: self.timer.stop()
        self.play.setText('pause' if on else 'play')

    def goto(self, k):
        """Jump the eye into frame k: its estimated camera (looking where it looked) or its GPS position (looking along the track)."""
        if self.gl is None or self.rec is None or self.p is None: return
        self.highlight(k)
        if not self.follow.isChecked():                                  # free view: only highlight the camera
            self.leave_camera(); self.update_hud(); return
        if not self.fly.isChecked(): self.fly.setChecked(True)
        if self.src.currentIndex() == 0:
            self.enter_camera(k)
        else:
            self.leave_camera()
            gps = self.p.gps()
            if gps is None: self.hud.setText('this project has no GPS'); return
            if not gps['valid'][k]: self.hud.setText(f'frame {k}: no GPS fix'); return
            ok = np.flatnonzero(gps['valid']); j = ok[np.searchsorted(ok, k)]; nxt = ok[min(len(ok) - 1, np.searchsorted(ok, k) + 3)]
            d = gps['enu'][nxt] - gps['enu'][j] if nxt != j else np.array([1.0, 0, 0]); d[2] = 0
            if not d.any(): d = np.array([1.0, 0, 0])
            d = d / np.linalg.norm(d) + np.array([0, 0, -0.35])      # along the track, tilted a little down
            self.gl.look_from(gps['enu'][j] - self.o, d)
        self.update_hud()

    def highlight(self, k):
        """Thick red frustum on the current camera (or a marker on its GPS position) + its image in the corner."""
        g, gl = self.glmod, self.gl
        for it in self.hl_items: _safe_remove(gl, it)
        self.hl_items = []; self.cur_k = k; names = self.p.names()
        idx = {kk: im for kk, im in self.cam_list}; o = self.o
        if self.src.currentIndex() == 0 and k in idx:
            im = idx[k]; cam = self.rec.cameras[im.camera_id]; R = im.cam_from_world().rotation.matrix(); c = im.projection_center() - o
            s = self.fsize.value() * 2.2; f = cam.mean_focal_length(); hw, hh = cam.width / 2 / f * s, cam.height / 2 / f * s
            cr = [R.T @ np.array([x, y, s]) + c for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]; seg = []
            for q in cr: seg += [c, q]
            for a_, b_ in ((0, 1), (1, 2), (2, 3), (3, 0)): seg += [cr[a_], cr[b_]]
            it = g.GLLinePlotItem(pos=np.array(seg, np.float32), color=(.86, .15, .15, 1), width=3.5, mode='lines', antialias=True); gl.addItem(it); self.hl_items.append(it)
            it = g.GLScatterPlotItem(pos=np.array([c], np.float32), color=(.86, .15, .15, 1), size=12, pxMode=True); gl.addItem(it); self.hl_items.append(it)
        else:
            gps = self.p.gps()
            if gps is not None and k < len(gps['valid']) and gps['valid'][k]:
                it = g.GLScatterPlotItem(pos=np.array([gps['enu'][k] - o], np.float32), color=(.86, .15, .15, 1), size=14, pxMode=True); gl.addItem(it); self.hl_items.append(it)
        if names and k < len(names):                    # picture-in-picture of the image
            r = QtGui.QImageReader(os.path.join(self.p.images_dir, names[k])); sz = r.size()
            if sz.isValid(): r.setScaledSize(sz.scaled(320, 200, Qt.KeepAspectRatio))
            img = r.read()
            if not img.isNull(): self.pip.setPixmap(QtGui.QPixmap.fromImage(img)); self.pip.adjustSize(); self.pip.show(); self._place_pip()

    def update_hud(self):
        if self.gl is None: return
        e = self.gl.eye() + getattr(self, 'o', 0)
        names = self.p.names() if self.p else []; k = self.frame.value() - 1
        self.hud.setText(('first person' if self.fly.isChecked() else 'orbit') + f'   eye {np.round(e, 2).tolist()}   speed {self.gl.speed:.2f}/s'
                         + (f'   frame {frame_label(names[k], k)}: {names[k]}' if names and k < len(names) else ''))

    def resize_pts(self):
        if getattr(self, 'sc', None): self.sc.setData(size=self.size.value())


# ======================================================================================== map / reference
PASS_COLORS = ['#2563eb', '#ea580c', '#16a34a', '#9333ea', '#db2777', '#0891b2', '#ca8a04', '#4b5563']


class MapView(QtWidgets.QWidget):
    """Plan view of the GPS trajectory with frame numbers, estimated cameras and their offsets from GPS, a camera table and a preview."""
    frame_selected = QtCore.Signal(str)

    def __init__(self):
        super().__init__(); self.p = None; self.names = []; self.g = None; self.labels = []; self.sel = None
        import pyqtgraph as pg; self.pg = pg
        pg.setConfigOptions(antialias=True)
        v = QtWidgets.QVBoxLayout(self); row = FlowLayout(); v.addLayout(row)
        self.show_gps = QtWidgets.QCheckBox('GPS'); self.show_gps.setChecked(True)
        self.show_cams = QtWidgets.QCheckBox('estimated cameras'); self.show_cams.setChecked(True)
        self.model = QtWidgets.QComboBox(); self.model.setMinimumWidth(120)
        self.show_err = QtWidgets.QCheckBox('offset lines'); self.show_err.setChecked(True)
        self.err_scale = QtWidgets.QDoubleSpinBox(); self.err_scale.setRange(1, 1000); self.err_scale.setValue(10); self.err_scale.setPrefix('× '); self.err_scale.setDecimals(0)
        self.err_scale.setToolTip('Offset lines (GPS -> estimated camera) are drawn this many times longer so small offsets are visible')
        self.show_ids = QtWidgets.QCheckBox('frame numbers'); self.show_ids.setChecked(True)
        self.colour = QtWidgets.QComboBox(); self.colour.addItems(['by pass', 'by frame', 'by height', 'by offset from GPS'])
        row.setSpacing(10) if hasattr(row, 'setSpacing') else None
        g_ = fgroup(row, 'Show'); g_.addWidget(self.show_gps); g_.addWidget(self.show_ids)
        g_ = fgroup(row, 'Estimated cameras'); g_.addWidget(self.show_cams); g_.addWidget(self.model)
        g_ = fgroup(row, 'Offset from GPS'); g_.addWidget(self.show_err); g_.addWidget(flabel('drawn')); g_.addWidget(self.err_scale); g_.addWidget(flabel('longer'))
        g_ = fgroup(row, 'Colour points'); g_.addWidget(self.colour)
        g_ = fgroup(row, 'View'); b = QtWidgets.QPushButton('Fit all'); b.setToolTip('zoom to show the whole track'); b.clicked.connect(lambda: self.plot.autoRange()); g_.addWidget(b)
        self.show_err.setText('lines'); self.show_cams.setText('show'); self.show_gps.setText('GPS track')
        for w_ in (self.show_gps, self.show_cams, self.show_err, self.show_ids): w_.toggled.connect(self.redraw)
        self.err_scale.valueChanged.connect(self.redraw); self.colour.currentIndexChanged.connect(self.redraw); self.model.activated.connect(lambda _: self.load_model_cams())
        sp = QtWidgets.QSplitter(Qt.Horizontal); v.addWidget(sp, 1)
        self.plot = pg.PlotWidget(background='#ffffff'); self.plot.setAspectLocked(True); self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setLabel('bottom', 'east (m)'); self.plot.setLabel('left', 'north (m)'); sp.addWidget(self.plot)
        self.plot.getViewBox().sigRangeChanged.connect(lambda *_: self.update_labels())
        right = QtWidgets.QWidget(); rv = QtWidgets.QVBoxLayout(right); rv.setContentsMargins(6, 0, 0, 0); sp.addWidget(right); sp.setSizes([900, 460])
        self.thumb = QtWidgets.QLabel('Click a point on the map or a row in the table.'); self.thumb.setAlignment(Qt.AlignCenter); self.thumb.setMinimumHeight(220)
        self.thumb.setStyleSheet('background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; color:#6b7280'); rv.addWidget(self.thumb)
        self.navbar = NavBar('frame'); rv.addWidget(self.navbar, 0, Qt.AlignCenter)
        self.navbar.prev.connect(lambda: self.select(max(0, (self.sel or 0) - 1))); self.navbar.next.connect(lambda: self.select(min(len(self.names) - 1, (self.sel if self.sel is not None else -1) + 1)))
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key_PageUp), self, activated=self.navbar.prev.emit); QtGui.QShortcut(QtGui.QKeySequence(Qt.Key_PageDown), self, activated=self.navbar.next.emit)
        self.detail = QtWidgets.QLabel(); self.detail.setWordWrap(True); self.detail.setTextInteractionFlags(Qt.TextSelectableByMouse); rv.addWidget(self.detail)
        self.table = QtWidgets.QTableWidget(0, 7); self.table.setHorizontalHeaderLabels(['frame', 'name', 'pass', 'latitude', 'longitude', 'altitude', 'offset (m)'])
        self.table.verticalHeader().setVisible(False); self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows); self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True); self.table.itemSelectionChanged.connect(self.table_pick); rv.addWidget(self.table, 1)
        self.cam_xy = {}

    # ------------------------------------------------------------------ data
    def set_project(self, p):
        self.p = p; self.names = p.names(); self.g = p.gps()
        cur = self.model.currentText(); self.model.clear(); self.model.addItems(models_of(p))
        if cur and self.model.findText(cur) >= 0: self.model.setCurrentText(cur)
        elif p.state.get('model'): self.model.setCurrentText(p.state['model'])
        self.load_model_cams(redraw=False); self.fill_table(); self.redraw(); self.plot.autoRange()

    def load_model_cams(self, redraw=True):
        self.cam_xy = {}
        m = self.model.currentText()
        if self.p and m:
            try:
                rec = load_model(self.p.model_dir(m))
                self.cam_xy = {im.name: im.projection_center() for im in rec.images.values() if im.has_pose}
            except Exception: self.cam_xy = {}
        if redraw: self.fill_table(); self.redraw()

    def offsets(self):
        if self.g is None or not self.cam_xy: return {}
        idx = {n: k for k, n in enumerate(self.names)}
        return {n: float(np.linalg.norm(c[:2] - self.g['enu'][idx[n]][:2])) for n, c in self.cam_xy.items() if n in idx and self.g['valid'][idx[n]]}

    def fill_table(self):
        self.table.setSortingEnabled(False); self.table.setRowCount(len(self.names)); off = self.offsets()
        g = self.g; pid = g['pass_id'] if g is not None and 'pass_id' in g else None
        for k, n in enumerate(self.names):
            lla = g['lla'][k] if g is not None and g['valid'][k] and g.get('lla') is not None and len(g['lla']) else None
            vals = [frame_label(n, k), n, int(pid[k]) + 1 if pid is not None else '', f'{lla[0]:.7f}' if lla is not None else '', f'{lla[1]:.7f}' if lla is not None else '',
                    f'{lla[2]:.1f}' if lla is not None else '', f'{off[n]:.2f}' if n in off else ('not placed' if self.cam_xy and n not in self.cam_xy else '')]
            for c, val in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(); it.setData(Qt.DisplayRole, val); self.table.setItem(k, c, it)
        self.table.setSortingEnabled(True); self.table.sortByColumn(0, Qt.AscendingOrder); self.table.resizeColumnsToContents()

    def xy(self):
        """Plan coordinates per frame (GPS when there is GPS, else the model's cameras) and which ones exist."""
        N = len(self.names)
        if self.g is not None and self.show_gps.isChecked():
            return self.g['enu'][:, :2], self.g['valid'].copy()
        XY = np.full((N, 2), np.nan); ok = np.zeros(N, bool)
        for k, n in enumerate(self.names):
            if n in self.cam_xy: XY[k] = self.cam_xy[n][:2]; ok[k] = True
        return XY, ok

    # ------------------------------------------------------------------ drawing
    def redraw(self):
        if not self.p: return
        pg = self.pg; pl = self.plot; pl.clear(); self.labels = []; N = len(self.names)
        XY, ok = self.xy(); self.XY, self.ok = XY, ok
        if not ok.any():
            t = pg.TextItem('No GPS and no model yet.\nRun step 1 with a GPS source, or step 5 to see the estimated cameras.', color='#6b7280', anchor=(0.5, 0.5)); pl.addItem(t); return
        idx = np.flatnonzero(ok); mode = self.colour.currentIndex(); off = self.offsets()
        pid = self.g['pass_id'] if self.g is not None and 'pass_id' in self.g else np.zeros(N, int)
        import matplotlib.cm as cm
        if mode == 0: cols = [PASS_COLORS[int(pid[k]) % len(PASS_COLORS)] for k in idx]
        elif mode == 1: cols = [QtGui.QColor.fromRgbF(*cm.viridis(k / max(1, N - 1))[:3]) for k in idx]
        elif mode == 2 and self.g is not None:
            z = self.g['enu'][idx, 2]; lo, hi = np.percentile(z, [2, 98]); cols = [QtGui.QColor.fromRgbF(*cm.turbo(np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1))[:3]) for v in z]
        else: cols = [QtGui.QColor.fromRgbF(*cm.turbo(min(1, off.get(self.names[k], 0) / 2))[:3]) if self.names[k] in off else QtGui.QColor('#9ca3af') for k in idx]
        pl.addItem(pg.PlotCurveItem(XY[idx, 0], XY[idx, 1], pen=pg.mkPen('#cbd5e1', width=1.5)))
        sc = pg.ScatterPlotItem(XY[idx, 0], XY[idx, 1], size=7, brush=[pg.mkBrush(c) for c in cols], pen=pg.mkPen('#ffffff', width=0.6), data=idx,
                                hoverable=True, hoverSize=12, tip=self._tip)
        sc.sigClicked.connect(lambda _it, pts, *_: self.select(int(pts[0].data())) if len(pts) else None); pl.addItem(sc); self.scatter = sc
        if self.show_cams.isChecked() and self.cam_xy and self.g is not None and self.show_gps.isChecked():
            C = np.array([self.cam_xy[self.names[k]][:2] for k in idx if self.names[k] in self.cam_xy]).reshape(-1, 2)
            G = np.array([XY[k] for k in idx if self.names[k] in self.cam_xy]).reshape(-1, 2)
            if self.show_err.isChecked() and len(C):
                E = G + (C - G) * self.err_scale.value(); seg = np.empty((2 * len(C), 2)); seg[0::2] = G; seg[1::2] = E
                pl.addItem(pg.PlotCurveItem(seg[:, 0], seg[:, 1], connect='pairs', pen=pg.mkPen('#dc2626', width=1)))
            pl.addItem(pg.ScatterPlotItem(C[:, 0], C[:, 1], size=4, brush=pg.mkBrush('#f59e0b'), pen=None))
        self.sel_item = pg.ScatterPlotItem(size=18, brush=pg.mkBrush(0, 0, 0, 0), pen=pg.mkPen('#111827', width=2.5)); pl.addItem(self.sel_item)
        self.update_labels()
        if self.sel is not None: self.mark(self.sel)

    def _tip(self, x, y, data):
        k = int(data); n = self.names[k]; s = f'frame {frame_label(n, k)}: {n}\neast {x:.2f} m, north {y:.2f} m'
        if self.g is not None and 'pass_id' in self.g: s += f'\npass {int(self.g["pass_id"][k]) + 1}'
        off = self.offsets()
        if n in off: s += f'\noffset from GPS {off[n]:.2f} m'
        return s

    def update_labels(self):
        """Frame numbers: as many as fit - more appear when you zoom in."""
        if not self.p or not hasattr(self, 'XY'): return
        pl = self.plot
        for t in self.labels: pl.removeItem(t)
        self.labels = []
        if not self.show_ids.isChecked(): return
        (x0, x1), (y0, y1) = pl.getViewBox().viewRange(); XY, ok = self.XY, self.ok
        vis = np.flatnonzero(ok & (XY[:, 0] >= x0) & (XY[:, 0] <= x1) & (XY[:, 1] >= y0) & (XY[:, 1] <= y1))
        if not len(vis): return
        step = int(np.ceil(len(vis) / 40)); nice = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000]
        step = next((s for s in nice if s >= step), step)
        for k in vis[vis % step == 0]:
            t = self.pg.TextItem(str(frame_label(self.names[k], k)), color='#334155', anchor=(-0.15, 1.1)); t.setPos(XY[k, 0], XY[k, 1]); f = t.textItem.font(); f.setPointSize(8); t.setFont(f)
            pl.addItem(t); self.labels.append(t)

    # ------------------------------------------------------------------ selection
    def mark(self, k):
        if hasattr(self, 'sel_item') and self.ok[k]: self.sel_item.setData([self.XY[k, 0]], [self.XY[k, 1]])

    def select(self, k, from_table=False):
        if not self.p or k is None or k >= len(self.names): return
        self.sel = k; n = self.names[k]; self.mark(k); self.navbar.set(k, len(self.names))
        r = QtGui.QImageReader(os.path.join(self.p.images_dir, n)); sz = r.size()
        if sz.isValid(): r.setScaledSize(sz.scaled(440, 260, Qt.KeepAspectRatio))
        im = r.read()
        if not im.isNull(): self.thumb.setPixmap(QtGui.QPixmap.fromImage(im))
        txt = [f'<b>frame {frame_label(n, k)}</b> &nbsp; {n}']
        if self.g is not None and self.g['valid'][k]:
            e = self.g['enu'][k]; txt.append(f'GPS: east {e[0]:.2f}, north {e[1]:.2f}, up {e[2]:.2f} m' + (f', pass {int(self.g["pass_id"][k]) + 1}' if 'pass_id' in self.g else ''))
        if n in self.cam_xy:
            c = self.cam_xy[n]; txt.append(f'estimated camera ({self.model.currentText()}): {np.round(c, 2).tolist()}')
            off = self.offsets()
            if n in off: txt.append(f'horizontal offset from GPS: <b>{off[n]:.2f} m</b>')
        elif self.cam_xy: txt.append('<span style="color:#dc2626">not placed in this model</span>')
        self.detail.setText('<br>'.join(txt))
        if not from_table:
            self.table.blockSignals(True)
            for r_ in range(self.table.rowCount()):
                it = self.table.item(r_, 1)
                if it and it.data(Qt.DisplayRole) == n: self.table.selectRow(r_); self.table.scrollToItem(it); break
            self.table.blockSignals(False)
        self.frame_selected.emit(n)

    def table_pick(self):
        r = self.table.currentRow()
        if r < 0: return
        it = self.table.item(r, 1)
        if it is not None and it.data(Qt.DisplayRole) in self.names: self.select(self.names.index(it.data(Qt.DisplayRole)), from_table=True)


# ======================================================================================== frames
class FrameInspector(QtWidgets.QWidget):
    def __init__(self):
        super().__init__(); self.p = None
        h = QtWidgets.QHBoxLayout(self); left = QtWidgets.QVBoxLayout(); h.addLayout(left)
        self.filter = QtWidgets.QLineEdit(); self.filter.setPlaceholderText('filter frames...'); self.filter.textChanged.connect(self.fill); left.addWidget(self.filter)
        self.list = QtWidgets.QListWidget(); self.list.setMaximumWidth(260); self.list.currentTextChanged.connect(self.show); left.addWidget(self.list)
        right = QtWidgets.QVBoxLayout(); h.addLayout(right, 1); row = FlowLayout(); right.addLayout(row)
        self.overlay = QtWidgets.QComboBox(); self.overlay.addItems(['nothing', 'keypoints (step 2)', 'dense keypoints (step 9)', '3D points of the model (coloured by error)', 'mask'])
        self.overlay.currentIndexChanged.connect(lambda _: self.show(self.list.currentItem().text() if self.list.currentItem() else ''))
        g_ = fgroup(row, 'Draw on the image'); g_.addWidget(self.overlay)
        self.model = QtWidgets.QComboBox(); self.model.setToolTip('model used by the "3D points" overlay'); g_.addWidget(flabel('model')); g_.addWidget(self.model)
        self.model.activated.connect(lambda _: self.show(self.list.currentItem().text() if self.list.currentItem() else ''))
        self.navbar = NavBar('frame'); g_ = fgroup(row, 'Frame'); g_.addWidget(self.navbar)
        self.view = ZoomView(nav=True); right.addWidget(self.view, 1)
        for sig, d in ((self.view.prev_requested, -1), (self.view.next_requested, 1), (self.navbar.prev, -1), (self.navbar.next, 1)): sig.connect(lambda d=d: step_list(self.list, d))
        self.info = QtWidgets.QLabel(); self.info.setWordWrap(True); self.info.setTextInteractionFlags(Qt.TextSelectableByMouse); right.addWidget(self.info)

    def set_project(self, p):
        self.p = p; self.names = p.names(); self.fill(); cur = self.model.currentText(); self.model.clear(); self.model.addItems(models_of(p))
        self.model.setCurrentText(cur or p.state.get('model') or '')

    def fill(self):
        if not self.p: return
        f = self.filter.text().strip(); self.list.clear(); self.list.addItems([n for n in self.names if f in n])

    @busy
    def show(self, name):
        if not self.p or not name: return
        img = QtGui.QImage(os.path.join(self.p.images_dir, name)); pa = QtGui.QPainter(img); pa.setRenderHint(QtGui.QPainter.Antialiasing)
        mode = self.overlay.currentIndex(); txt = [f'<b>{name}</b>  {img.width()}x{img.height()}']
        g = self.p.gps(); k = self.names.index(name)
        if g is not None and g['valid'][k]: txt.append(f'GPS east/north/up {np.round(g["enu"][k], 2).tolist()} m, pass {int(g["pass_id"][k]) + 1}')
        def dots(xy, cols, r=3):
            for (x, y), c in zip(xy, cols):
                pa.setPen(QtGui.QPen(c, 2)); pa.drawEllipse(QtCore.QPointF(x, y), r, r)
        if mode in (1, 2):
            f = self.p.path('features' if mode == 1 else 'features_dense', name + '.kp.npy')
            if os.path.exists(f): kp = np.load(f); dots(kp, [QtGui.QColor(0, 255, 100)] * len(kp)); txt.append(f'{len(kp)} keypoints')
            else: txt.append('not computed yet')
        if mode == 3 and self.model.currentText():
            rec = load_model(self.p.model_dir(self.model.currentText())); im = rec.find_image_with_name(name)
            if im is None: txt.append('this frame is NOT placed in the model')
            else:
                xy, e = [], []
                for q in im.points2D:
                    if not q.has_point3D(): continue
                    pr = im.project_point(rec.points3D[q.point3D_id].xyz)
                    if pr is None: continue
                    xy.append(q.xy); e.append(np.linalg.norm(pr - q.xy))
                import matplotlib.cm as cm
                cols = [QtGui.QColor.fromRgbF(*cm.turbo(min(v / 3, 1))[:3]) for v in e]; dots(np.array(xy).reshape(-1, 2), cols)
                c = im.projection_center(); txt.append(f'{len(e)} 3D points seen, error mean {np.mean(e) if e else 0:.2f} px, median {np.median(e) if e else 0:.2f} px (blue 0 -> red 3+ px)')
                txt.append(f'camera position {np.round(c, 2).tolist()}' + (f', distance to GPS {np.linalg.norm(c - g["enu"][k]):.2f} m' if g is not None and g['valid'][k] else ''))
        if mode == 4:
            from ..common import mask_for
            m = mask_for(self.p, name, (img.width(), img.height()))
            if m is None: txt.append('no mask')
            else:
                ov = np.zeros((m.shape[0], m.shape[1], 4), np.uint8); ov[~m] = (255, 40, 40, 110)
                q = QtGui.QImage(ov.data, ov.shape[1], ov.shape[0], ov.shape[1] * 4, QtGui.QImage.Format_RGBA8888); pa.drawImage(0, 0, q); txt.append(f'{100 * (~m).mean():.1f} % masked (red)')
        try:
            pi = np.load(self.p.path('pair_inliers.npz')); sel = (pi['pairs'] == k).any(1)
            txt.append(f'{sel.sum()} verified pairs, inliers median {int(np.median(pi["n"][sel])) if sel.any() else 0}')
        except Exception: pass
        pa.end(); self.view.set_image(img, keep_view=True); self.info.setText('<br>'.join(txt)); self.navbar.set(self.list.currentRow(), self.list.count())


# ======================================================================================== pairs
class PairInspector(QtWidgets.QWidget):
    def __init__(self):
        super().__init__(); self.p = None
        h = QtWidgets.QHBoxLayout(self); left = QtWidgets.QVBoxLayout(); h.addLayout(left)
        self.filter = QtWidgets.QLineEdit(); self.filter.setPlaceholderText('frame number or name...'); self.filter.textChanged.connect(self.fill); left.addWidget(self.filter)
        self.sort = QtWidgets.QComboBox(); self.sort.addItems(['weakest first', 'strongest first', 'farthest apart first']); self.sort.currentIndexChanged.connect(self.fill); left.addWidget(self.sort)
        self.table = QtWidgets.QTableWidget(0, 4); self.table.setHorizontalHeaderLabels(['frame A', 'frame B', 'apart', 'kept']); self.table.setMaximumWidth(380)
        self.table.setToolTip('frame numbers of each verified pair, how many frames apart they are, and how many matches passed the geometry check')
        hh_ = self.table.horizontalHeader(); [hh_.setSectionResizeMode(c_, QtWidgets.QHeaderView.Stretch) for c_ in range(4)]
        self.table.verticalHeader().setVisible(False); self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows); self.table.itemSelectionChanged.connect(self.show); left.addWidget(self.table)
        g = QtWidgets.QGroupBox('any two frames (live FlexGlue)'); gl = QtWidgets.QFormLayout(g); self.a = QtWidgets.QLineEdit(); self.b = QtWidgets.QLineEdit()
        gl.addRow('A', self.a); gl.addRow('B', self.b); bm = QtWidgets.QPushButton('Match now'); bm.clicked.connect(self.match_now); gl.addRow(bm)
        ba = QtWidgets.QPushButton('Add A-B to the pairs (pairs_add.txt)'); ba.clicked.connect(lambda: self.edit_list('pairs_add.txt')); gl.addRow(ba); left.addWidget(g)
        br = QtWidgets.QPushButton('Remove selected pair (pairs_remove.txt)'); br.clicked.connect(lambda: self.edit_list('pairs_remove.txt', selected=True)); left.addWidget(br)
        right = QtWidgets.QVBoxLayout(); h.addLayout(right, 1); row = FlowLayout(); right.addLayout(row)
        self.what = QtWidgets.QComboBox(); self.what.addItems(['kept matches (inliers)', 'rejected matches', 'both']); self.what.currentIndexChanged.connect(self.show)
        self.nl = QtWidgets.QSpinBox(); self.nl.setRange(10, 5000); self.nl.setValue(400); self.nl.setSingleStep(100); self.nl.setToolTip('largest number of match lines drawn'); self.nl.valueChanged.connect(self.show)
        g_ = fgroup(row, 'Draw'); g_.addWidget(self.what); g_.addWidget(flabel('at most')); g_.addWidget(self.nl); g_.addWidget(flabel('lines'))
        self.navbar = NavBar('pair'); g_ = fgroup(row, 'Pair'); g_.addWidget(self.navbar)
        self.view = ZoomView(nav=True); right.addWidget(self.view, 1); self.info = QtWidgets.QLabel(); right.addWidget(self.info)
        for sig, d in ((self.view.prev_requested, -1), (self.view.next_requested, 1), (self.navbar.prev, -1), (self.navbar.next, 1)): sig.connect(lambda d=d: step_list(self.table, d))
        self.rows = []

    def set_project(self, p):
        self.p = p; self.names = p.names(); self.fill()

    def fill(self):
        if not self.p: return
        if not os.path.exists(self.p.path('pair_inliers.npz')):
            self.rows = []; self.table.setRowCount(0); self.info.setText('No verified pairs yet - run step 4 (matching). "Match now" works after step 2.'); return
        pi = np.load(self.p.path('pair_inliers.npz')); rows = [(int(a), int(b), int(n)) for (a, b), n in zip(pi['pairs'], pi['n'])]
        f = self.filter.text().strip()
        if f: rows = [r for r in rows if f in self.names[r[0]] or f in self.names[r[1]]]
        s = self.sort.currentIndex(); rows.sort(key=lambda r: r[2] if s == 0 else -r[2] if s == 1 else -(r[1] - r[0]))
        self.rows = rows[:3000]; self.table.setRowCount(len(self.rows))
        for i, (a, b, n) in enumerate(self.rows):
            for c, t in enumerate([str(frame_label(self.names[a], a)), str(frame_label(self.names[b], b)), str(b - a), str(n)]):
                it_ = QtWidgets.QTableWidgetItem(t); it_.setToolTip(f'{self.names[a]}  ↔  {self.names[b]}'); self.table.setItem(i, c, it_)

    def _draw(self, na, nb, ka, kb, inl, rej, title):
        ia, ib = QtGui.QImage(os.path.join(self.p.images_dir, na)), QtGui.QImage(os.path.join(self.p.images_dir, nb))
        W, H = max(ia.width(), ib.width()), ia.height() + ib.height(); img = QtGui.QImage(W, H, QtGui.QImage.Format_RGB32); img.fill(0)
        pa = QtGui.QPainter(img); pa.drawImage(0, 0, ia); pa.drawImage(0, ia.height(), ib); pa.setRenderHint(QtGui.QPainter.Antialiasing)
        rng = np.random.default_rng(0); m = self.what.currentIndex()
        for arr, col, on in ((rej, QtGui.QColor(255, 60, 60, 200), m in (1, 2)), (inl, QtGui.QColor(60, 255, 100, 200), m in (0, 2))):
            if not on or arr is None or not len(arr): continue
            s = arr[rng.choice(len(arr), min(len(arr), self.nl.value()), replace=False)]; pa.setPen(QtGui.QPen(col, 2))
            for u, v in s: pa.drawLine(QtCore.QPointF(*ka[u]), QtCore.QPointF(kb[v][0], kb[v][1] + ia.height()))
        pa.end(); self.view.set_image(img); self.info.setText(title)

    @busy
    def show(self):
        r = self.table.currentRow()
        if not self.p or r < 0 or r >= len(self.rows): return
        import pycolmap
        a, b, n = self.rows[r]; na, nb = self.names[a], self.names[b]
        with pycolmap.Database.open(self.p.path('database.db')) as d:
            ida = d.read_image_with_name(na).image_id; idb = d.read_image_with_name(nb).image_id
            ka, kb = d.read_keypoints(ida)[:, :2] - 0.5, d.read_keypoints(idb)[:, :2] - 0.5
            raw = d.read_matches(ida, idb); inl = d.read_two_view_geometry(ida, idb).inlier_matches
        s = set(map(tuple, inl.tolist())); rej = np.array([x for x in raw.tolist() if tuple(x) not in s]).reshape(-1, 2)
        self.a.setText(na); self.b.setText(nb)
        self.navbar.set(r, len(self.rows))
        self._draw(na, nb, ka, kb, inl, rej, f'{na} (top) - {nb} (bottom), {b - a} frames apart: {len(raw)} FlexGlue matches, {len(inl)} kept, {len(rej)} rejected by geometry')

    @busy
    def match_now(self):
        if not self.p: return
        from ..common import FeatureStore, fg_match, magsac
        na, nb = self.a.text().strip(), self.b.text().strip()
        if na not in self.names or nb not in self.names: raise ValueError('type two frame names from the project')
        fs = FeatureStore(self.p.path('features'), cache=4); snap = self.p.params('match').get('snap_px', 1.0)
        m, cov = fg_match(fs, na, nb, snap_px=snap); ka, kb = fs.kp(na), fs.kp(nb); inl = magsac(ka, kb, m, self.p.params('match')['ransac_px'])
        s = set(map(tuple, inl.tolist())); rej = np.array([x for x in m.tolist() if tuple(x) not in s]).reshape(-1, 2)
        self._draw(na, nb, ka, kb, inl, rej, f'LIVE {na} - {nb}: covisibility {cov:.2f}, {len(m)} FlexGlue matches, {len(inl)} pass the geometry check (MAGSAC)')

    def edit_list(self, fname, selected=False):
        if not self.p: return
        if selected:
            r = self.table.currentRow()
            if r < 0: return
            a, b, _ = self.rows[r]; pair = (self.names[a], self.names[b])
        else:
            pair = (self.a.text().strip(), self.b.text().strip())
            if pair[0] not in self.names or pair[1] not in self.names: QtWidgets.QMessageBox.warning(self, 'Pairs', 'type two frame names'); return
        with open(self.p.path(fname), 'a') as f: f.write(f'{pair[0]} {pair[1]}\n')
        QtWidgets.QMessageBox.information(self, 'Pairs', f'{pair[0]} {pair[1]} written to {fname}. Re-run step 3 (pairs) and the steps after it.')


# ======================================================================================== markers
class MarkerTool(QtWidgets.QWidget):
    def __init__(self):
        super().__init__(); self.p = None; self.obs = {}
        h = QtWidgets.QHBoxLayout(self); left = QtWidgets.QVBoxLayout(); h.addLayout(left)
        left.addWidget(QtWidgets.QLabel('<b>markers</b> (from the CSV in step 6, or add)'))
        self.mlist = QtWidgets.QListWidget(); self.mlist.setMaximumWidth(260); self.mlist.currentTextChanged.connect(self.marker_changed); left.addWidget(self.mlist)
        b = QtWidgets.QPushButton('Add marker name...'); b.clicked.connect(self.add_marker); left.addWidget(b)
        left.addWidget(QtWidgets.QLabel('<b>images</b> (* = clicked, ~ = predicted inside)'))
        self.ilist = QtWidgets.QListWidget(); self.ilist.setMaximumWidth(260); self.ilist.currentRowChanged.connect(self.show); left.addWidget(self.ilist)
        self.only = QtWidgets.QCheckBox('only images where the marker is predicted'); self.only.toggled.connect(self.fill_images); left.addWidget(self.only)
        right = QtWidgets.QVBoxLayout(); h.addLayout(right, 1)
        _ml = QtWidgets.QLabel('Left click = place the selected marker in this image. Right click = remove it here. Wheel = zoom, drag = pan, F = fit. '
                                         'After 2 clicks the marker is predicted in other images (red circle).'); _ml.setWordWrap(True); right.addWidget(_ml)
        self.navbar = NavBar('image'); right.addWidget(self.navbar, 0, Qt.AlignRight)
        self.view = ZoomView(nav=True); self.view.clicked.connect(self.clicked); right.addWidget(self.view, 1); self.info = QtWidgets.QLabel(); right.addWidget(self.info)
        for sig, d in ((self.view.prev_requested, -1), (self.view.next_requested, 1), (self.navbar.prev, -1), (self.navbar.next, 1)): sig.connect(lambda d=d: step_list(self.ilist, d))
        self.pred = {}

    def set_project(self, p):
        self.p = p; self.names = p.names()
        f = p.path('markers_obs.json'); self.obs = json.load(open(f)) if os.path.exists(f) else {}
        names = set(self.obs)
        csvp = p.params('markers').get('markers_csv')
        if csvp and os.path.exists(os.path.expanduser(csvp)):
            for l in open(os.path.expanduser(csvp)):
                t = [x.strip() for x in l.split(',')]
                if t and t[0] and not t[0].startswith('#') and t[0].lower() not in ('name', 'label', 'marker'): names.add(t[0])
        cur = self.mlist.currentItem().text() if self.mlist.currentItem() else None
        self.mlist.clear(); self.mlist.addItems(sorted(names))
        if cur: [self.mlist.setCurrentRow(i) for i in range(self.mlist.count()) if self.mlist.item(i).text() == cur]
        self.fill_images()

    def save(self):
        json.dump(self.obs, open(self.p.path('markers_obs.json'), 'w'), indent=1)

    def add_marker(self):
        n, ok = QtWidgets.QInputDialog.getText(self, 'Marker', 'marker name (same as in the CSV):')
        if ok and n.strip(): self.obs.setdefault(n.strip(), {}); self.save(); self.mlist.addItem(n.strip()); self.mlist.setCurrentRow(self.mlist.count() - 1)

    def marker(self):
        return self.mlist.currentItem().text() if self.mlist.currentItem() else None

    def marker_changed(self, _):
        self.predict(); self.fill_images()

    def predict(self):
        self.pred = {}; m = self.marker()
        if not m or not self.p or not self.p.current_model: return
        clicks = self.obs.get(m, {})
        if len(clicks) < 2: return
        try:
            from .. import geom
            rec = load_model(self.p.current_model); id_of = {im.name: i for i, im in rec.images.items()}
            o = [(id_of[n], xy) for n, xy in clicks.items() if n in id_of]
            if len(o) < 2: return
            X, err = geom.triangulate_track(rec, o)
            if X is None: return
            for i, im in rec.images.items():
                q = im.project_point(X); cam = rec.cameras[im.camera_id]
                if q is not None and 0 <= q[0] < cam.width and 0 <= q[1] < cam.height: self.pred[im.name] = (float(q[0]), float(q[1]))
            self.pred_err = err
        except Exception:
            self.pred = {}

    def fill_images(self):
        if not self.p: return
        m = self.marker(); clicks = self.obs.get(m, {}) if m else {}
        cur = self.ilist.currentItem().text()[2:] if self.ilist.currentItem() else None
        self.ilist.blockSignals(True); self.ilist.clear()
        for n in self.names:
            if self.only.isChecked() and n not in self.pred and n not in clicks: continue
            self.ilist.addItem(('* ' if n in clicks else '~ ' if n in self.pred else '  ') + n)
        self.ilist.blockSignals(False)
        if cur:
            for i in range(self.ilist.count()):
                if self.ilist.item(i).text()[2:] == cur: self.ilist.setCurrentRow(i); break

    def name(self):
        return self.ilist.currentItem().text()[2:] if self.ilist.currentItem() else None

    def show(self, *_):
        n = self.name()
        if not n or not self.p: return
        img = QtGui.QImage(os.path.join(self.p.images_dir, n)); pa = QtGui.QPainter(img); pa.setRenderHint(QtGui.QPainter.Antialiasing)
        f = QtGui.QFont(); f.setPointSize(28); pa.setFont(f); m = self.marker(); txt = [f'<b>{n}</b>']
        for mk, clicks in self.obs.items():
            if n in clicks:
                x, y = clicks[n]; c = QtGui.QColor(60, 255, 100) if mk == m else QtGui.QColor(255, 255, 255)
                pa.setPen(QtGui.QPen(c, 3)); pa.drawLine(QtCore.QPointF(x - 25, y), QtCore.QPointF(x + 25, y)); pa.drawLine(QtCore.QPointF(x, y - 25), QtCore.QPointF(x, y + 25))
                pa.drawText(QtCore.QPointF(x + 30, y - 10), mk)
        if m and n in self.pred:
            x, y = self.pred[n]; pa.setPen(QtGui.QPen(QtGui.QColor(255, 50, 50), 3)); pa.drawEllipse(QtCore.QPointF(x, y), 40, 40)
            txt.append(f'predicted {m} at ({x:.0f}, {y:.0f}); clicks agree within {getattr(self, "pred_err", 0):.1f} px')
        self.navbar.set(self.ilist.currentRow(), self.ilist.count())
        pa.end(); self.view.set_image(img, keep_view=True); txt.append(f'marker {m}: {len(self.obs.get(m, {}))} clicks' if m else 'select a marker'); self.info.setText('  |  '.join(txt))

    def clicked(self, x, y, button):
        m, n = self.marker(), self.name()
        if not m or not n: return
        if button == int(Qt.LeftButton.value): self.obs.setdefault(m, {})[n] = [round(x, 2), round(y, 2)]
        elif button == int(Qt.RightButton.value): self.obs.get(m, {}).pop(n, None)
        self.save(); self.predict(); self.fill_images(); self.show()

    def activate(self):
        if self.p and not self.mlist.count(): self.set_project(self.p)
