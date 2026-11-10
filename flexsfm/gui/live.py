"""'Live' tab: everything a running step does, as it happens.

- the exact command, process id, start time, elapsed time and the current stage
- CPU / RAM of the process (and its children) and GPU use, refreshed every 2 s
- the raw, unfiltered output (including COLMAP / GLOMAP / Ceres logs in verbose mode), searchable, warnings and errors coloured
- every file the step creates or grows inside the project folder
- each figure as soon as it is saved
Works for steps started in this window and for steps started elsewhere (command line, another window): those are followed
through their log file and process id.
"""
import os, re, shutil, subprocess, time
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from .flow import FlowLayout

SKIP_DIRS = {'__pycache__'}


def _fmt_bytes(n):
    for u in ('B', 'KB', 'MB', 'GB', 'TB'):
        if abs(n) < 1024 or u == 'TB': return f'{n:.0f} {u}' if u == 'B' else f'{n:.1f} {u}'
        n /= 1024


class Meter(QtWidgets.QWidget):
    def __init__(self, label):
        super().__init__(); h = QtWidgets.QVBoxLayout(self); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(2)
        self.lab = QtWidgets.QLabel(label); self.lab.setObjectName('subtle'); h.addWidget(self.lab)
        self.bar = QtWidgets.QProgressBar(); self.bar.setRange(0, 1000); self.bar.setTextVisible(True); self.bar.setFormat(''); h.addWidget(self.bar)

    def set(self, frac, text):
        self.bar.setValue(int(max(0, min(1, frac)) * 1000)); self.bar.setFormat(text)


class LiveView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__(); self.p = None; self.pid = None; self.step = None; self.t0 = None; self.ext_log = None; self.ext_pos = 0
        self.snap = {}; self.changes = {}; self.n_lines = 0; self.stage = ''; self.walk_every = 3.0; self.last_walk = 0
        v = QtWidgets.QVBoxLayout(self)
        # ---- header card
        card = QtWidgets.QFrame(); card.setObjectName('card'); cv = QtWidgets.QGridLayout(card); v.addWidget(card)
        self.head = QtWidgets.QLabel('No step running. Start a step (or open a project where one is running) to follow it here.')
        self.head.setTextInteractionFlags(Qt.TextSelectableByMouse); self.head.setWordWrap(True); cv.addWidget(self.head, 0, 0, 1, 4)
        self.m_cpu, self.m_ram, self.m_gpu, self.m_vram = Meter('CPU (process + children)'), Meter('RAM'), Meter('GPU use'), Meter('GPU memory')
        for k, m in enumerate((self.m_cpu, self.m_ram, self.m_gpu, self.m_vram)): cv.addWidget(m, 1, k)
        # ---- body: visual viewer (top) over output + side panels (bottom)
        vsplit = QtWidgets.QSplitter(Qt.Vertical); v.addWidget(vsplit, 1)
        top = QtWidgets.QSplitter(Qt.Horizontal); vsplit.addWidget(top)
        self.vis = QtWidgets.QTabWidget(); top.addWidget(self.vis); self._build_visual()
        sp = QtWidgets.QSplitter(Qt.Horizontal); vsplit.addWidget(sp); vsplit.setSizes([620, 330])
        left = QtWidgets.QWidget(); lv = QtWidgets.QVBoxLayout(left); lv.setContentsMargins(0, 0, 0, 0); sp.addWidget(left)
        row = FlowLayout(); lv.addLayout(row)
        self.search = QtWidgets.QLineEdit(); self.search.setPlaceholderText('find in output...'); self.search.returnPressed.connect(self.find); self.search.setMinimumWidth(240); row.addWidget(self.search, 1)
        self.follow = QtWidgets.QCheckBox('follow'); self.follow.setChecked(True); row.addWidget(self.follow)
        self.only_issues = QtWidgets.QCheckBox('only warnings / errors'); self.only_issues.toggled.connect(self.refilter); row.addWidget(self.only_issues)
        cb = QtWidgets.QPushButton('Copy all'); cb.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.raw.toPlainText())); row.addWidget(cb)
        self.raw = QtWidgets.QPlainTextEdit(); self.raw.setObjectName('log'); self.raw.setReadOnly(True); self.raw.setMaximumBlockCount(200000)
        self.raw.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)); lv.addWidget(self.raw, 1); self.lines = []
        right = QtWidgets.QSplitter(Qt.Vertical); top.addWidget(right); top.setSizes([1100, 380])
        g1 = QtWidgets.QGroupBox('stages'); l1 = QtWidgets.QVBoxLayout(g1); self.stages = QtWidgets.QListWidget(); l1.addWidget(self.stages); right.addWidget(g1)
        g2 = QtWidgets.QGroupBox('files written in the project (live)'); l2 = QtWidgets.QVBoxLayout(g2); fsp = sp
        self.files = QtWidgets.QTableWidget(0, 3); self.files.setHorizontalHeaderLabels(['file', 'size', 'change']); self.files.verticalHeader().setVisible(False)
        hh = self.files.horizontalHeader(); hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents); hh.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents); self.files.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.files.itemDoubleClicked.connect(self.open_file); l2.addWidget(self.files); fsp.addWidget(g2); fsp.setSizes([900, 500])
        g3 = QtWidgets.QGroupBox('figures (as they are saved)'); l3 = QtWidgets.QVBoxLayout(g3)
        self.figs = QtWidgets.QListWidget(); self.figs.setViewMode(QtWidgets.QListView.IconMode); self.figs.setIconSize(QtCore.QSize(200, 120)); self.figs.setResizeMode(QtWidgets.QListView.Adjust)
        self.figs.itemDoubleClicked.connect(lambda it: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(it.data(Qt.UserRole)))); l3.addWidget(self.figs); right.addWidget(g3)
        right.setSizes([200, 400])
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(2000)
        try:
            import psutil; self.psutil = psutil
        except Exception: self.psutil = None
        self.has_smi = shutil.which('nvidia-smi') is not None

    # ------------------------------------------------------------------ visual viewer
    def _build_visual(self):
        from .viewers import ZoomView
        # what it sees
        w1 = QtWidgets.QWidget(); l1 = QtWidgets.QVBoxLayout(w1); l1.setContentsMargins(4, 4, 4, 4)
        r = QtWidgets.QHBoxLayout(); self.vcap = QtWidgets.QLabel('When a step works on images, the image or pair it is looking at appears here.'); self.vcap.setWordWrap(True)
        self.vcap.setStyleSheet('font-weight:600'); r.addWidget(self.vcap, 1)
        self.pause = QtWidgets.QCheckBox('pause'); self.pause.setToolTip('Stop updating the view (the step keeps running)'); r.addWidget(self.pause); l1.addLayout(r)
        self.vview = ZoomView(); l1.addWidget(self.vview, 1)
        self.strip = QtWidgets.QListWidget(); self.strip.setViewMode(QtWidgets.QListView.IconMode); self.strip.setFlow(QtWidgets.QListView.LeftToRight)
        self.strip.setWrapping(False); self.strip.setIconSize(QtCore.QSize(120, 68)); self.strip.setFixedHeight(96); self.strip.setToolTip('recent views - click to look again')
        self.strip.itemClicked.connect(self._strip_pick); l1.addWidget(self.strip); self.snaps = []
        self.vis.addTab(w1, 'What it sees')
        # cameras in 3D
        w2 = QtWidgets.QWidget(); l2 = QtWidgets.QVBoxLayout(w2); l2.setContentsMargins(4, 4, 4, 4)
        r2 = QtWidgets.QHBoxLayout(); self.scap = QtWidgets.QLabel('When a step solves or refines camera poses, the cameras (frustums) and points appear here and move as they are optimised.')
        self.scap.setWordWrap(True); self.scap.setStyleSheet('font-weight:600'); r2.addWidget(self.scap, 1)
        r2.addWidget(QtWidgets.QLabel('frustum size')); self.fs = QtWidgets.QDoubleSpinBox(); self.fs.setRange(0.01, 1000); self.fs.setSingleStep(0.10); self.fs.setValue(1.0); self.fs.setSuffix(' m')
        self.fs.valueChanged.connect(lambda _: self._draw_scene(refit=False)); r2.addWidget(self.fs)
        self.trails = QtWidgets.QCheckBox('motion lines'); self.trails.setChecked(True); self.trails.setToolTip('Lines from each camera\'s position when the stage started to its position now (exaggerated like the cameras)')
        self.trails.toggled.connect(lambda _: self._draw_scene(refit=False)); r2.addWidget(self.trails)
        self.gpsbox = QtWidgets.QCheckBox('GPS'); self.gpsbox.setChecked(True); self.gpsbox.setToolTip('GPS positions (blue dots) and each camera\'s offset from its GPS position (orange lines), once the model is aligned to GPS')
        self.gpsbox.toggled.connect(lambda _: self._draw_scene(refit=False)); r2.addWidget(self.gpsbox)
        self.gpsx = QtWidgets.QSpinBox(); self.gpsx.setRange(1, 200); self.gpsx.setValue(1); self.gpsx.setPrefix('offset × '); self.gpsx.setToolTip('Draw the camera-to-GPS offsets this many times longer (GPS offsets are often much smaller than the camera spacing)')
        self.gpsx.valueChanged.connect(lambda _: self._draw_scene(refit=False)); r2.addWidget(self.gpsx)
        self.motx = QtWidgets.QSpinBox(); self.motx.setRange(0, 5000); self.motx.setValue(0); self.motx.setPrefix('motion × '); self.motx.setSpecialValueText('motion × auto')
        self.motx.setToolTip('Optimisation often moves cameras by centimetres in a scene of hundreds of metres. Cameras are drawn at their pose when the stage started '
                             'plus the motion so far, this many times larger (position and rotation). auto = picked so the motion is visible; 1 = true size.')
        self.motx.valueChanged.connect(lambda _: self._draw_scene(refit=False)); r2.addWidget(self.motx); l2.addLayout(r2)
        self.legend = QtWidgets.QLabel(''); self.legend.setObjectName('subtle'); self.legend.setWordWrap(True); l2.addWidget(self.legend)
        self.gl_holder = QtWidgets.QVBoxLayout(); l2.addLayout(self.gl_holder, 1); self.gl = None; self.scene = None; self.prev_C = {}
        self.vis.addTab(w2, 'Cameras in 3D')
        # convergence
        import pyqtgraph as pg
        w3 = QtWidgets.QWidget(); l3 = QtWidgets.QVBoxLayout(w3)
        self.conv = pg.PlotWidget(background='#ffffff'); self.conv.showGrid(x=True, y=True, alpha=0.2); self.conv.setLabel('left', 'mean reprojection error (px)')
        self.conv.setLabel('bottom', 'update'); l3.addWidget(self.conv, 1)
        self.conv2 = pg.PlotWidget(background='#ffffff'); self.conv2.showGrid(x=True, y=True, alpha=0.2); self.conv2.setLabel('left', 'cameras placed'); self.conv2.setLabel('bottom', 'update')
        self.conv2.setMaximumHeight(200); l3.addWidget(self.conv2)
        self.ccap = QtWidgets.QLabel('Each point is one update of the reconstruction (a mapper stage, a camera added, or a burst of bundle-adjustment iterations). '
                                     'A falling error means the optimisation is converging.'); self.ccap.setObjectName('subtle'); self.ccap.setWordWrap(True); l3.addWidget(self.ccap)
        self.vis.addTab(w3, 'Convergence')
        self._seen = set()

    def _show_image(self, path, caption):
        if self.pause.isChecked(): return
        img = QtGui.QImage(path)
        if img.isNull(): return
        self.vview.set_image(img); self.vcap.setText(caption)
        self.snaps.append((img, caption)); self.snaps = self.snaps[-40:]
        it = QtWidgets.QListWidgetItem(QtGui.QIcon(QtGui.QPixmap.fromImage(img.scaled(240, 136, Qt.KeepAspectRatio, Qt.SmoothTransformation))), '')
        it.setToolTip(caption); it.setData(Qt.UserRole, len(self.snaps) - 1); self.strip.addItem(it)
        while self.strip.count() > 40: self.strip.takeItem(0)
        for i in range(self.strip.count()): self.strip.item(i).setData(Qt.UserRole, len(self.snaps) - self.strip.count() + i)
        self.strip.scrollToBottom()
        if 'img' not in self._seen: self._seen.add('img'); self.vis.setCurrentIndex(0)

    def _strip_pick(self, it):
        k = it.data(Qt.UserRole)
        if 0 <= k < len(self.snaps): self.pause.setChecked(True); img, cap = self.snaps[k]; self.vview.set_image(img); self.vcap.setText('(paused) ' + cap)

    def _load_scene(self, path, caption):
        import numpy as np
        try:
            z = np.load(path, allow_pickle=False); sc = {k: z[k] for k in z.files}
        except Exception: return
        old = self.scene; self.scene = sc
        names = [str(n) for n in sc['names']]; C = sc['C']
        self.prev = {n: self.prev_C.get(n) for n in names}
        ext_new = float(np.ptp(C, 0).max()) if len(C) > 1 else 1.0
        refit = old is None or abs(np.log(max(ext_new, 1e-6) / max(getattr(self, '_ext', 1e-6), 1e-6))) > 1.0
        self._ext = ext_new
        if old is not None and refit and len(self.prev_C):   # the model changed frame (e.g. aligned to GPS, now in metres): old positions are not comparable
            self.prev = {n: c for n, c in zip(names, C)}
        stg = str(sc['stage'])
        if refit or stg != getattr(self, '_base_stage', None):    # a new stage (or a new frame): motion is measured from here
            self.base = {n: (C[k].astype(np.float64), sc['R'][k].astype(np.float64)) for k, n in enumerate(names)}; self._base_stage = stg; self._auto_x = None
        if self.gl is not None: self.gl.scene_ext = float(max(ext_new, np.ptp(sc['X'], 0).max() if len(sc['X']) else 0, 1.0))
        if refit or self.fs.value() == 1.0: self.fs.blockSignals(True); self.fs.setValue(round(max(0.02, ext_new / 90), 2)); self.fs.blockSignals(False)
        if not caption or caption == 'update': caption = f"{sc['stage']}: {sc['caption']}" if str(sc['caption']) else str(sc['stage'])
        self.scap.setText(caption)
        self._draw_scene(refit)
        self.prev_C = {n: c for n, c in zip(names, C)}
        self._draw_convergence(sc)
        if 'scene' not in self._seen: self._seen.add('scene'); self.vis.setCurrentIndex(1)

    STAGE_SHORT = [('global mapping', 'mapping'), ('global positioning', 'positioning'), ('rotations fixed', 'BA, rotations fixed'),
                   ('joint, refined', 'BA refined'), ('(joint)', 'BA joint'), ('retriangulation', 'retriangulation'), ('mapping finished', 'mapping done'),
                   ('aligned to GPS', 'aligned to GPS'), ('GPS bundle adjustment', 'GPS bundle adjustment'), ('refinement', 'refinement'),
                   ('initial pair', 'initial pair'), ('adding cameras', 'adding cameras'), ('bundle adjustment', 'bundle adjustment')]

    def _short(self, st):
        for k, v in self.STAGE_SHORT:
            if k in st: return v
        return st[:24]

    def _draw_convergence(self, sc):
        """Error per update: stages as shaded bands with their name, a thick line with a soft fill, the latest value called out."""
        import numpy as np, pyqtgraph as pg
        e = np.asarray(sc['hist_err'], float); nc = np.asarray(sc['hist_ncam'], float); x = np.arange(1, len(e) + 1)
        stg = [self._short(str(t)) for t in sc['hist_stage']]
        f_tick = QtGui.QFont(); f_tick.setPointSize(10); f_lab = QtGui.QFont(); f_lab.setPointSize(9); f_lab.setBold(True)
        for pw in (self.conv, self.conv2):
            pw.clear(); pw.showGrid(x=False, y=True, alpha=0.12); pw.getPlotItem().hideButtons()
            for ax in ('left', 'bottom'): pw.getAxis(ax).setStyle(tickFont=f_tick, tickTextOffset=6); pw.getAxis(ax).setPen('#cbd5e1'); pw.getAxis(ax).setTextPen('#475569')
        if not len(e): return
        ok = np.isfinite(e); hi = float(np.nanmax(e)) * 1.18 if ok.any() else 1.0
        runs, i0 = [], 0                                           # contiguous runs of the same stage
        for k in range(1, len(stg) + 1):
            if k == len(stg) or stg[k] != stg[i0]: runs.append((i0, k - 1, stg[i0])); i0 = k
        band = ['#f1f5f9', '#e0e7ff']
        for j, (a_, b_, name) in enumerate(runs):
            for pw, top in ((self.conv, hi), (self.conv2, None)):
                r = pg.LinearRegionItem(values=(a_ + 0.5, b_ + 1.5), movable=False, brush=pg.mkBrush(band[j % 2]), pen=pg.mkPen(None)); r.setZValue(-20); pw.addItem(r)
            w_ = b_ - a_ + 1
            t = pg.TextItem(f'{j + 1}  {name}' if w_ >= 3 else str(j + 1), color='#334155', anchor=(0, 0) if w_ >= 3 else (0.5, 0))   # number (+ name if it fits) on top
            t.setFont(f_lab); t.setPos(a_ + 0.6 if w_ >= 3 else a_ + 1, hi * 0.985); self.conv.addItem(t)
        self.ccap.setText('<b>Stages:</b> ' + ' &nbsp;·&nbsp; '.join(f'<b>{j + 1}</b> {n}' for j, (_, _, n) in enumerate(runs))
                          + '<br><span style="color:#6b7280">Each point is one update (a mapper stage, a camera added, or a burst of bundle-adjustment iterations). '
                          'A falling curve means the optimisation is converging.</span>')
        lo = 0.0
        self.conv.plot(x, e, pen=pg.mkPen('#2563eb', width=3), fillLevel=lo, brush=pg.mkBrush(37, 99, 235, 40),
                       symbol='o', symbolSize=8, symbolBrush='#ffffff', symbolPen=pg.mkPen('#2563eb', width=2))
        last = pg.TextItem(f'{e[-1]:.2f} px', color='#1d4ed8', anchor=(1.05, 1.1)); fl = QtGui.QFont(); fl.setPointSize(14); fl.setBold(True); last.setFont(fl)
        last.setPos(x[-1], e[-1]); self.conv.addItem(last)
        self.conv.setYRange(lo, hi, padding=0); self.conv.setXRange(0.5, len(e) + 0.5, padding=0.02)
        self.conv2.plot(x, nc, pen=pg.mkPen('#16a34a', width=2.5), stepMode=None, symbol='o', symbolSize=6, symbolBrush='#16a34a', symbolPen=None)
        self.conv2.setXRange(0.5, len(e) + 0.5, padding=0.02)

    def _draw_scene(self, refit=False):
        import numpy as np
        sc = self.scene
        if sc is None: return
        from .viewers import Model3D, _make_flyview
        if self.gl is None:
            if not Model3D.gl_available():
                self.legend.setText('3D needs OpenGL (not available in this session) - the convergence tab still works.'); return
            import pyqtgraph.opengl as glm
            self.glm = glm; self.gl = _make_flyview(glm)(); self.gl.setBackgroundColor('#ffffff'); self.gl_holder.addWidget(self.gl); refit = True
        gl, g = self.gl, self.glm
        for it in list(gl.items):
            if it in gl.items: gl.removeItem(it)
        C = sc['C'].astype(np.float64); X = sc['X'].astype(np.float64); names = [str(n) for n in sc['names']]
        o = getattr(self, '_o', None)
        if refit or o is None: o = np.median(C, 0) if len(C) else np.zeros(3); self._o = o
        if len(X):
            col = np.c_[sc['col'].astype(np.float32) / 255, np.ones(len(X), np.float32)]
            it = g.GLScatterPlotItem(pos=(X - o).astype(np.float32), color=col, size=2, pxMode=True); it.setGLOptions('opaque'); gl.addItem(it)
        # motion since the stage started, optionally exaggerated (cameras typically move cm in a scene of 100s of m)
        from scipy.spatial.transform import Rotation as Rot
        base = getattr(self, 'base', {}); has = np.array([n in base for n in names])
        B = np.array([base[n][0] if n in base else C[k] for k, n in enumerate(names)]); mv = np.linalg.norm(C - B, axis=1)
        x_m = self.motx.value()
        if x_m == 0:                                               # auto: median move becomes ~1.5 % of the scene; fixed for the stage
            if self._auto_x is None and has.any() and np.median(mv[has]) > 0:
                raw = 0.015 * max(self._ext, 1e-6) / np.median(mv[has]); e10 = 10 ** np.floor(np.log10(max(raw, 1)))
                self._auto_x = int(min(2000, max(1, max(m_ * e10 for m_ in (1, 2, 5) if m_ * e10 <= max(raw, 1)))))
            x_m = self._auto_x or 1
        Cd = B + (C - B) * x_m; Rd = []
        for k, n in enumerate(names):
            Rm = sc['R'][k].astype(np.float64)
            if x_m > 1 and n in base:
                Rw0 = base[n][1].T; dR = Rot.from_matrix(Rm.T @ Rw0.T).as_rotvec() * x_m
                ang = np.linalg.norm(dR); dR = dR * (min(ang, np.radians(40)) / ang) if ang > 0 else dR
                Rm = (Rot.from_rotvec(dR).as_matrix() @ Rw0).T
            Rd.append(Rm)
        scale = np.percentile(mv[has], 95) if has.any() and mv[has].max() > 0 else 1.0
        import matplotlib.cm as cm
        s = self.fs.value(); seg, cols = [], []
        for k, n in enumerate(names):
            Rm = Rd[k]; w, h, f = sc['K'][k]; c = Cd[k] - o
            hw, hh = w / 2 / f * s, h / 2 / f * s
            cr = [Rm.T @ np.array([x, y, s]) + c for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
            colr = (0.25, 0.25, 0.25, 1.0) if not has[k] else tuple(cm.turbo(min(1.0, mv[k] / max(scale, 1e-12)) * 0.9 + 0.05)[:3]) + (1.0,)
            for q in cr: seg += [c, q]; cols += [colr, colr]
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)): seg += [cr[a], cr[b]]; cols += [colr, colr]
        if seg:
            it = g.GLLinePlotItem(pos=np.array(seg, np.float32), color=np.array(cols, np.float32), width=1.6, mode='lines', antialias=True); gl.addItem(it)
        if self.trails.isChecked() and (mv[has] > 0).any():
            tr = np.empty((2 * has.sum(), 3)); tr[0::2] = B[has] - o; tr[1::2] = Cd[has] - o
            gl.addItem(g.GLLinePlotItem(pos=tr.astype(np.float32), color=(0.85, 0.1, 0.65, 1), width=2.5, mode='lines', antialias=True))
        new = ~has
        G = sc['G'].astype(np.float64) if 'G' in sc and len(sc['G']) == len(C) else None; gtxt = ''
        if G is not None and self.gpsbox.isChecked():
            okg = np.isfinite(G).all(1)
            if okg.any():
                x_ = self.gpsx.value(); Gd = Cd + (G - C) * x_           # attached to the drawn camera; optionally exaggerated
                it = g.GLScatterPlotItem(pos=(Gd[okg] - o).astype(np.float32), color=(0.15, 0.39, 0.92, 1), size=7, pxMode=True); it.setGLOptions('translucent'); gl.addItem(it)
                ln = np.empty((2 * okg.sum(), 3)); ln[0::2] = Cd[okg] - o; ln[1::2] = Gd[okg] - o
                gl.addItem(g.GLLinePlotItem(pos=ln.astype(np.float32), color=(0.98, 0.45, 0.09, 1), width=2, mode='lines', antialias=True))
                d = np.linalg.norm(C[okg] - G[okg], axis=1); gtxt = f' &nbsp;·&nbsp; offset median {np.median(d):.2f} m, max {d.max():.2f} m' + (f' (drawn {x_}× longer)' if x_ > 1 else '')
        nn = int(new.sum()); moved = mv[has]; u = lambda m: f'{m * 100:.1f} cm' if m < 1 else f'{m:.2f} m'
        self.motion_txt = (f'camera motion since this stage started: median {u(np.median(moved))}, max {u(np.max(moved))}'
                           + (f' (drawn ×{x_m})' if x_m > 1 else '')) if len(moved) and moved.max() > 0 else ''
        sw = lambda c: f'<span style="color:{c}">&#9632;</span>'
        rows = [f'<b>{len(names)} cameras, {len(X):,} points</b>' + (f' &nbsp;·&nbsp; {self.motion_txt}' if self.motion_txt else ''),
                f'{sw("#3b4cc0")} still &nbsp; {sw("#b40426")} moved most ({u(scale)}) &nbsp; frustum colour = motion since the stage started'
                + (f' &nbsp; {sw("#404040")} newly placed ({nn})' if nn and nn < len(names) else ''),
                f'{sw("#d91aa6")} pink lines = motion since the stage started' + (f' &nbsp;·&nbsp; cameras drawn at start pose + {x_m}× the motion' if x_m > 1 else '')]
        if gtxt: rows.append(f'{sw("#2563eb")} GPS &nbsp; {sw("#f97316")} camera-to-GPS offset' + gtxt)
        rows.append('<span style="color:#6b7280">drag = rotate &nbsp;·&nbsp; wheel = zoom &nbsp;·&nbsp; F = fly (W/A/S/D)</span>')
        self.legend.setText('<br>'.join(rows))
        if refit:                                                  # frame everything: cameras and the bulk of the points
            allp = np.vstack([C - o] + ([X - o] if len(X) else []))
            lo_, hi_ = np.percentile(allp, 1, 0), np.percentile(allp, 99, 0); ctr = (lo_ + hi_) / 2; ext = float(np.max(hi_ - lo_))
            gl.opts['center'] = QtGui.QVector3D(*[float(v) for v in ctr]); gl.setCameraPosition(distance=max(ext, 1.0) * 1.35, elevation=89.9, azimuth=-90)   # top view

    # ------------------------------------------------------------------ lifecycle
    def set_project(self, p):
        if self.p is None or p.root != self.p.root: self.p = p

    def start(self, p, step, cmd, verbose):
        self.p = p; self.step = step; self.pid = None; self.t0 = time.time(); self.ext_log = None; self.cmd = cmd; self.code = None
        self.raw.clear(); self.lines = []; self.stages.clear(); self.figs.clear(); self.files.setRowCount(0); self.changes = {}; self.stage = ''
        self.verbose = verbose; self.snap = self._walk(); self.last_walk = time.time(); self._header()
        self.snaps = []; self.strip.clear(); self.scene = None; self.prev_C = {}; self._seen = set(); self.vcap.setText('waiting for the step to show something ...')
        self.live_mt = {}; self.conv.clear(); self.conv2.clear()

    def started(self, pid):
        self.pid = pid; self._header()

    def attach(self, p, step, pid):
        """Follow a step started elsewhere: tail its log file and watch its process."""
        if self.step == step and self.pid == pid and self.ext_log: return
        self.start(p, step, f'(started outside this window) python -m flexsfm run {p.root} {step}', None)
        self.pid = pid; self.ext_log = p.path('logs', f'{step}.log'); self.ext_pos = 0; self._header()

    def finish(self, code):
        self.code = code; self._resources(); self._header(); self.pid = None
        if self.p: now = self._walk(); self._diff(now); self.snap = now          # final list of what the step wrote

    # ------------------------------------------------------------------ output
    def line(self, s):
        """Every raw output line of the step (called by the main window, or by the log tail)."""
        self.n_lines += 1
        if s.startswith('@@P '):
            parts = s.split(' ', 2); msg = parts[2] if len(parts) > 2 else ''
            if msg and msg != self.stage:
                self.stage = msg; el = time.time() - (self.t0 or time.time())
                self.stages.addItem(f'{el:7.1f} s   {msg}'); self.stages.scrollToBottom(); self._header()
            return                                  # progress ticks are summarised in 'stages', not repeated in the output
        if s.startswith('@@V ') or s.startswith('@@S '):
            path, _, cap = s[4:].partition('\t')
            (self._show_image if s.startswith('@@V') else self._load_scene)(path.strip(), cap.strip())
            self.live_mt[s[:3]] = os.path.getmtime(path.strip()) if os.path.exists(path.strip()) else 0
            return
        if s.startswith('@@F '):
            path = s[4:].strip()
            if os.path.exists(path):
                pm = QtGui.QPixmap(path).scaled(200, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                it = QtWidgets.QListWidgetItem(QtGui.QIcon(pm), os.path.basename(path)); it.setData(Qt.UserRole, path); self.figs.addItem(it); self.figs.scrollToBottom()
        self.lines.append(s)
        if self.only_issues.isChecked() and not self._is_issue(s): return
        self._append(s)

    @staticmethod
    def _is_issue(s):
        l = s.lower(); return any(k in l for k in ('error', 'fail', 'traceback', 'warn', 'exception')) or s.startswith(('E2', 'W2', 'F2'))

    def _append(self, s):
        l = s.lower()
        col = '#fca5a5' if ('error' in l or 'traceback' in l or 'fail' in l or s.startswith(('E2', 'F2'))) else \
              '#fcd34d' if ('warn' in l or s.startswith('W2')) else '#86efac' if s.startswith('@@C PASS') else \
              '#93c5fd' if s.startswith(('I2', '@@')) else None
        sb = self.raw.verticalScrollBar(); at_end = sb.value() >= sb.maximum() - 4
        cur = QtGui.QTextCursor(self.raw.document()); cur.movePosition(QtGui.QTextCursor.End)
        fmt = QtGui.QTextCharFormat()
        if col: fmt.setForeground(QtGui.QColor(col))
        if self.raw.document().characterCount() > 1: cur.insertText('\n', QtGui.QTextCharFormat())
        from .style import tidy
        cur.insertText(tidy(s), fmt)
        if self.follow.isChecked() and at_end: sb.setValue(sb.maximum())

    def refilter(self):
        self.raw.clear()
        for s in self.lines:
            if not self.only_issues.isChecked() or self._is_issue(s): self._append(s)

    def find(self):
        t = self.search.text()
        if t and not self.raw.find(t):
            self.raw.moveCursor(QtGui.QTextCursor.Start); self.raw.find(t)

    # ------------------------------------------------------------------ periodic
    def tick(self):
        if self.ext_log and os.path.exists(self.ext_log):           # follow a step started elsewhere
            try:
                with open(self.ext_log, errors='replace') as f:
                    f.seek(self.ext_pos); new = f.read(); self.ext_pos = f.tell()
                for s in new.splitlines(): self.line(s)
            except Exception: pass
            if self.pid and not self._alive(self.pid): self.code = 'ended'; self.pid = None
            self._poll_live_files()
        if self.step: self._resources(); self._header()
        if self.step and self.p and time.time() - self.last_walk > self.walk_every and (self.pid or self.code is None):
            t = time.time(); now = self._walk(); self.walk_every = max(3.0, (time.time() - t) * 10); self.last_walk = time.time()
            self._diff(now); self.snap = now

    def _poll_live_files(self):
        """For steps started elsewhere: pick up the live image / 3D snapshot files the step writes."""
        import json
        if not (self.p and self.step): return
        d = self.p.report_dir(self.step); mt = getattr(self, 'live_mt', {})
        js = os.path.join(d, 'live_state.json'); sc = os.path.join(d, 'live_scene.npz')
        try:
            if os.path.exists(js) and os.path.getmtime(js) > mt.get('js', 0):
                mt['js'] = os.path.getmtime(js); info = json.load(open(js)); self._show_image(info['path'], info.get('caption', ''))
            if os.path.exists(sc) and os.path.getmtime(sc) > mt.get('sc', 0):
                mt['sc'] = os.path.getmtime(sc); self._load_scene(sc, 'update')
        except Exception: pass
        self.live_mt = mt

    def _alive(self, pid):
        try: os.kill(pid, 0); return True
        except ProcessLookupError: return False
        except Exception: return True

    def _resources(self):
        if not self.pid or not self.psutil:
            for m in (self.m_cpu, self.m_ram): m.set(0, '-')
        else:
            try:
                cache = getattr(self, '_procs', {})                   # cpu_percent needs the same Process object between calls
                pr = cache.get(self.pid) or self.psutil.Process(self.pid); cache[self.pid] = pr
                procs = [pr] + [cache.setdefault(c.pid, c) for c in pr.children(recursive=True)]; self._procs = cache
                cpu = sum(q.cpu_percent(interval=None) for q in procs); rss = sum(q.memory_info().rss for q in procs)
                n = self.psutil.cpu_count() or 1; tot = self.psutil.virtual_memory().total
                thr = sum(q.num_threads() for q in procs)
                self.m_cpu.set(cpu / (100 * n), f'{cpu:.0f} %  ({cpu / 100:.1f} of {n} cores, {thr} threads)')
                self.m_ram.set(rss / tot, f'{_fmt_bytes(rss)} of {_fmt_bytes(tot)}')
            except Exception:
                self.m_cpu.set(0, 'process ended'); self.m_ram.set(0, '-')
        if self.has_smi:
            try:
                q = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, timeout=2).stdout.strip().split('\n')[0].split(',')
                util, used, tot = (float(x) for x in q)
                mine = 0.0
                if self.pid:
                    a = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                                       capture_output=True, text=True, timeout=2).stdout
                    kids = {self.pid}
                    try: kids |= {c.pid for c in self.psutil.Process(self.pid).children(recursive=True)} if self.psutil else set()
                    except Exception: pass
                    for l in a.strip().split('\n'):
                        t = [x.strip() for x in l.split(',')]
                        if len(t) == 2 and t[0].isdigit() and int(t[0]) in kids: mine += float(t[1])
                self.m_gpu.set(util / 100, f'{util:.0f} % (whole GPU)')
                self.m_vram.set(used / tot, f'this step {mine / 1024:.1f} GB, all {used / 1024:.1f} of {tot / 1024:.0f} GB')
            except Exception: self.m_gpu.set(0, 'n/a'); self.m_vram.set(0, 'n/a')
        else:
            self.m_gpu.set(0, 'no NVIDIA GPU'); self.m_vram.set(0, '-')

    def _walk(self):
        """size + mtime of the files under the project (big folders are summarised)."""
        out = {}
        if not self.p: return out
        root = self.p.root
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in SKIP_DIRS]
            for f in fns:
                fp = os.path.join(dp, f)
                try: st = os.stat(fp); out[os.path.relpath(fp, root)] = (st.st_size, st.st_mtime)
                except OSError: pass
        return out

    def _diff(self, now):
        t = time.strftime('%H:%M:%S')
        for k, (sz, mt) in now.items():
            old = self.snap.get(k)
            if old is None: self.changes[k] = (sz, f'new  {t}')
            elif old != (sz, mt): self.changes[k] = (sz, (f'+{_fmt_bytes(sz - old[0])}  {t}' if sz > old[0] else f'rewritten  {t}'))
        for k in set(self.snap) - set(now): self.changes[k] = (0, f'deleted  {t}')
        # group many files in the same folder (frames, features...) into one row
        groups = {}
        for k, (sz, ch) in self.changes.items():
            d = os.path.dirname(k); groups.setdefault(d, []).append((k, sz, ch))
        rows = []
        for d, items in groups.items():
            if len(items) > 12:
                rows.append((f'{d}/  ({len(items)} files)', sum(i[1] for i in items), max(i[2] for i in items)))
            else: rows += items
        rows.sort(key=lambda r: r[2].split()[-1], reverse=True)
        self.files.setRowCount(len(rows))
        for r, (k, sz, ch) in enumerate(rows):
            for c, val in enumerate((k, _fmt_bytes(sz), ch)):
                it = QtWidgets.QTableWidgetItem(val); it.setData(Qt.UserRole, k); self.files.setItem(r, c, it)

    def open_file(self, it):
        k = it.data(Qt.UserRole)
        if self.p and k: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(os.path.join(self.p.root, k.split('  (')[0].rstrip('/'))))

    def _header(self):
        if not self.step: return
        from .style import tidy
        el = time.time() - self.t0 if self.t0 else 0
        state = ('running' if self.code is None else 'finished OK' if self.code == 0 else 'ended' if self.code == 'ended' else f'FAILED (exit {self.code})')
        vb = '' if self.verbose is None else ('   verbose: on (library logs shown)' if self.verbose else '   verbose: off (tick "verbose" in the toolbar to see library logs)')
        self.head.setText(f'<b>{self.step}</b> &nbsp; {state} &nbsp; elapsed {int(el // 60)}:{int(el % 60):02d} &nbsp; process {self.pid or "-"}'
                          f' &nbsp; output lines {len(self.lines)}<br><span style="color:#6b7280">command: {tidy(self.cmd)}</span>'
                          f'<br><span style="color:#6b7280">now: {self.stage or "-"}{vb}</span>')
