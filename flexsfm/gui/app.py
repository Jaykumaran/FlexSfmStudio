"""FlexSfM Studio desktop app (PySide6). Every step runs as a separate process (python -m flexsfm run ...), so the window
stays responsive and a step can be stopped at any time."""
import json, os, sys, time
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from ..project import Project, STEP_ORDER, OPTIONAL, step_module
from . import viewers, style, live, timeline

STATUS_COLOR = {'done': '#2e7d32', 'stale': '#b26a00', 'running': '#1565c0', 'failed': '#c62828', 'interrupted': '#c62828', 'not run': '#757575'}
MODEL_INFO = {'initial': 'step 5, before GPS alignment', 'aligned': 'step 5, in GPS metres', 'markers': 'step 6, fitted to markers',
              'repaired': 'step 8, revisits tied', 'refined': 'step 9, refined poses', 'dense_raw': 'step 9, all dense points',
              'dense': 'step 10, filtered dense points'}
LEVEL_COLOR = {'pass': '#e8f5e9', 'warn': '#fff4e0', 'fail': '#fdecea'}
RECENT = os.path.expanduser('~/.flexsfm_recent.json')
_LAST_DIR = [None]   # last folder used in a file dialog (starts next to the most recent project, else home)


def start_dir(*cands):
    for c in cands:
        c = (c or '').strip()
        if c and os.path.isdir(c): return c
        if c and os.path.isdir(os.path.dirname(c)): return os.path.dirname(c)
    if _LAST_DIR[0] is None:
        r = recent_list(); _LAST_DIR[0] = os.path.dirname(r[0]) if r else os.path.expanduser('~')
    return _LAST_DIR[0]


def remember(path):
    if path: _LAST_DIR[0] = path if os.path.isdir(path) else os.path.dirname(path)
    return path


FILTERS = {'srt': 'DJI subtitles (*.SRT *.srt);;All files (*)', 'csv': 'CSV (*.csv *.txt);;All files (*)',
           'video': 'Video (*.mp4 *.MP4 *.mov *.MOV *.avi *.mkv);;All files (*)'}


def reflow(text):
    """Step descriptions are hard-wrapped in the source: join wrapped lines back into paragraphs, but keep
    '- ' bullets, '1. ' numbered points and 'Check:' / 'Note' lines each on their own line (indented lines continue the item above)."""
    import re
    items = []                                   # (kind, text): kind = p / ul / ol / check
    for line in text.split('\n'):
        t = line.strip()
        if not t: continue
        num = re.match(r'^(\d+)[.)]\s+(.*)', t)
        if t.startswith('- '): items.append(['ul', t[2:]])
        elif num: items.append(['ol', num.group(2)])
        elif t.startswith(('Check', 'Checks', 'Note')): items.append(['check' if t.startswith('Check') else 'p', t])
        elif items and (line[:1].isspace() and items[-1][0] in ('ul', 'ol') or items[-1][0] == 'p' and not items[-1][1].endswith(':')):
            items[-1][1] += ' ' + t              # continuation of the item / paragraph above
        else: items.append(['p', t])
    esc = lambda x: x.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    out, open_ = [], None
    for kind, t in items:
        if open_ and kind != open_: out.append(f'</{open_}>'); open_ = None
        if kind in ('ul', 'ol'):
            if not open_: out.append(f'<{kind} style="margin-top:2px; margin-bottom:6px">'); open_ = kind
            out.append(f'<li style="margin-bottom:3px">{esc(t)}</li>')
        elif kind == 'check': out.append(f'<p style="color:#065f46"><b>{esc(t.split(":")[0])}:</b>{esc(":".join(t.split(":")[1:]))}</p>')
        else: out.append(f'<p>{esc(t)}</p>')
    if open_: out.append(f'</{open_}>')
    return ''.join(out)


def recent_list():
    try: return [p for p in json.load(open(RECENT)) if os.path.exists(os.path.join(p, 'project.yaml'))]
    except Exception: return []


def add_recent(p):
    r = [p] + [x for x in recent_list() if x != p]
    try: json.dump(r[:10], open(RECENT, 'w'))
    except Exception: pass


# ======================================================================================== settings form
class ParamForm(QtWidgets.QWidget):
    def __init__(self):
        super().__init__(); self.lay = QtWidgets.QVBoxLayout(self); self.lay.setContentsMargins(0, 0, 0, 0); self.widgets = {}; self.mod = None
        self.lay.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)      # never squeeze rows - let the scroll area scroll
        self.adv = QtWidgets.QCheckBox('show advanced settings'); self.adv.toggled.connect(lambda _: self.build(self.mod, self.vals))
        self.lay.addWidget(self.adv); self.box = QtWidgets.QWidget(); self.form = QtWidgets.QFormLayout(self.box); self.lay.addWidget(self.box); self.lay.addStretch(1)
        self.vals = {}

    def build(self, mod, values):
        if mod is None: return
        self.mod, self.vals = mod, dict(values)
        while self.form.rowCount(): self.form.removeRow(0)
        self.widgets = {}
        for q in mod.PARAMS:
            if q['advanced'] and not self.adv.isChecked(): continue
            w = self._widget(q, values.get(q['key'], q['default'])); w.setMinimumHeight(30)
            lab = QtWidgets.QLabel(q['label']); lab.setToolTip(q['help']); w.setToolTip(q['help'])
            import textwrap
            hl = QtWidgets.QLabel('\n'.join(textwrap.wrap(q['help'], 72))); hl.setWordWrap(False); hl.setStyleSheet('color:#666; font-size:11px')
            col = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(col); v.setContentsMargins(0, 0, 0, 8); v.setSpacing(3); v.addWidget(w); v.addWidget(hl)
            self.form.addRow(lab, col); self.widgets[q['key']] = (q, w)

    def _widget(self, q, val):
        k = q['kind']
        if k == 'bool': w = QtWidgets.QCheckBox(); w.setChecked(bool(val)); return w
        if k == 'choice': w = QtWidgets.QComboBox(); w.addItems(q['choices']); w.setCurrentText(str(val)); return w
        if k == 'int':
            w = QtWidgets.QSpinBox(); w.setRange(int(q['lo']) if q['lo'] is not None else -10 ** 9, int(q['hi']) if q['hi'] is not None else 10 ** 9); w.setValue(int(val)); return w
        if k == 'float':
            w = QtWidgets.QDoubleSpinBox(); w.setDecimals(4); w.setRange(q['lo'] if q['lo'] is not None else -1e9, q['hi'] if q['hi'] is not None else 1e9)
            w.setSingleStep(max(0.01, abs(float(val)) / 10 if val else 0.1)); w.setValue(float(val)); return w
        if k in ('path', 'file', 'dir'):
            w = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
            e = QtWidgets.QLineEdit(str(val or '')); b = QtWidgets.QPushButton('Browse...'); h.addWidget(e, 1); h.addWidget(b); w.edit = e
            def pick():
                sd = start_dir(e.text())
                if k == 'dir': r = QtWidgets.QFileDialog.getExistingDirectory(self, q['label'], sd)
                elif k == 'file':
                    flt = FILTERS['srt'] if q['key'] == 'srt' else FILTERS['csv'] if 'csv' in q['key'] else 'All files (*)'
                    r = QtWidgets.QFileDialog.getOpenFileName(self, q['label'], sd, flt)[0]
                else:
                    m = QtWidgets.QMessageBox(self); m.setText('Pick a video file or an image folder?'); bv = m.addButton('Video file', QtWidgets.QMessageBox.AcceptRole)
                    m.addButton('Image folder', QtWidgets.QMessageBox.AcceptRole); m.exec()
                    r = QtWidgets.QFileDialog.getOpenFileName(self, 'Video', sd, FILTERS['video'])[0] if m.clickedButton() is bv \
                        else QtWidgets.QFileDialog.getExistingDirectory(self, 'Image folder', sd)
                if r: e.setText(remember(r))
            b.clicked.connect(pick); return w
        w = QtWidgets.QLineEdit(str(val)); return w

    def values(self):
        out = dict(self.vals)
        for key, (q, w) in self.widgets.items():
            k = q['kind']
            out[key] = (w.isChecked() if k == 'bool' else w.currentText() if k == 'choice' else w.value() if k in ('int', 'float')
                        else w.edit.text().strip() if k in ('path', 'file', 'dir') else w.text())
        return out


# ======================================================================================== results
class ResultsView(QtWidgets.QWidget):
    open_figure = QtCore.Signal(str, str)

    def __init__(self):
        super().__init__(); lay = QtWidgets.QVBoxLayout(self)
        sp = QtWidgets.QSplitter(Qt.Vertical); lay.addWidget(sp)
        top = QtWidgets.QSplitter(Qt.Horizontal); sp.addWidget(top)
        self.metrics = QtWidgets.QTreeWidget(); self.metrics.setHeaderLabels(['number', 'value']); self.metrics.setColumnWidth(0, 260); top.addWidget(self.metrics)
        self.checks = QtWidgets.QTableWidget(0, 4); self.checks.setHorizontalHeaderLabels(['check', 'result', 'expected', 'what to do'])
        self.checks.horizontalHeader().setStretchLastSection(True); self.checks.setWordWrap(True); self.checks.verticalHeader().setVisible(False); top.addWidget(self.checks)
        top.setSizes([380, 820])
        self.figs = QtWidgets.QListWidget(); self.figs.setViewMode(QtWidgets.QListView.IconMode); self.figs.setIconSize(QtCore.QSize(360, 220))
        self.figs.setResizeMode(QtWidgets.QListView.Adjust); self.figs.setWordWrap(True); self.figs.setSpacing(8)
        self.figs.itemActivated.connect(lambda it: self.open_figure.emit(it.data(Qt.UserRole), it.toolTip()))
        self.figs.itemClicked.connect(lambda it: self.open_figure.emit(it.data(Qt.UserRole), it.toolTip()))
        sp.addWidget(self.figs); sp.setSizes([330, 520])

    def show_step(self, p, step):
        d = p.report_dir(step); self.metrics.clear(); self.checks.setRowCount(0); self.figs.clear()
        def add(parent, k, v):
            it = QtWidgets.QTreeWidgetItem([str(k), '' if isinstance(v, (dict, list)) and len(str(v)) > 80 else str(v)])
            (parent.addChild(it) if parent else self.metrics.addTopLevelItem(it))
            if isinstance(v, dict) and len(str(v)) > 80:
                for a, b in v.items(): add(it, a, b)
            elif isinstance(v, list) and len(str(v)) > 80:
                for a, b in enumerate(v): add(it, a, b)
        st = p.step_state(step)
        if st.get('error'): add(None, 'ERROR', st['error'])
        if 'seconds' in st: add(None, 'run time (s)', st['seconds'])
        try:
            for k, v in json.load(open(os.path.join(d, 'metrics.json'))).items(): add(None, k, v)
        except Exception: pass
        try:
            ch = json.load(open(os.path.join(d, 'checks.json')))
        except Exception: ch = []
        self.checks.setRowCount(len(ch))
        for r, c in enumerate(ch):
            for col, txt in enumerate([c['name'], f"{c['level'].upper()}  {c['value']}", c['expect'], c['fix']]):
                it = QtWidgets.QTableWidgetItem(txt); it.setBackground(QtGui.QColor(LEVEL_COLOR.get(c['level'], '#ffffff'))); it.setForeground(QtGui.QColor('#202020'))
                self.checks.setItem(r, col, it)
        self.checks.resizeRowsToContents(); self.checks.resizeColumnToContents(0); self.checks.resizeColumnToContents(1)
        try:
            figs = json.load(open(os.path.join(d, 'figures.json')))
        except Exception:
            figs = [dict(file=f, caption=f) for f in sorted(os.listdir(d)) if f.endswith('.png')]
        for f in figs:
            path = os.path.join(d, f['file'])
            if not os.path.exists(path): continue
            pm = QtGui.QPixmap(path).scaled(360, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            it = QtWidgets.QListWidgetItem(QtGui.QIcon(pm), f['caption'][:70]); it.setData(Qt.UserRole, path); it.setToolTip(f['caption']); self.figs.addItem(it)


# ======================================================================================== issues
class IssuesView(QtWidgets.QWidget):
    apply_fix = QtCore.Signal(str, dict)

    def __init__(self):
        super().__init__(); self.lay = QtWidgets.QVBoxLayout(self)
        _l = QtWidgets.QLabel('Problems found by the checks of every step and by step 7 (diagnose). "Apply" sets the suggested settings and runs the step.'); _l.setWordWrap(True); self.lay.addWidget(_l)
        self.list = QtWidgets.QTableWidget(0, 4); self.list.setHorizontalHeaderLabels(['step', 'problem', 'what to do', ''])
        self.list.horizontalHeader().setStretchLastSection(False); self.list.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.list.setWordWrap(True); self.list.verticalHeader().setVisible(False); self.lay.addWidget(self.list)

    def refresh(self, p):
        rows = []
        try:
            for it in json.load(open(p.path('issues.json'))): rows.append(('diagnose', it['severity'], it['title'] + ': ' + it['detail'], it['action']['label'], it['action']))
        except Exception: pass
        for s in STEP_ORDER:
            try: ch = json.load(open(os.path.join(p.report_dir(s), 'checks.json')))
            except Exception: continue
            for c in ch:
                if c['level'] != 'pass': rows.append((s, c['level'], f"{c['name']}: {c['value']} (expected {c['expect']})", c['fix'], None))
        self.list.setRowCount(len(rows))
        for r, (s, lvl, txt, fix, act) in enumerate(rows):
            for col, t in enumerate([s, txt, fix]):
                it = QtWidgets.QTableWidgetItem(t); it.setBackground(QtGui.QColor(LEVEL_COLOR.get(lvl, '#fff'))); it.setForeground(QtGui.QColor('#202020')); self.list.setItem(r, col, it)
            if act:
                b = QtWidgets.QPushButton('Apply: ' + act['label']); b.clicked.connect(lambda _=False, a=act: self.apply_fix.emit(a['step'], a.get('params', {})))
                self.list.setCellWidget(r, 3, b)
            else:
                self.list.setCellWidget(r, 3, None)
        self.list.resizeRowsToContents(); self.list.resizeColumnToContents(0); self.list.setColumnWidth(1, 480); self.list.resizeColumnToContents(3)


# ======================================================================================== main window
class Main(QtWidgets.QMainWindow):
    def __init__(self, project=None):
        super().__init__(); self.setWindowTitle('FlexSfM Studio'); self.fit_to_screen(); self.p = None; self.proc = None; self.queue = []
        tb = self.addToolBar('main'); tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon); tb.setIconSize(QtCore.QSize(18, 18))
        for txt, fn, ic, col, name in (('New project', self.new_project, 'new', '#374151', ''), ('Open project', self.open_project, 'open', '#374151', ''), (None,) * 5,
                                       ('Run step', self.run_selected, 'play', '#ffffff', 'run'), ('Run from here to the end', self.run_from_here, 'playall', '#ffffff', 'runall'),
                                       ('Stop', self.stop, 'stop', '#ffffff', 'stop'), (None,) * 5,
                                       ('Open project folder', self.open_folder, 'folder', '#374151', ''), ('Help', self.help, 'help', '#2563eb', '')):
            if txt is None: tb.addSeparator(); continue
            a = QtGui.QAction(timeline.make_icon(ic, col), txt, self); a.triggered.connect(fn); tb.addAction(a)
            setattr(self, 'act_' + (name or ic), a)
            if name: tb.widgetForAction(a).setObjectName(name)
        self.act_stop.setEnabled(False)
        self.verbose = QtWidgets.QCheckBox('verbose'); self.verbose.setToolTip('Show everything the libraries print (COLMAP / GLOMAP / Ceres iterations, '
                                                                                  'library warnings) in the Live tab. Takes effect for the next step that starts.')
        tb.addWidget(self.verbose); tb.addSeparator()
        self.recent = QtWidgets.QComboBox(); self.recent.setMinimumWidth(380); self.recent.addItem('recent projects...'); self.recent.addItems(recent_list())
        self.recent.activated.connect(lambda i: self.load(self.recent.itemText(i)) if i > 0 else None); tb.addWidget(self.recent)
        # left: steps
        dock = QtWidgets.QDockWidget('Steps', self); dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        self.steps = QtWidgets.QListWidget(); self.steps.setObjectName('steps'); self.steps.setItemDelegate(timeline.TimelineDelegate(self.steps))
        self.steps.setMouseTracking(True); self.steps.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff); self.steps.setMinimumWidth(240); self.steps.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.pulse = QtCore.QTimer(self); self.pulse.timeout.connect(self.steps.viewport().update)
        self.steps.setMinimumWidth(240); self.steps.currentRowChanged.connect(self.select_step); v.addWidget(self.steps)
        self.model_lbl = QtWidgets.QLabel(); self.model_lbl.setWordWrap(True); self.model_lbl.setTextFormat(Qt.RichText); v.addWidget(self.model_lbl)
        self.model_pick = QtWidgets.QComboBox(); self.model_pick.activated.connect(self.set_current_model)
        self.model_pick.setToolTip('The model (cameras + points) that the next steps start from. Each step that changes the model '
                                   'writes a new one under models/ and makes it current; pick an earlier one here to go back.')
        v.addWidget(self.model_pick); self._models = None
        dock.setWidget(w); self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        # centre tabs
        self.tabs = QtWidgets.QTabWidget(); self.setCentralWidget(self.tabs)
        stepw = QtWidgets.QWidget(); sl = QtWidgets.QVBoxLayout(stepw)
        self.title = QtWidgets.QLabel(); self.title.setObjectName('title'); sl.addWidget(self.title)
        sp = QtWidgets.QSplitter(Qt.Vertical); sl.addWidget(sp, 1)
        upper = QtWidgets.QSplitter(Qt.Horizontal); sp.addWidget(upper)
        self.what = QtWidgets.QTextBrowser(); upper.addWidget(self.what)
        sc = QtWidgets.QScrollArea(); sc.setWidgetResizable(True); self.form = ParamForm(); sc.setWidget(self.form); upper.addWidget(sc); upper.setSizes([520, 760])
        low = QtWidgets.QWidget(); ll = QtWidgets.QVBoxLayout(low); ll.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout(); self.run_btn = QtWidgets.QPushButton('Run this step'); self.run_btn.setObjectName('primary')
        self.run_btn.clicked.connect(self.run_selected); row.addWidget(self.run_btn)
        self.stop_btn = QtWidgets.QPushButton('Stop'); self.stop_btn.setObjectName('danger'); self.stop_btn.setIcon(timeline.make_icon('stop', '#ffffff'))
        self.stop_btn.clicked.connect(self.stop); self.stop_btn.setEnabled(False); row.addWidget(self.stop_btn)
        self.run_btn.setIcon(timeline.make_icon('play', '#ffffff'))
        rd = QtWidgets.QPushButton('Reset to defaults'); rd.clicked.connect(self.reset_defaults); row.addWidget(rd)
        sv = QtWidgets.QPushButton('Save settings'); sv.clicked.connect(self.save_settings); row.addWidget(sv)
        self.prog = QtWidgets.QProgressBar(); self.prog.setRange(0, 1000); self.prog.setMinimumWidth(80); row.addWidget(self.prog, 1); self.prog_lbl = QtWidgets.QLabel(); self.prog_lbl.setMinimumWidth(0); self.prog_lbl.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred); row.addWidget(self.prog_lbl, 1)
        ll.addLayout(row)
        self.log = QtWidgets.QPlainTextEdit(); self.log.setObjectName('log'); self.log.setReadOnly(True); self.log.setMaximumBlockCount(20000)
        self.log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)); ll.addWidget(self.log)
        sp.addWidget(low); sp.setSizes([430, 450])
        self.tabs.addTab(stepw, 'Step')
        self.live = live.LiveView(); self.tabs.addTab(self.live, 'Live')
        self.map = viewers.MapView(); self.tabs.addTab(self.map, 'Map')
        self.results = ResultsView(); self.results.open_figure.connect(self.show_figure); self.tabs.addTab(self.results, 'Results (checks + figures)')
        self.issues = IssuesView(); self.issues.apply_fix.connect(self.apply_fix); self.tabs.addTab(self.issues, 'Issues')
        self.fig = viewers.ImageView(); self.tabs.addTab(self.fig, 'Figure')
        self.view3d = viewers.Model3D(); self.tabs.addTab(self.view3d, '3D view')
        self.frames = viewers.FrameInspector(); self.tabs.addTab(self.frames, 'Frames')
        self.pairs = viewers.PairInspector(); self.tabs.addTab(self.pairs, 'Pairs / matches')
        self.markers = viewers.MarkerTool(); self.tabs.addTab(self.markers, 'Markers')
        self.tabs.currentChanged.connect(self.tab_changed)
        self.statusBar().showMessage('Open or create a project')
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.refresh_steps); self.timer.start(3000)
        if project: self.load(project)
        else: self.what.setHtml(WELCOME)

    def fit_to_screen(self):
        """Open at a size that fits the screen under the mouse (maximized on small screens), centred."""
        scr = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos()) or QtGui.QGuiApplication.primaryScreen(); g = scr.availableGeometry()
        w, h = min(1600, int(g.width() * 0.9)), min(1000, int(g.height() * 0.9)); self.resize(w, h)
        self.move(g.x() + (g.width() - w) // 2, g.y() + (g.height() - h) // 2)
        self._start_max = g.width() < 1366 or g.height() < 800

    # ------------------------------------------------------------------ project
    def new_project(self):
        d = NewProjectDialog(self)
        if d.exec() != QtWidgets.QDialog.Accepted: return
        v = d.values()
        try:
            Project.create(v['project'], source=v['source'], srt=v['srt'], csv=v['csv'], masks=v['masks'], step=v['step'], gps_source=v['gps'])
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'New project', str(e)); return
        self.load(v['project'])

    def open_project(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, 'Open project folder', start_dir())
        if d: self.load(d)

    def load(self, path):
        try: self.p = Project(path)
        except Exception as e: QtWidgets.QMessageBox.critical(self, 'Open project', str(e)); return
        add_recent(self.p.root); self.setWindowTitle(style.tidy(f'FlexSfM Studio - {self.p.root}'))
        for v in (self.live, self.map, self.view3d, self.frames, self.pairs, self.markers): v.set_project(self.p)
        self.refresh_steps(); self.steps.setCurrentRow(0 if self.p.status('ingest') != 'done' else self.first_not_done())
        self.statusBar().showMessage(style.tidy(f'project {self.p.root}'))

    def first_not_done(self):
        for k, s in enumerate(STEP_ORDER):
            if s not in OPTIONAL and self.p.status(s) != 'done': return k
        return len(STEP_ORDER) - 1

    def open_folder(self):
        if self.p: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.p.root))

    def help(self):
        self.what.setHtml(HELP); self.tabs.setCurrentIndex(0)

    # ------------------------------------------------------------------ steps list
    def refresh_steps(self):
        if not self.p: return
        try: self.p.reload()
        except Exception: return                      # state file being written - try again on the next tick
        if self.steps.count() != len(STEP_ORDER):
            self.steps.blockSignals(True); self.steps.clear(); self.steps.addItems([''] * len(STEP_ORDER)); self.steps.blockSignals(False)
        for k, s in enumerate(STEP_ORDER):              # update in place (keeps selection and scroll position)
            st = self.p.status(s); c = self.p.step_state(s).get('checks', {})
            chk = f"   {c.get('fail', 0)} fail, {c.get('warn', 0)} warn" if c and st in ('done', 'stale') else ''
            if st == 'running' and self.proc is None:
                chk = '   (in another process)'
                pid = self.p.step_state(s).get('pid')
                if pid: self.live.attach(self.p, s, int(pid))
            num, title = step_module(s).TITLE.split(None, 1); title = title.replace('(optional)', '').strip()
            it = self.steps.item(k); it.setText(f'{num}. {title}'); it.setToolTip(step_module(s).WHAT.split('\n')[0] + (chk and f'\n{chk.strip()}'))
            ss = self.p.step_state(s)
            vals = {timeline.R_STATUS: st, timeline.R_CHECKS: c, timeline.R_SECS: ss.get('seconds'), timeline.R_OPT: s in OPTIONAL, timeline.R_NUM: num,
                    timeline.R_TITLE: title, timeline.R_PREV: self.p.status(STEP_ORDER[k - 1]) if k else None,
                    timeline.R_NEXT: self.p.status(STEP_ORDER[k + 1]) if k + 1 < len(STEP_ORDER) else None}
            for role, val in vals.items():
                if it.data(role) != val: it.setData(role, val)
        running = bool(self.proc) or any(self.p.status(x) == 'running' for x in STEP_ORDER)
        if running and not self.pulse.isActive(): self.pulse.start(60)
        elif not running and self.pulse.isActive(): self.pulse.stop(); self.steps.viewport().update()
        order = list(MODEL_INFO); models = sorted(viewers.models_of(self.p), key=lambda m: (order.index(m) if m in order else 99, m))
        cur = self.p.state.get('model')
        if models != self._models and not self.model_pick.view().isVisible():
            self._models = models; self.model_pick.blockSignals(True); self.model_pick.clear()
            for m in models: self.model_pick.addItem(f'{m}   ({MODEL_INFO.get(m, "")})' if m in MODEL_INFO else m, m)
            self.model_pick.blockSignals(False)
        if not self.model_pick.view().isVisible():
            self.model_pick.blockSignals(True); k = self.model_pick.findData(cur); self.model_pick.setCurrentIndex(k); self.model_pick.blockSignals(False)
        self.model_pick.setEnabled(bool(models))
        if cur: self.model_lbl.setText(f"<b>Current model:</b> {cur}<br><span style='color:#666'>The cameras that markers, diagnose, repair and "
                                       f"densify start from (models/{cur}). Densify / filter write refined, dense_raw and dense from it; export uses "
                                       f"dense when it exists. Pick another model below to go back to it.</span>")
        else: self.model_lbl.setText("<b>Current model:</b> none yet<br><span style='color:#666'>Step 5 (Reconstruct) creates the first model "
                                     "(cameras + points). Later steps add more; the list below lets you go back to any of them.</span>")

    def set_current_model(self, i):
        if not self.p: return
        name = self.model_pick.itemData(i)
        if name and name != self.p.state.get('model'):
            if QtWidgets.QMessageBox.question(self, 'Current model', f'Use models/{name} as the current model for the following steps?') == QtWidgets.QMessageBox.Yes:
                self.p.set_model(name)
        self.refresh_steps()

    def step(self):
        return STEP_ORDER[max(0, self.steps.currentRow())]

    def select_step(self, row):
        if not self.p or row < 0: return
        s = STEP_ORDER[row]; mod = step_module(s)
        self.title.setText(mod.TITLE + ('   (optional)' if s in OPTIONAL and 'optional' not in mod.TITLE else ''))
        self.what.setHtml('<div style="font-size:13px; line-height:1.5">' + reflow(mod.WHAT) + '</div>'
                          + (f'<p><b>needs:</b> {", ".join(mod.NEEDS)}</p>' if mod.NEEDS else ''))
        self.form.build(mod, self.p.params(s))
        lp = self.p.path('logs', f'{s}.log'); self.log.setPlainText(style.tidy(open(lp).read()) if os.path.exists(lp) else '')
        self.log.moveCursor(QtGui.QTextCursor.End)
        self.results.show_step(self.p, s)

    def save_settings(self):
        if self.p: self.p.set_params(self.step(), self.form.values()); self.refresh_steps(); self.statusBar().showMessage('settings saved')

    def reset_defaults(self):
        if not self.p: return
        s = self.step(); keep = {k: v for k, v in (self.p.cfg.get(s) or {}).items() if k in ('source', 'srt', 'csv', 'masks', 'markers_csv')}
        self.p.cfg[s] = keep; self.p.save_cfg(); self.select_step(self.steps.currentRow())

    # ------------------------------------------------------------------ running
    def run_selected(self):
        if not self.p: return
        self.save_settings(); self.queue = [self.step()]; self.start_next()

    def run_from_here(self):
        if not self.p: return
        self.save_settings(); i = self.steps.currentRow()
        self.queue = [s for s in STEP_ORDER[i:] if s not in OPTIONAL or s == self.step()]; self.start_next()

    def start_next(self):
        if self.proc or not self.queue: return
        if self.p.status(self.queue[0]) == 'running':
            QtWidgets.QMessageBox.warning(self, 'Run', f'"{step_module(self.queue[0]).TITLE}" is already running in another process '
                                          '(another window or the command line). Wait for it to finish.'); self.queue = []; return
        s = self.queue.pop(0); self._buf = ''; self.steps.setCurrentRow(STEP_ORDER.index(s))
        if self.tabs.currentWidget() is not self.live: self.tabs.setCurrentIndex(0)      # keep watching Live if that is open
        self.log.clear(); self.prog.setValue(0); self.prog_lbl.setText('starting ...')
        self.proc = QtCore.QProcess(self); self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        env = QtCore.QProcessEnvironment.systemEnvironment(); env.insert('PYTHONUNBUFFERED', '1'); vb = self.verbose.isChecked()
        env.insert('GLOG_minloglevel', '0' if vb else '1'); env.insert('GLOG_logtostderr', '1'); env.insert('FLEXSFM_VERBOSE', '1' if vb else '0')
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.proc.readyReadStandardOutput.connect(self.read_out); self.proc.finished.connect(lambda code, _st, s=s: self.finished(s, code))
        self.live.start(self.p, s, f'python -m flexsfm run {self.p.root} {s}', vb)
        self.proc.started.connect(lambda: self.live.started(self.proc.processId() if self.proc else None))
        self.proc.start(sys.executable, ['-m', 'flexsfm', 'run', self.p.root, s]); self.running_step = s
        self.statusBar().showMessage(f'running {s} ...'); self.t_start = time.time(); self.set_running(True)
        QtCore.QTimer.singleShot(1500, self.refresh_steps)

    def read_out(self):
        if not self.proc: return
        self._buf += bytes(self.proc.readAllStandardOutput()).decode(errors='replace')
        *lines, self._buf = self._buf.replace('\r', '\n').split('\n')     # keep an unfinished last line for the next read
        for line in lines:
            if not line.strip(): continue
            self.live.line(line)                              # the Live tab gets everything, unfiltered
            if line.startswith('@@P '):
                parts = line.split(' ', 2)
                try: f = float(parts[1])
                except (IndexError, ValueError): continue
                self.prog.setValue(int(f * 1000))
                el = time.time() - self.t_start; eta = f' - about {el / f * (1 - f) / 60:.0f} min left' if f > 0.03 else ''
                self.prog_lbl.setText((parts[2] if len(parts) > 2 else '') + eta)
            elif line.startswith('@@F ') or line.startswith('@@C '):
                if line.startswith('@@C ') and not line.startswith('@@C PASS'): self.log.appendPlainText('CHECK ' + line[4:])
            elif 'warn' in line.lower() and ('torch' in line or 'UserWarning' in line):
                continue
            else:
                self.log.appendPlainText(style.tidy(line))

    def finished(self, s, code):
        if self.proc:                                   # flush the last (possibly unterminated) output line
            self.read_out(); self._buf += '\n'; self.read_out()
        self.live.finish(code)
        self.proc = None; self.set_running(False); self.prog.setValue(1000 if code == 0 else self.prog.value())
        self.prog_lbl.setText('done' if code == 0 else f'FAILED (exit {code}) - see the log'); self.refresh_steps()
        self.results.show_step(self.p, s); self.issues.refresh(self.p)
        if code != 0:
            self.queue = []; self.statusBar().showMessage(f'{s} failed - see the log'); return
        self.statusBar().showMessage(f'{s} finished')
        for v in (self.live, self.map, self.view3d, self.frames, self.pairs, self.markers): v.set_project(self.p)
        if self.queue: self.start_next()
        else:
            c = self.p.step_state(s).get('checks', {})
            if c.get('fail') or c.get('warn'): self.tabs.setCurrentWidget(self.results)
            elif s == 'ingest': self.tabs.setCurrentWidget(self.map)          # show the trajectory right after import

    def set_running(self, on):
        for w in (self.run_btn, self.act_run, self.act_runall, self.act_new, self.act_open): w.setEnabled(not on)
        self.act_stop.setEnabled(on); self.stop_btn.setEnabled(on)
        if on and not self.pulse.isActive(): self.pulse.start(60)

    def stop(self):
        self.queue = []
        if self.proc:
            self.proc.terminate()
            if not self.proc.waitForFinished(5000): self.proc.kill()
            self.statusBar().showMessage('stopped')

    def apply_fix(self, step, params):
        if params: self.p.set_params(step, params)
        self.steps.setCurrentRow(STEP_ORDER.index(step))
        if QtWidgets.QMessageBox.question(self, 'Apply fix', f'Run "{step_module(step).TITLE}" now' + (f' with {params}' if params else '') + '?') == QtWidgets.QMessageBox.Yes:
            self.queue = [step]; self.start_next()

    def show_figure(self, path, caption):
        self.fig.load(path, caption); self.tabs.setCurrentWidget(self.fig)

    def tab_changed(self, i):
        w = self.tabs.widget(i)
        if w is self.issues and self.p: self.issues.refresh(self.p)
        if hasattr(w, 'activate'): w.activate()

    def closeEvent(self, e):
        if self.proc and QtWidgets.QMessageBox.question(self, 'Quit', 'A step is running. Stop it and quit?') != QtWidgets.QMessageBox.Yes:
            e.ignore(); return
        self.stop(); e.accept()


class NewProjectDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent); self.setWindowTitle('New project'); self.setMinimumWidth(min(760, int(QtGui.QGuiApplication.primaryScreen().availableGeometry().width() * 0.9))); f = QtWidgets.QFormLayout(self); self.w = {}
        def row(key, label, kind, tip):
            w = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0); e = QtWidgets.QLineEdit(); h.addWidget(e, 1)
            if kind:
                b = QtWidgets.QPushButton('Browse...'); h.addWidget(b)
                def pick():
                    sd = start_dir(e.text(), self.w['source'].text() if 'source' in self.w else '')
                    if kind == 'dir': r = QtWidgets.QFileDialog.getExistingDirectory(self, label, sd)
                    elif kind == 'save': r = QtWidgets.QFileDialog.getExistingDirectory(self, 'Folder that will hold the project (pick or create an empty folder)', sd)
                    elif kind == 'src':
                        m = QtWidgets.QMessageBox(self); m.setText('Is the source a video file or a folder of images?')
                        bv = m.addButton('Video file', QtWidgets.QMessageBox.AcceptRole); m.addButton('Image folder', QtWidgets.QMessageBox.AcceptRole); m.exec()
                        r = QtWidgets.QFileDialog.getOpenFileName(self, 'Video', sd, FILTERS['video'])[0] if m.clickedButton() is bv \
                            else QtWidgets.QFileDialog.getExistingDirectory(self, 'Image folder', sd)
                    else:
                        r = QtWidgets.QFileDialog.getOpenFileName(self, label, sd, FILTERS.get(key, 'All files (*)'))[0]
                    if r:
                        e.setText(remember(r))
                        if key == 'source': self.suggest_project()
                        if key == 'source' and os.path.isfile(r):
                            for c in (os.path.splitext(r)[0] + x for x in ('.SRT', '.srt')):
                                if os.path.exists(c) and not self.w['srt'].text(): self.w['srt'].setText(c)
                b.clicked.connect(pick)
            e.setToolTip(tip); f.addRow(label, w); lab = QtWidgets.QLabel(tip); lab.setStyleSheet('color:#666; font-size:11px'); lab.setWordWrap(True); f.addRow('', lab); self.w[key] = e
        row('project', 'project folder', 'save', 'Where all results go - type a new folder name, it is created. Filled in automatically from the source. Needs space (about 2-3x the frames); use the big disk.')
        row('source', 'video or image folder', 'src', 'A video file from any camera (frames are extracted) or a folder of photos / frames')
        row('srt', 'DJI .SRT (optional)', 'file', 'DJI drone videos: GPS per frame. Found automatically next to the video. Photos with GPS in EXIF need nothing here.')
        row('csv', 'GPS CSV (optional)', 'file', 'name + lat/lon/alt per image, e.g. metashape_reference.csv')
        row('masks', 'masks folder (optional)', 'dir', 'PNG per image, black = ignore. "<image folder>/masks" is used automatically.')
        self.step = QtWidgets.QSpinBox(); self.step.setRange(1, 300); self.step.setValue(15); f.addRow('frame step', self.step)
        f.addRow('', QtWidgets.QLabel('Video: keep every Nth frame. Image folder + SRT: the step the frames were extracted with.'))
        self.gps = QtWidgets.QComboBox(); self.gps.addItems(['auto', 'srt', 'exif', 'csv', 'none']); f.addRow('GPS source', self.gps)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel); bb.accepted.connect(self.accept); bb.rejected.connect(self.reject); f.addRow(bb)

    def values(self):
        return dict({k: os.path.expanduser(e.text().strip()) for k, e in self.w.items()}, step=self.step.value(), gps=self.gps.currentText())

    def suggest_project(self):
        src = self.w['source'].text().strip()
        if src and not self.w['project'].text().strip():
            base = os.path.dirname(src) if os.path.isfile(src) else os.path.dirname(src.rstrip('/'))
            name = os.path.splitext(os.path.basename(src.rstrip('/')))[0]
            self.w['project'].setText(os.path.join(base, 'flexsfm_' + name))

    def accept(self):
        v = self.values(); err = []
        if not v['source']: err.append('Pick the video file or the image folder ("video or image folder").')
        elif not os.path.exists(v['source']): err.append(f'Source not found: {v["source"]}')
        if not v['project']: err.append('Type or pick a project folder (it is created if it does not exist).')
        elif os.path.exists(os.path.join(v['project'], 'project.yaml')):
            err.append(f'{v["project"]} already holds a project - use "Open project" instead, or choose another folder.')
        for k in ('srt', 'csv', 'masks'):
            if v[k] and not os.path.exists(v[k]): err.append(f'{k} not found: {v[k]}')
        if err:
            QtWidgets.QMessageBox.warning(self, 'New project', '\n'.join(err)); return
        try:
            os.makedirs(v['project'], exist_ok=True)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'New project', f'Cannot create {v["project"]}: {e}'); return
        super().accept()


WELCOME = """<h2>FlexSfM Studio</h2><p>Step-by-step structure from motion: FlexGlue matching + COLMAP/GLOMAP reconstruction, with checks and
figures at every step.</p><ol><li><b>New project</b>: pick a video (with its .SRT) or an image folder (with or without GPS).</li>
<li>Click a step on the left, check its settings, press <b>Run this step</b> - or <b>Run from here to the end</b>.</li>
<li>After each step look at <b>Results</b>: green/yellow/red checks with what to do, and figures. <b>Issues</b> lists every problem with a fix.</li>
<li>Inspect intermediate outputs in <b>3D view</b>, <b>Frames</b> (keypoints / points / errors on each image) and <b>Pairs</b> (matches of any pair).</li>
<li>Step 11 writes <code>export/undistorted</code> for Gaussian-splat training.</li></ol>"""

HELP = WELCOME + """<h3>Typical fixes</h3><ul>
<li><b>Model broke into pieces / frames not placed</b>: step 4 link strength shows where. Raise <i>sequential</i> in step 3, re-run 3-5.</li>
<li><b>Doubled surfaces / ghosting where the capture passes again</b>: step 7 flags it; run step 8 (repair).</li>
<li><b>Splats soft / edges doubled</b>: densify check 'dense points precise' must pass: detector sift, snap px 0.7, track px 1.5.</li>
<li><b>Too few points for training</b>: step 10 table shows counts per filter setting; for more, raise <i>max keypoints</i> in step 9.</li>
<li><b>Bent model vs GPS over long distances</b>: step 5 <i>GPS bundle adjustment passes</i> = 1 (slow on CPU).</li>
<li><b>No GPS</b>: everything works; for metres/orientation use step 6 markers (3+ known points, or a known distance).</li></ul>
<p>Everything a step writes stays in the project folder: reports/&lt;step&gt;/ (figures, checks.json, metrics.json), logs/, models/.</p>
<p>Command line: <code>python -m flexsfm all PROJECT</code>, <code>python -m flexsfm run PROJECT STEP --set key=value</code>, <code>python -m flexsfm status PROJECT</code>.</p>"""


def main(project=None):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName('FlexSfM Studio'); style.apply(app)
    app.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True)   # the desktop's native dialog can hang / hide behind
    w = Main(project)
    w.showMaximized() if getattr(w, '_start_max', False) else w.show()
    app.exec()
