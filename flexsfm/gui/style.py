"""Look of the desktop app: one light theme (Qt style sheet), status icons and toolbar icons."""
from PySide6 import QtCore, QtGui, QtWidgets

ACCENT = '#2563eb'
STATUS = {'done': '#16a34a', 'stale': '#d97706', 'running': '#2563eb', 'failed': '#dc2626',
          'interrupted': '#dc2626', 'not run': '#9ca3af'}

QSS = """
QWidget { font-size: 13px; color: #1f2937; }
QMainWindow, QDialog { background: #f3f4f6; }
QToolBar { background: #ffffff; border: none; border-bottom: 1px solid #e5e7eb; padding: 4px 8px; spacing: 4px; }
QToolBar QToolButton { padding: 5px 10px; border-radius: 6px; }
QToolBar QToolButton:hover { background: #eef2ff; }
QToolBar QToolButton:pressed { background: #e0e7ff; }
QDockWidget { titlebar-close-icon: none; }
QDockWidget::title { background: #ffffff; padding: 8px 10px; font-weight: 600; border-bottom: 1px solid #e5e7eb; }
QListWidget#steps { background: #ffffff; border: none; outline: 0; padding: 6px 2px; }
QListWidget#steps::item, QListWidget#steps::item:hover, QListWidget#steps::item:selected { background: transparent; border: none; }
QToolButton#run { background: #16a34a; color: white; font-weight: 600; }
QToolButton#run:hover { background: #15803d; }
QToolButton#runall { background: #2563eb; color: white; font-weight: 600; }
QToolButton#runall:hover { background: #1d4ed8; }
QToolButton#stop { background: #dc2626; color: white; font-weight: 600; }
QToolButton#stop:hover { background: #b91c1c; }
QToolButton#run:disabled, QToolButton#runall:disabled, QToolButton#stop:disabled { background: #e5e7eb; color: #9ca3af; }
QPushButton#danger { background: #dc2626; color: white; border: 1px solid #dc2626; font-weight: 600; padding: 6px 18px; }
QPushButton#danger:hover { background: #b91c1c; }
QPushButton#danger:disabled { background: #fecaca; border-color: #fecaca; color: #ffffff; }
QTabWidget::pane { border: 1px solid #e5e7eb; border-radius: 8px; background: #ffffff; top: -1px; }
QTabBar::tab { background: transparent; padding: 8px 14px; margin-right: 2px; border: none; color: #4b5563; }
QTabBar::tab:selected { color: %(accent)s; border-bottom: 2px solid %(accent)s; font-weight: 600; }
QTabBar::tab:hover:!selected { color: #111827; }
QPushButton { background: #ffffff; border: 1px solid #d1d5db; border-radius: 6px; padding: 5px 12px; }
QPushButton:hover { border-color: #9ca3af; background: #f9fafb; }
QPushButton:disabled { color: #9ca3af; background: #f3f4f6; }
QPushButton#primary { background: %(accent)s; color: white; border: 1px solid %(accent)s; font-weight: 600; padding: 6px 18px; }
QPushButton#primary:hover { background: #1d4ed8; }
QPushButton#primary:disabled { background: #93c5fd; border-color: #93c5fd; }
QLineEdit, QComboBox { background: #ffffff; border: 1px solid #d1d5db; border-radius: 6px; padding: 4px 6px; min-height: 20px; }
QLineEdit:focus, QComboBox:focus { border-color: %(accent)s; }
QComboBox::drop-down { border: none; width: 20px; }
QProgressBar { background: #e5e7eb; border: none; border-radius: 6px; height: 12px; text-align: center; color: #111827; font-size: 11px; }
QProgressBar::chunk { background: %(accent)s; border-radius: 6px; }
QPlainTextEdit, QTextBrowser { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 6px; }
QPlainTextEdit#log { background: #0f172a; color: #e2e8f0; border: none; font-size: 12px; }
QTableWidget, QTreeWidget, QListWidget { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; gridline-color: #f1f5f9; }
QHeaderView::section { background: #f9fafb; border: none; border-bottom: 1px solid #e5e7eb; padding: 6px; font-weight: 600; color: #374151; }
QGroupBox { border: 1px solid #e5e7eb; border-radius: 8px; margin-top: 14px; padding: 8px; background: #ffffff; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #374151; font-weight: 600; }
QScrollArea { border: none; background: transparent; }
QSplitter::handle { background: #f3f4f6; }
QStatusBar { background: #ffffff; border-top: 1px solid #e5e7eb; color: #4b5563; }
QLabel#title { font-size: 20px; font-weight: 700; color: #111827; }
QLabel#subtle { color: #6b7280; font-size: 12px; }
QFrame#card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; }
QToolTip { background: #111827; color: #f9fafb; border: none; padding: 6px; border-radius: 4px; }
""" % dict(accent=ACCENT)


def dot(color, size=14):
    """Round status icon."""
    pm = QtGui.QPixmap(size * 2, size * 2); pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm); p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setBrush(QtGui.QColor(color)); p.setPen(QtCore.Qt.NoPen); p.drawEllipse(4, 4, size * 2 - 8, size * 2 - 8); p.end()
    pm.setDevicePixelRatio(2); return QtGui.QIcon(pm)


def std_icon(name):
    return QtWidgets.QApplication.style().standardIcon(getattr(QtWidgets.QStyle.StandardPixmap, name))


def apply(app):
    app.setStyle('Fusion'); app.setStyleSheet(QSS)
    pal = app.palette(); pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(ACCENT)); pal.setColor(QtGui.QPalette.Window, QtGui.QColor('#f3f4f6'))
    pal.setColor(QtGui.QPalette.Base, QtGui.QColor('#ffffff')); pal.setColor(QtGui.QPalette.Text, QtGui.QColor('#1f2937'))
    pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor('#1f2937')); pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor('#1f2937'))
    app.setPalette(pal)


_HOME = None


def tidy(text):
    """Show the home folder as ~ in anything displayed (logs, commands, paths) - shorter, and no user name on screen."""
    global _HOME
    import os
    if _HOME is None: _HOME = os.path.expanduser('~')
    return text.replace(_HOME, '~') if _HOME and _HOME != '/' else text
