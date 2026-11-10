"""FlowLayout: lays widgets out left to right and wraps to the next line when the window is narrow (no horizontal overflow)."""
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt


class FlowLayout(QtWidgets.QLayout):
    def __init__(self, parent=None, hspacing=6, vspacing=6):
        super().__init__(parent); self.items = []; self.hs, self.vs = hspacing, vspacing; self.setContentsMargins(0, 0, 0, 0)

    # drop-in for the QHBoxLayout calls used in the app
    def addWidget(self, w, *args, **kw): super().addWidget(w)
    def addStretch(self, *a): pass
    def addSpacing(self, *a): pass
    def addLayout(self, lay, *a):
        w = QtWidgets.QWidget(); w.setLayout(lay); self.addWidget(w)

    def addItem(self, item): self.items.append(item)
    def count(self): return len(self.items)
    def itemAt(self, i): return self.items[i] if 0 <= i < len(self.items) else None
    def takeAt(self, i): return self.items.pop(i) if 0 <= i < len(self.items) else None
    def expandingDirections(self): return Qt.Orientations(0)
    def hasHeightForWidth(self): return True
    def heightForWidth(self, w): return self._do(QtCore.QRect(0, 0, w, 0), True)

    def setGeometry(self, r):
        super().setGeometry(r); self._do(r, False)

    def sizeHint(self): return self.minimumSize()

    def minimumSize(self):
        s = QtCore.QSize()
        for it in self.items: s = s.expandedTo(it.minimumSize())
        m = self.contentsMargins(); return s + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do(self, rect, test):
        m = self.contentsMargins(); r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line = r.x(), r.y(), 0
        for it in self.items:
            if it.widget() is not None and not it.widget().isVisibleTo(it.widget().parentWidget() or it.widget()): continue
            sh = it.sizeHint(); nx = x + sh.width() + self.hs
            if nx - self.hs > r.right() + 1 and line > 0:
                x = r.x(); y += line + self.vs; nx = x + sh.width() + self.hs; line = 0
            if not test: it.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), sh))
            x = nx; line = max(line, sh.height())
        return y + line - rect.y() + m.bottom()


def group(flow, title):
    """A titled box added to a FlowLayout; returns the box's own horizontal layout (controls go in there)."""
    g = QtWidgets.QGroupBox(title); lay = QtWidgets.QHBoxLayout(g); lay.setContentsMargins(10, 6, 10, 6); lay.setSpacing(6); flow.addWidget(g); return lay


def label(text):
    l = QtWidgets.QLabel(text); l.setStyleSheet('color:#4b5563'); return l
