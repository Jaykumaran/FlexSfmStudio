"""`flexsfm doctor`: check that everything FlexSfM Studio needs is installed and working, and say how to fix what is not."""
import importlib, os, shutil, subprocess, sys

OK, WARN, FAIL = 'ok', 'warn', 'FAIL'
_rows = []


def _row(level, what, detail, fix=''):
    _rows.append((level, what, detail, fix))
    mark = {OK: '\033[32m  ok \033[0m', WARN: '\033[33m warn\033[0m', FAIL: '\033[31m FAIL\033[0m'}[level] if sys.stdout.isatty() else f'{level:>5}'
    print(f'{mark}  {what:22s} {detail}')
    if fix and level != OK: print(f'{"":29s}-> {fix}')


def _version(mod):
    try:
        m = importlib.import_module(mod); return getattr(m, '__version__', 'installed'), m
    except Exception as e:
        return None, e


def main(full=False):
    print('FlexSfM Studio - installation check\n')
    v = sys.version_info
    _row(OK if v >= (3, 10) else FAIL, 'python', sys.version.split()[0], 'Python 3.10 or newer is needed - run ./install.sh (it can fetch one with uv)')
    for mod, why in (('numpy', ''), ('scipy', ''), ('cv2', 'opencv-python'), ('PIL', 'pillow'), ('yaml', 'pyyaml'), ('matplotlib', '')):
        ver, m = _version(mod)
        _row(OK if ver else FAIL, mod, ver or f'missing ({m})', f'pip install {why or mod}')
    # torch / GPU
    ver, torch = _version('torch')
    if not ver:
        _row(FAIL, 'torch', f'missing ({torch})', 'run ./install.sh, or pip install torch (see pytorch.org for the CUDA build)')
    else:
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else 'no CUDA GPU visible'
        _row(OK if cuda else WARN, 'torch', f'{ver}  ({dev})',
             'FlexGlue runs on the CPU too, but much slower. For an NVIDIA GPU install the CUDA build of torch matching your driver (./install.sh does this).')
    ver, tv = _version('torchvision'); _row(OK if ver else FAIL, 'torchvision', ver or 'missing', 'pip install torchvision (same CUDA build as torch)')
    # pycolmap
    ver, pc = _version('pycolmap')
    if not ver: _row(FAIL, 'pycolmap', 'missing', 'pip install "pycolmap>=4.0"')
    else:
        major = int(str(ver).split('.')[0]) if str(ver)[0].isdigit() else 0
        has_global = hasattr(pc, 'global_mapping')
        _row(OK if major >= 4 and has_global else FAIL, 'pycolmap (COLMAP)', f'{ver}' + ('' if has_global else ' - no global (GLOMAP) mapper'),
             'pip install -U "pycolmap>=4.0"')
    # matchers
    ver, fg = _version('flexglue'); _row(OK if ver else FAIL, 'flexglue', ver or 'missing', 'pip install "flexglue @ git+https://github.com/cvg/FlexGlue.git"')
    ver, lg = _version('lightglue'); _row(OK if ver else WARN, 'lightglue (ALIKED)', ver or 'missing - only needed for the ALIKED detector',
                                         'pip install "lightglue @ git+https://github.com/cvg/LightGlue.git"')
    # ffmpeg
    exe = shutil.which('ffmpeg')
    if exe: _row(OK, 'ffmpeg', exe)
    else:
        ver, iff = _version('imageio_ffmpeg')
        _row(OK if ver else WARN, 'ffmpeg', f'bundled with imageio-ffmpeg {ver}' if ver else 'missing - needed only to read video files',
             'sudo apt install ffmpeg   (or: pip install imageio-ffmpeg)')
    # GUI
    ver, qt = _version('PySide6')
    if not ver: _row(WARN, 'PySide6 (app)', 'missing - the command line still works', 'pip install -e ".[gui]"')
    else:
        _row(OK, 'PySide6 (app)', ver)
        for mod in ('pyqtgraph', 'OpenGL'):
            v2, _ = _version(mod); _row(OK if v2 else WARN, mod, v2 or 'missing', 'pip install -e ".[gui]"')
        if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
            code = ('import sys\nfrom PySide6 import QtWidgets, QtGui\napp = QtWidgets.QApplication([])\n'
                    'c = QtGui.QOpenGLContext(); ok = c.create()\nprint(app.platformName(), "opengl", ok and c.format().version())')
            r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                out = r.stdout.strip().split('\n')[-1]; gl_ok = 'opengl False' not in out
                _row(OK if gl_ok else WARN, 'desktop + OpenGL', out, 'The 3D view needs OpenGL 2.1+; other tabs work without it.')
            else:
                err = (r.stderr or r.stdout).strip().split('\n')
                hint = 'sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 libgl1 libegl1'
                _row(FAIL, 'desktop (Qt)', next((l for l in err if 'plugin' in l.lower()), err[-1])[:110], hint + '   (then run again)')
        else:
            _row(WARN, 'desktop', 'no display ($DISPLAY not set) - use the command line, or ssh -X', '')
    # disk
    try:
        free = shutil.disk_usage(os.getcwd()).free / 1e9
        _row(OK if free > 20 else WARN, 'free disk here', f'{free:.0f} GB', 'Projects need about 2-3x the size of the frames; put them on a larger disk.')
    except Exception: pass
    if full and fg and not isinstance(fg, Exception):
        try:
            import numpy as np, torch as t
            from flexglue import FlexGlue
            dev = 'cuda' if t.cuda.is_available() else 'cpu'; m = FlexGlue().eval().to(dev)
            img = t.rand(1, 3, 256, 256, device=dev)
            with t.no_grad(): m.descriptor({'image': img})
            _row(OK, 'FlexGlue run', f'weights loaded, ran on {dev}')
        except Exception as e:
            _row(FAIL, 'FlexGlue run', f'{type(e).__name__}: {str(e)[:100]}', 'check the internet connection (weights are downloaded once) and the torch install')
    n_fail = sum(r[0] == FAIL for r in _rows); n_warn = sum(r[0] == WARN for r in _rows)
    print(f'\n{n_fail} problem(s), {n_warn} warning(s).' + ('  Ready.' if not n_fail else ''))
    return 1 if n_fail else 0
