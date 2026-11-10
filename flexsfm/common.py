"""Shared helpers: parameter declaration, plotting, feature cache, FlexGlue loading."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

IMG_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')


def P(key, default, help, kind=None, choices=None, lo=None, hi=None, label=None, advanced=False):
    """One step setting. kind: int | float | bool | str | choice | path | file | dir (inferred from default)."""
    if kind is None:
        kind = 'bool' if isinstance(default, bool) else 'int' if isinstance(default, int) else 'float' if isinstance(default, float) else 'str'
    return dict(key=key, default=default, help=help, kind=kind if not choices else 'choice', choices=choices, lo=lo, hi=hi,
                label=label or key.replace('_', ' '), advanced=advanced)


def device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def plan_axes(ax, title=''):
    ax.set_aspect('equal'); ax.set_xlabel('east (m)' if title != 'model' else 'x'); ax.set_ylabel('north (m)'); ax.grid(alpha=.3)


def route_plot(ax, xy, c, label, cmap='turbo', vmin=None, vmax=None, s=4):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, s=s, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, fraction=0.035, label=label); return sc


def load_image(path, max_side=None):
    from PIL import Image
    im = Image.open(path)
    if max_side and max(im.size) > 2 * max_side and im.format in ('JPEG', 'MPO'):
        f = max(im.size) / max_side; im.draft('RGB', (int(im.size[0] / f * 1.01), int(im.size[1] / f * 1.01)))   # decode JPEG at reduced size: much faster
    im = im.convert('RGB')
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size); im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BILINEAR)
    return np.asarray(im)


def mask_for(project, name, shape_wh=None):
    """Mask (True = use pixel) for an image name, or None. Masks: <masks_dir>/<stem>.png or <name>.png, black = ignore."""
    md = project.masks_dir
    if not md: return None
    from PIL import Image
    stem = os.path.splitext(name)[0]
    for c in (stem + '.png', name + '.png', name, stem + '.jpg'):
        f = os.path.join(md, c)
        if os.path.exists(f):
            m = Image.open(f).convert('L')
            if shape_wh and m.size != tuple(shape_wh): m = m.resize(shape_wh, Image.NEAREST)
            return np.asarray(m) >= 128
    return None


_DET = {}


def detect(img_rgb, detector, long_side, max_kp, threshold=0.005):
    """Keypoints in full-resolution pixel coordinates. detector: sift (most precise) | aliked (most robust) | sift+aliked."""
    import torch, cv2
    H, W = img_rgb.shape[:2]; s = min(1.0, long_side / max(H, W)); out = []
    if 'sift' in detector:
        g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        if s < 1: g = cv2.resize(g, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
        key = ('sift', max_kp)
        if key not in _DET: _DET[key] = cv2.SIFT_create(nfeatures=max_kp, contrastThreshold=0.02)
        kps = _DET[key].detect(g, None)
        k = np.array([q.pt for q in kps], np.float32).reshape(-1, 2)
        if len(k):  # unique locations only (SIFT repeats a point for several orientations)
            k = np.unique(np.round(k * 100) / 100, axis=0)
        out.append((k + 0.5) / s - 0.5)
    if 'aliked' in detector:
        from lightglue import ALIKED
        key = ('aliked', max_kp, threshold)
        if key not in _DET: _DET[key] = ALIKED(max_num_keypoints=max_kp, detection_threshold=threshold).eval().to(device())
        t = torch.from_numpy(img_rgb.copy()).permute(2, 0, 1).float().div(255)[None].to(device())
        if s < 1: t = torch.nn.functional.interpolate(t, scale_factor=s, mode='area')
        with torch.no_grad(): out.append((_DET[key].extract(t)['keypoints'][0] / s).cpu().numpy())
    k = np.concatenate(out) if out else np.zeros((0, 2), np.float32)
    if len(out) > 1:   # merge: drop ALIKED points within 2 px of a SIFT point
        from scipy.spatial import cKDTree
        d, _ = cKDTree(out[0]).query(out[1]); k = np.concatenate([out[0], out[1][d > 2]])
    return k.astype(np.float32)


class FeatureStore:
    """Keypoints + FlexGlue descriptors per image (in features/ or another folder), with a small GPU cache."""

    def __init__(self, folder, dev=None, cache=400):
        self.d, self.dev, self.n, self.c = folder, dev or device(), cache, {}

    def kp(self, name):
        return np.load(os.path.join(self.d, name + '.kp.npy'))

    def get(self, name, kp_folder=None):
        import torch
        key = (name, kp_folder)
        if key not in self.c:
            if len(self.c) > self.n: self.c.pop(next(iter(self.c)))
            d = torch.load(os.path.join(self.d, name + '.pt'))
            kp = np.load(os.path.join(kp_folder or self.d, name + '.kp.npy'))
            self.c[key] = (kp, {'features': d['features'].to(self.dev).float(), 'keypoints': torch.from_numpy(kp).to(self.dev)[None],
                                'image_size': torch.tensor([d['size']], device=self.dev, dtype=torch.float32)})
        return self.c[key]


_FG = {}


def flexglue(describe=False):
    from flexglue import FlexGlue
    if describe not in _FG: _FG[describe] = FlexGlue(describe=describe).eval().to(device())
    return _FG[describe]


def fg_match(fs, a, b, kp_folder=None, snap_px=3.0):
    """FlexGlue sparse matching between two images -> (Nx2 index array, covisibility score).
    snap_px: FlexGlue predicts where a keypoint lands in the other image and takes the nearest detected keypoint within
    this distance. Small values (0.7-1 px) give precise, consistent tracks (sharp splats); large values give more matches."""
    import torch
    ka, fa = fs.get(a, kp_folder); kb, fb = fs.get(b, kp_folder)
    with torch.no_grad():
        p = flexglue(False).match(fa, fb, sparse=True, max_distance_px=snap_px)
    m0 = p['matches0'][0].cpu().numpy(); v = np.flatnonzero(m0 >= 0)
    cov = float(p['covisibility_score'][0]) if 'covisibility_score' in p else 1.0
    return np.c_[v, m0[v]].astype(np.uint32), cov


def magsac(ka, kb, m, thr=4.0):
    import cv2
    if len(m) < 8: return m[:0]
    _, msk = cv2.findFundamentalMat(ka[m[:, 0]].astype(np.float64), kb[m[:, 1]].astype(np.float64), cv2.USAC_MAGSAC, thr, 0.999, 10000)
    return m[:0] if msk is None else m[msk.ravel().astype(bool)]


def draw_matches(ax, ia, ib, ka, kb, m, color, n=300, seed=0):
    from matplotlib.collections import LineCollection
    H = ia.shape[0]; ax.imshow(np.vstack([ia, ib])); ax.axis('off')
    if len(m):
        s = m[np.random.default_rng(seed).choice(len(m), min(n, len(m)), replace=False)]
        seg = np.stack([ka[s[:, 0]], kb[s[:, 1]] + [0, H]], 1)
        ax.add_collection(LineCollection(seg, colors=color, lw=0.6, alpha=.8)); ax.scatter(seg[:, :, 0].ravel(), seg[:, :, 1].ravel(), s=2, c=color)


def verbose():
    """True when the user asked to see everything (GUI 'verbose' switch or FLEXSFM_VERBOSE=1)."""
    return os.environ.get('FLEXSFM_VERBOSE') == '1'


def ba_logging(opt):
    """In verbose mode let Ceres print every bundle-adjustment iteration (cost, step size, time) to the log."""
    if verbose():
        opt.ceres.solver_options.minimizer_progress_to_stdout = True
        try: opt.print_summary = True
        except Exception: pass
    return opt


# ---------------------------------------------------------------- live previews (what a step is looking at right now)
_HUES = None


def _bgr(img_rgb, max_side=1280):
    import cv2
    h, w = img_rgb.shape[:2]; s = min(1.0, max_side / max(h, w))
    out = cv2.resize(img_rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img_rgb.copy()
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR), s


def live_keypoints(img_rgb, kp, mask=None, max_side=1280):
    """Image with its keypoints (magenta dots); masked-out pixels shaded red."""
    import cv2
    out, s = _bgr(img_rgb, max_side)
    if mask is not None:
        mk = cv2.resize(mask.astype(np.uint8), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST) == 0
        out[mk] = (out[mk] * 0.45 + np.array([40, 40, 200]) * 0.55).astype(np.uint8)
    kp = np.asarray(kp).reshape(-1, 2)
    if len(kp) > 6000: kp = kp[np.random.default_rng(0).choice(len(kp), 6000, replace=False)]   # a readable sample; the caption gives the real count
    for x, y in (kp * s).astype(int):
        cv2.circle(out, (int(x), int(y)), 3, (40, 20, 40), -1, cv2.LINE_AA); cv2.circle(out, (int(x), int(y)), 2, (255, 60, 255), -1, cv2.LINE_AA)   # magenta: visible on foliage and grass
    return out


def live_pair(img_a, img_b, ka, kb, m, max_side=1100, n=250, rejected=None):
    """Two images side by side with match lines (colour = position across the image); rejected matches in faint red."""
    import cv2
    a, s = _bgr(img_a, max_side); b, s2 = _bgr(img_b, max_side)
    H = max(a.shape[0], b.shape[0]); out = np.zeros((H, a.shape[1] + b.shape[1] + 8, 3), np.uint8); out[:a.shape[0], :a.shape[1]] = a
    ox = a.shape[1] + 8; out[:b.shape[0], ox:] = b
    rng = np.random.default_rng(0)
    if rejected is not None and len(rejected):
        for u, v in rejected[rng.choice(len(rejected), min(len(rejected), n // 3), replace=False)]:
            p1 = (int(ka[u][0] * s), int(ka[u][1] * s)); p2 = (int(kb[v][0] * s2) + ox, int(kb[v][1] * s2))
            cv2.line(out, p1, p2, (70, 70, 230), 1, cv2.LINE_AA)
    if m is not None and len(m):
        sel = m[rng.choice(len(m), min(len(m), n), replace=False)]
        W = max(1, img_a.shape[1])
        for u, v in sel:
            hue = int(170 * ka[u][0] / W); col = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
            p1 = (int(ka[u][0] * s), int(ka[u][1] * s)); p2 = (int(kb[v][0] * s2) + ox, int(kb[v][1] * s2))
            cv2.line(out, p1, p2, col, 1, cv2.LINE_AA); cv2.circle(out, p1, 2, col, -1, cv2.LINE_AA); cv2.circle(out, p2, 2, col, -1, cv2.LINE_AA)
    return out


def live_topview(xy_points, xy_cams, size=1000):
    """Plan view: points (grey) and camera positions (orange path)."""
    import cv2
    out = np.full((size, size, 3), 255, np.uint8); P = np.asarray(xy_points).reshape(-1, 2); C = np.asarray(xy_cams).reshape(-1, 2)
    allp = np.vstack([P, C]) if len(P) else C
    if not len(allp): return out
    lo, hi = np.percentile(allp, 1, 0), np.percentile(allp, 99, 0); span = max(float((hi - lo).max()), 1e-6) * 1.1; c = (lo + hi) / 2
    f = lambda X: (((X - c) / span + 0.5) * np.array([1, -1]) * size + np.array([0, size])).astype(int)
    for x, y in f(P[:: max(1, len(P) // 200000)]):
        if 0 <= x < size and 0 <= y < size: out[y, x] = (110, 110, 110)
    pc = f(C)
    for q0, q1 in zip(pc[:-1], pc[1:]): cv2.line(out, tuple(map(int, q0)), tuple(map(int, q1)), (0, 140, 255), 2, cv2.LINE_AA)
    return out
