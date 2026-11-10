"""Revisit analysis shared by diagnose and repair: images that look at the same place from different passes /
far apart in time, how strongly the model ties them together, and how well real matches agree with the poses."""
import os
import numpy as np
from . import geom
from .common import FeatureStore, fg_match, magsac


def groups_for(names, g, gap):
    """Group id per frame: GPS pass if available, else blocks of `gap` frames (revisit = different group)."""
    if g is not None and 'pass_id' in g and int(g['pass_id'].max()) > 0: return np.asarray(g['pass_id']), 'pass'
    return np.arange(len(names)) // max(1, gap), 'time'


def candidate_pairs(rec, names, grp, foot_dist=4.0, k=4, min_angle=0.0, max_shared=None):
    """Pairs of images from different groups whose footprints (where they look) are within foot_dist.
    Returns list of (image_id_a, image_id_b, footprint distance, view angle deg, shared 3D points)."""
    idx = {n: i for i, n in enumerate(names)}
    fp = geom.footprints(rec); ids = [i for i in fp if rec.images[i].name in idx]
    F = np.array([fp[i][0][:2] for i in ids]); L = np.array([fp[i][1] for i in ids]); Gp = np.array([grp[idx[rec.images[i].name]] for i in ids])
    from scipy.spatial import cKDTree
    tr = cKDTree(F); out = {}
    pts = {i: {p.point3D_id for p in rec.images[i].points2D if p.has_point3D()} for i in ids}
    for a, i in enumerate(ids):
        c = [b for b in tr.query_ball_point(F[a], foot_dist) if Gp[b] != Gp[a]]
        c.sort(key=lambda b: np.linalg.norm(F[b] - F[a]))
        took = 0
        for b in c:
            ang = float(np.degrees(np.arccos(np.clip(L[a] @ L[b], -1, 1))))
            if ang < min_angle: continue
            key = tuple(sorted((i, ids[b])))
            if key not in out: out[key] = (key[0], key[1], float(np.linalg.norm(F[b] - F[a])), ang, len(pts[i] & pts[ids[b]]))
            took += 1
            if took >= k: break
    res = list(out.values())
    if max_shared is not None: res = [r for r in res if r[4] <= max_shared]
    return res, fp


def match_and_check(p, rec, pairs, ctx=None, ransac_px=4.0, min_inliers=25, gate_px=None):
    """FlexGlue + MAGSAC on image pairs (rec image ids) using the project's features. Returns {pair: (matches, epipolar px vs current poses)}."""
    fs = FeatureStore(p.path('features')); out = {}
    for k, (a, b) in enumerate(pairs):
        na, nb = rec.images[a].name, rec.images[b].name
        m, _ = fg_match(fs, na, nb, snap_px=1.5)
        if len(m) < min_inliers: continue
        ka, kb = fs.kp(na), fs.kp(nb); m = magsac(ka, kb, m, ransac_px)
        if len(m) < min_inliers: continue
        e = geom.epipolar_px(rec, a, b, ka[m[:, 0]], kb[m[:, 1]])
        if gate_px: keep = e < gate_px; m, e = m[keep], e[keep]
        if len(m) >= min_inliers: out[(a, b)] = (m, e)
        if ctx and ctx.live_due(1.5):
            def _mk(na=na, nb=nb, ka=ka, kb=kb, m=m.copy()):
                from .common import live_pair, load_image
                from PIL import Image
                ia, ib = load_image(os.path.join(p.images_dir, na), 1100), load_image(os.path.join(p.images_dir, nb), 1100)
                sa = ia.shape[1] / Image.open(os.path.join(p.images_dir, na)).size[0]; sb = ib.shape[1] / Image.open(os.path.join(p.images_dir, nb)).size[0]
                return live_pair(ia, ib, ka * sa, kb * sb, m)
            ctx.live_async(_mk, f'revisit pair {k + 1} / {len(pairs)}: {na} <-> {nb} - {len(m)} matches'
                     + (f', median disagreement with current poses {np.median(e):.1f} px' if len(e) else ''))
        if ctx and k % 200 == 0: ctx.progress(k / max(1, len(pairs)), f'matching revisit pairs {k}/{len(pairs)}')
    return out
