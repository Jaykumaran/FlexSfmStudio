"""Step 6 (optional): markers / ground control points."""
import json, os
import numpy as np
from ..common import P, plt, load_image
from .. import geom, gps as G

TITLE = '6  Markers (optional)'
WHAT = """Uses points you know on the ground (survey markers / GCPs) to fix the georeference, like Metashape markers.
1. Write the marker coordinates into a CSV: name, lat, lon, alt (or name, x, y, z in metres), optional 'type'
   column with control or check. Check markers are not used for the fit, only to measure accuracy.
2. In the 'Markers' tab, pick a marker and click its centre in at least 2 images (more is better, from different passes).
3. Run this step: each marker is triangulated from your clicks, the model is fitted onto the control markers
   (similarity: shift, rotation, scale) and the error at every marker is reported.
Without GPS and without coordinates you can still fix the scale: give two marker names and their distance."""
NEEDS = ['reconstruct']
PARAMS = [
    P('markers_csv', '', 'CSV with marker coordinates (name, lat, lon, alt / name, x, y, z, optional type)', kind='file'),
    P('scale_a', '', 'Scale only: first marker name'),
    P('scale_b', '', 'Scale only: second marker name'),
    P('scale_distance_m', 0.0, 'Scale only: true distance between the two markers (m)', lo=0.0),
    P('min_clicks', 2, 'Markers with fewer image clicks are skipped', lo=2),
]


def read_markers(path, ref):
    import csv
    rows = list(csv.reader([l for l in open(path) if l.strip() and not l.startswith('#')]))
    hdr = [h.strip().lower() for h in rows[0]]; has_hdr = not any(c.replace('.', '').replace('-', '').isdigit() for c in rows[0][1:3])
    body = rows[1:] if has_hdr else rows
    def ci(*o):
        for k, h in enumerate(hdr if has_hdr else []):
            if any(h.startswith(x) for x in o): return k
    la, lo, al = ci('lat'), ci('lon', 'lng'), ci('alt', 'height', 'z')
    x, y, ty = ci('x', 'east'), ci('y', 'north'), ci('type', 'use')
    out = {}
    for r in body:
        n = r[0].strip()
        if la is not None and lo is not None:
            v = np.array([[float(r[la]), float(r[lo]), float(r[al]) if al is not None else 0.0]]); geo = True
        else:
            xi, yi, zi = (x, y, al) if x is not None else (1, 2, 3)
            v = np.array([[float(r[xi]), float(r[yi]), float(r[zi])]]); geo = False
        out[n] = dict(raw=v[0], geo=geo, type=(r[ty].strip().lower() if ty is not None and len(r) > ty else 'control'))
    if out and next(iter(out.values()))['geo']:
        lla = np.array([m['raw'] for m in out.values()])
        enu, ref2 = G.lla_to_enu(lla, ref if ref and 'lat0' in ref else None)
        for m, e in zip(out.values(), enu): m['xyz'] = e
        return out, ref2
    for m in out.values(): m['xyz'] = m['raw']
    return out, ref


def run(p, prm, ctx):
    import pycolmap
    rec = pycolmap.Reconstruction(p.current_model); obs = json.load(open(p.path('markers_obs.json'))) if os.path.exists(p.path('markers_obs.json')) else {}
    if not obs: raise RuntimeError('no marker clicks yet: open the "Markers" tab, pick a marker, click it in 2+ images, then run again')
    name2id = {im.name: i for i, im in rec.images.items()}
    tri = {}
    for m, clicks in obs.items():
        o = [(name2id[n], xy) for n, xy in clicks.items() if n in name2id]
        if len(o) < prm['min_clicks']: ctx.log(f'marker {m}: {len(o)} clicks - skipped'); continue
        X, err = geom.triangulate_track(rec, o)
        if X is None: ctx.log(f'marker {m}: triangulation failed (behind a camera?) - check the clicks'); continue
        tri[m] = dict(xyz=X, max_click_err_px=err, n=len(o))
        ctx.check(f'marker {m} clicks consistent', err < 5, f'{err:.1f} px worst click', '< 5 px',
                  f'One click of {m} is off: re-click it in the Markers tab (the worst image is listed in the log).', level=None if err < 5 else 'warn')
    ctx.metric('markers_triangulated', len(tri))
    g = p.gps(); ref = g['ref'] if g is not None else None
    res = {}
    if prm['markers_csv']:
        mk, _ = read_markers(os.path.expanduser(prm['markers_csv']), ref)
        ctrl = [m for m in tri if m in mk and mk[m]['type'].startswith('control')]
        chk = [m for m in tri if m in mk and mk[m]['type'].startswith('check')]
        ctx.metric('control_markers', ctrl); ctx.metric('check_markers', chk)
        if len(ctrl) < 3: raise RuntimeError(f'need >= 3 control markers with clicks, have {len(ctrl)}: {ctrl}')
        A = np.array([tri[m]['xyz'] for m in ctrl]); B = np.array([mk[m]['xyz'] for m in ctrl])
        s, R, t = geom.umeyama(A, B); rec.transform(geom.sim3(s, R, t))
        ctx.metric('fit_scale', round(float(s), 6))
        for m in tri:
            tri[m]['xyz'] = (tri[m]['xyz'] @ R.T) * s + t
            if m in mk: res[m] = dict(type=mk[m]['type'], err=(tri[m]['xyz'] - mk[m]['xyz']).round(3).tolist(), dist=round(float(np.linalg.norm(tri[m]['xyz'] - mk[m]['xyz'])), 3))
        ec = np.array([res[m]['dist'] for m in ctrl]); ctx.metric('control_rms_m', round(float(np.sqrt((ec ** 2).mean())), 3))
        ctx.check('control markers fit', ec.max() < 0.3, f'worst {ec.max():.3f} m', '< 0.3 m',
                  'A marker fits badly: wrong coordinate, wrong click, or the model is bent (then use GPS bundle adjustment / repair).',
                  level=None if ec.max() < 0.3 else 'warn')
        if chk:
            ek = np.array([res[m]['dist'] for m in chk]); ctx.metric('check_rms_m', round(float(np.sqrt((ek ** 2).mean())), 3))
            ctx.check('check markers (independent accuracy)', ek.max() < 0.5, f'worst {ek.max():.3f} m, RMS {np.sqrt((ek ** 2).mean()):.3f} m', '< 0.5 m',
                      'Check markers are off: the model is not uniformly accurate - see diagnose.', level=None if ek.max() < 0.5 else 'warn')
    elif prm['scale_a'] and prm['scale_b'] and prm['scale_distance_m'] > 0:
        a, b = prm['scale_a'], prm['scale_b']
        if a not in tri or b not in tri: raise RuntimeError(f'scale markers {a}, {b} need clicks')
        d = np.linalg.norm(tri[a]['xyz'] - tri[b]['xyz']); s = prm['scale_distance_m'] / d
        rec.transform(geom.sim3(s, np.eye(3), np.zeros(3))); ctx.metric('scale_applied', round(float(s), 6))
        ctx.log(f'scaled the model by {s:.5f} ({d:.3f} -> {prm["scale_distance_m"]} m)')
    else:
        raise RuntimeError('give a markers CSV, or two marker names + their distance for scale only')
    d = p.model_dir('markers'); os.makedirs(d, exist_ok=True); rec.write(d); p.set_model('markers')
    json.dump({m: dict(v, xyz=np.asarray(v['xyz']).tolist()) for m, v in tri.items()}, open(p.path('markers_result.json'), 'w'), indent=1, default=float)
    if res:
        fig, ax = plt.subplots(figsize=(max(6, len(res) * 0.8), 4))
        ks = sorted(res); ax.bar(ks, [res[k]['dist'] for k in ks], color=['#1f77b4' if res[k]['type'].startswith('control') else '#ff7f0e' for k in ks])
        ax.set_ylabel('error (m)'); ax.set_title('marker error after the fit (blue = control, orange = check)'); plt.xticks(rotation=60)
        ctx.fig(fig, 'marker_errors', 'Distance between each triangulated marker and its coordinate')
    name2im = {im.name: im for im in rec.images.values()}
    for m, clicks in list(obs.items())[:12]:
        if m not in tri: continue
        ims = [n for n in clicks if n in name2im][:4]
        fig, ax = plt.subplots(1, len(ims), figsize=(4 * len(ims), 4))
        for a, n in zip(np.atleast_1d(ax), ims):
            img = load_image(os.path.join(p.images_dir, n)); x, y = clicks[n]; r = 120
            a.imshow(img[max(0, int(y) - r):int(y) + r, max(0, int(x) - r):int(x) + r]); a.plot(min(r, x), min(r, y), '+', c='lime', ms=20, mew=2)
            q = name2im[n].project_point(tri[m]['xyz'])
            if q is not None: a.plot(q[0] - max(0, int(x) - r), q[1] - max(0, int(y) - r), 'x', c='red', ms=14, mew=2)
            a.set_title(n, fontsize=8); a.axis('off')
        ctx.fig(fig, f'marker_{m}', f'Marker {m}: your click (green +) and the triangulated marker projected back (red x)')
