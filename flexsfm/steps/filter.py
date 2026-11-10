"""Step 10: choose which dense points to keep."""
import os
import numpy as np
from ..common import P, plt
from .. import geom

TITLE = '10  Filter points'
WHAT = """Removes unreliable points from the dense cloud. Fast - re-run with other values as often as you like.
- max error: points whose reprojection error is larger are dropped (lower = cleaner, fewer).
- min angle: points seen from almost the same direction have poor depth (lower = more points, noisier depth).
- min track: number of images that must see a point.
The table in the report shows how many points each combination keeps, so you can aim at a count (e.g. like Metashape)."""
NEEDS = ['densify']
PARAMS = [
    P('max_error_px', 2.0, 'Drop points with a larger reprojection error', lo=0.3, hi=10.0),
    P('min_angle_deg', 1.0, 'Drop points with a smaller triangulation angle', lo=0.0, hi=20.0),
    P('min_track', 2, 'Drop points seen by fewer images', lo=2, hi=50),
    P('target_points', 0, 'Optional: point count you aim for (0 = none). Only reported, nothing is changed.', lo=0),
]


def run(p, prm, ctx):
    import pycolmap
    rec = pycolmap.Reconstruction(p.model_dir('dense_raw')); n0 = rec.num_points3D(); ctx.log(f'{n0} points in dense_raw')
    ids = list(rec.points3D); e = np.empty(len(ids)); a = np.empty(len(ids)); t = np.empty(len(ids), int)
    for k, pid in enumerate(ids):
        pt = rec.points3D[pid]; e[k] = pt.error; t[k] = pt.track.length(); a[k] = geom.tri_angle_deg(rec, pt)
        if k % 200000 == 0: ctx.progress(0.7 * k / len(ids), f'measuring points {k}/{len(ids)}')
    keep = (e <= prm['max_error_px']) & (a >= prm['min_angle_deg']) & (t >= prm['min_track'])
    for pid in np.array(ids)[~keep]: rec.delete_point3D(int(pid))
    out = p.model_dir('dense'); os.makedirs(out, exist_ok=True); rec.write(out)
    ctx.metric('points_before', n0); ctx.metric('points_kept', int(keep.sum())); ctx.metric('mean_reprojection_px', round(rec.compute_mean_reprojection_error(), 3))
    ctx.metric('mean_track', round(rec.compute_mean_track_length(), 2))
    rows = []
    for me in (1.0, 1.5, 2.0, 3.0, 4.0):
        for ma in (0.5, 1.0, 2.0):
            for mt in (2, 3):
                rows.append((me, ma, mt, int(((e <= me) & (a >= ma) & (t >= mt)).sum())))
    ctx.metric('options_table', [dict(max_error=r[0], min_angle=r[1], min_track=r[2], points=r[3]) for r in rows])
    ctx.log('points kept for other settings (max error / min angle / min track -> points):')
    for r in rows: ctx.log(f'  {r[0]:>4} px  {r[1]:>4} deg  {r[2]}  -> {r[3]:,}')
    if prm['target_points']:
        best = min(rows, key=lambda r: abs(r[3] - prm['target_points']))
        ctx.check('point count near target', abs(keep.sum() - prm['target_points']) < 0.25 * prm['target_points'], f'{keep.sum():,}', f'~{prm["target_points"]:,}',
                  f'Closest setting: max error {best[0]} px, min angle {best[1]}, min track {best[2]} -> {best[3]:,} points. '
                  'For more points beyond that: raise "max keypoints" in step 9.', level=None if abs(keep.sum() - prm['target_points']) < 0.25 * prm['target_points'] else 'warn')
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.2))
    ax[0].hist(np.clip(e, 0, 6), bins=60); ax[0].axvline(prm['max_error_px'], c='r'); ax[0].set_title('reprojection error (px)')
    ax[1].hist(np.clip(a, 0, 30), bins=60); ax[1].axvline(prm['min_angle_deg'], c='r'); ax[1].set_title('triangulation angle (deg)')
    ax[2].hist(np.clip(t, 2, 40), bins=np.arange(2, 42)); ax[2].axvline(prm['min_track'], c='r'); ax[2].set_title('track length')
    ctx.fig(fig, 'filter_hist', 'Distributions with the chosen limits (red)')
    X = np.array([pt.xyz for pt in rec.points3D.values()])
    if len(X):
        if len(X) > 500000: X = X[np.random.default_rng(0).choice(len(X), 500000, replace=False)]
        fig, ax = plt.subplots(figsize=(14, 9)); lo, hi = np.percentile(X[:, :2], 1, 0), np.percentile(X[:, :2], 99, 0)
        ax.hexbin(X[:, 0], X[:, 1], gridsize=400, bins='log', cmap='viridis', extent=(lo[0], hi[0], lo[1], hi[1])); ax.set_aspect('equal')
        ax.set_title(f'{keep.sum():,} kept points from above'); ctx.fig(fig, 'kept_topview', 'Kept points from above')
