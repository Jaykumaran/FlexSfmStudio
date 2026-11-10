"""Step 11: export for Gaussian-splat training (undistorted COLMAP), PLY, camera table, comparison."""
import json, os, shutil
import numpy as np
from ..common import P, plt, mask_for
from .. import geom, gps as G

TITLE = '11  Export'
WHAT = """Writes everything a Gaussian-splat trainer needs, plus files for checking:
  export/undistorted/images/ + sparse/0/   pinhole camera (distortion removed) - point your trainer here
  export/undistorted/masks/                 masks undistorted the same way (if you have masks)
  export/points.ply                         the tie points (coloured)
  export/cameras.csv                        every camera: name, position (and lat/lon/alt with GPS), error
  export/georef.json                        how local metres relate to lat/lon/alt
Optionally compares the cameras with another model (e.g. Metashape's COLMAP export of the same frames)."""
NEEDS = ['reconstruct']
PARAMS = [
    P('model', 'auto', 'Which model to export: auto = dense (filtered) if it exists, else the current one',
      choices=['auto', 'dense', 'dense_raw', 'current']),
    P('undistort', True, 'Write undistorted images + pinhole model (needed by most 3DGS trainers)'),
    P('jpeg_quality', 95, 'JPEG quality of undistorted images', lo=50, hi=100),
    P('compare_with', '', 'Optional: another COLMAP model folder (e.g. Metashape export sparse/0) to compare cameras with', kind='dir'),
]


def _undistort_masks(cam_d, cam_u, names, p, out):
    import cv2
    os.makedirs(out, exist_ok=True)
    W, H = cam_u.width, cam_u.height; u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = np.asarray(cam_u.cam_from_img(np.c_[u.ravel(), v.ravel()]), float)
    src = np.asarray(cam_d.img_from_cam(np.c_[rays, np.ones(len(rays))]), float).reshape(H, W, -1)[..., :2] - 0.5
    mx, my = src[..., 0].astype(np.float32), src[..., 1].astype(np.float32); n = 0
    for nm in names:
        m = mask_for(p, nm)
        if m is None: continue
        mu = cv2.remap((m * 255).astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderValue=0)
        cv2.imwrite(os.path.join(out, os.path.splitext(nm)[0] + '.png'), mu); n += 1
    return n


def run(p, prm, ctx):
    import pycolmap
    names = p.names(); g = p.gps()
    pick = prm['model']
    if pick == 'auto': pick = 'dense' if os.path.exists(os.path.join(p.model_dir('dense'), 'images.bin')) else 'current'
    path = p.current_model if pick == 'current' else p.model_dir(pick)
    rec = pycolmap.Reconstruction(path); ctx.log(f'exporting {path}: {rec.num_reg_images()} cameras, {rec.num_points3D()} points')
    ex = p.path('export'); os.makedirs(ex, exist_ok=True)
    rec.export_PLY(os.path.join(ex, 'points.ply'))
    ref = g['ref'] if g is not None else None
    rows = []
    for im in sorted(rec.images.values(), key=lambda x: x.name):
        c = im.projection_center(); e = [rec.points3D[q.point3D_id].error for q in im.points2D if q.has_point3D()]
        r = dict(name=im.name, x=c[0], y=c[1], z=c[2], points=len(e), error_px=float(np.mean(e)) if e else np.nan)
        if ref and 'lat0' in ref:
            la = G.enu_to_lla(np.array([c]), ref)[0]; r.update(lat=la[0], lon=la[1], alt=la[2])
        rows.append(r)
    import csv
    with open(os.path.join(ex, 'cameras.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
    json.dump(dict(frame='local east/north/up metres' if ref else 'arbitrary (no GPS)', enu_origin=ref, model=pick, source=path), open(os.path.join(ex, 'georef.json'), 'w'), indent=1, default=float)
    ctx.metric('exported_model', pick); ctx.metric('cameras', rec.num_reg_images()); ctx.metric('points', rec.num_points3D())
    if prm['undistort']:
        ud = os.path.join(ex, 'undistorted')
        if os.path.exists(ud): shutil.rmtree(ud)
        ctx.progress(0.2, 'undistorting images'); ctx.log('undistorting images (CPU) ...')
        uo = pycolmap.UndistortCameraOptions()
        pycolmap.undistort_images(ud, path, p.images_dir, output_type='COLMAP', undistort_options=uo, jpeg_quality=prm['jpeg_quality'])
        sp = os.path.join(ud, 'sparse'); os.makedirs(os.path.join(sp, '0'), exist_ok=True)
        for f in os.listdir(sp):
            if f.endswith('.bin'): shutil.move(os.path.join(sp, f), os.path.join(sp, '0', f))
        ru = pycolmap.Reconstruction(os.path.join(sp, '0')); cu = ru.cameras[next(iter(ru.cameras))]
        ctx.metric('undistorted_camera', f'{cu.model.name} {cu.width}x{cu.height} f={cu.params[0]:.2f}')
        nimg = len([f for f in os.listdir(os.path.join(ud, 'images'))])
        ctx.check('all images undistorted', nimg == rec.num_reg_images(), f'{nimg}', f'{rec.num_reg_images()}', 'Some images failed to undistort - see the log.')
        ctx.check('pinhole camera for training', cu.model.name in ('PINHOLE', 'SIMPLE_PINHOLE'), cu.model.name, 'PINHOLE', 'Unexpected camera model after undistortion.')
        if p.masks_dir:
            ctx.progress(0.8, 'undistorting masks')
            n = _undistort_masks(rec.cameras[next(iter(rec.cameras))], cu, [im.name for im in rec.images.values()], p, os.path.join(ud, 'masks'))
            ctx.metric('masks_undistorted', n)
        ctx.log(f'training folder: {ud}  (images/ + sparse/0/)')
    if prm['compare_with']:
        ref_rec = pycolmap.Reconstruction(os.path.expanduser(prm['compare_with']))
        a = {im.name: im.projection_center() for im in rec.images.values()}; b = {im.name: im.projection_center() for im in ref_rec.images.values()}
        common = sorted(set(a) & set(b)); A = np.array([a[n] for n in common]); B = np.array([b[n] for n in common])
        s, R, t = geom.umeyama(B, A); Bm = (B @ R.T) * s + t; d = np.linalg.norm(A - Bm, axis=1)
        ctx.metric('compare', dict(common=len(common), other_points=ref_rec.num_points3D(), centre_diff_median_m=round(float(np.median(d)), 3),
                                   centre_diff_p90_m=round(float(np.percentile(d, 90)), 3), centre_diff_max_m=round(float(d.max()), 3)))
        idx = {n: i for i, n in enumerate(names)}
        fig, ax = plt.subplots(figsize=(16, 4)); ax.plot([idx[n] for n in common], d); ax.set_xlabel('frame'); ax.set_ylabel('m')
        ax.set_title('camera difference to the other model (after best similarity fit)'); ax.grid(alpha=.3); ctx.fig(fig, 'compare', 'Comparison with the other model')
