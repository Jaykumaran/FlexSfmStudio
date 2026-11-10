"""Step 9: many more tie points at full resolution, triangulated with the solved cameras, then refined."""
import os, shutil, time
import numpy as np
from ..common import P, plt, FeatureStore, fg_match, detect, load_image, mask_for, ba_logging, live_keypoints, live_pair
from .. import modelstats as MS

TITLE = '9  Densify + refine'
WHAT = """Builds the dense tie-point cloud used to start Gaussian-splat training (Metashape 'tie points' equivalent).
1. Detects many more keypoints at full resolution.
2. Re-matches the pairs that were verified in step 4 (+ revisit pairs from repair) with FlexGlue.
3. Triangulates them with the solved cameras.
4. Refine: bundle adjustment of cameras + lens against these precise full-resolution points (with GPS priors if any),
   then triangulates again. This is what makes edges sharp: cameras that agree to a fraction of a pixel.
   The adjustment uses a subset of the best points to keep the CPU time reasonable.
Then use step 10 (filter) to choose how many points to keep."""
NEEDS = ['reconstruct']
PARAMS = [
    P('detector', 'sift', 'Keypoint detector for the dense points (sift = most precise)', choices=['sift', 'aliked', 'sift+aliked']),
    P('max_keypoints', 16000, 'Keypoints per image (more = more points; matching slower)', lo=1000, hi=40000),
    P('min_pair_inliers', 30, 'Only re-match pairs that had at least this many verified inliers in step 4', lo=8),
    P('snap_px', 0.7, 'Match precision (px): smaller = more precise, consistent tracks = sharper splats. 0.7 measured best on 4K video.', lo=0.3, hi=5.0),
    P('ransac_px', 1.0, 'Geometric verification tolerance (px)', lo=0.5, hi=20.0),
    P('track_px', 1.5, 'Max error (px) for adding an observation to a track or merging tracks (COLMAP default 4 - too loose)', lo=0.5, hi=8.0),
    P('track_angle_deg', 0.5, 'Max ray-angle error (deg) when starting/continuing a track (COLMAP default 2)', lo=0.05, hi=5.0, advanced=True),
    P('min_angle_deg', 1.0, 'Minimum triangulation angle when creating points', lo=0.1, hi=10.0),
    P('refine', True, 'Bundle-adjust cameras + lens on the dense points, then re-triangulate (recommended; CPU, slow)'),
    P('refine_points', 600000, 'Points used in the refinement adjustment (best tracks first)', lo=50000, hi=5000000),
    P('refine_principal_point', True, 'Also refine the image centre (principal point)'),
    P('ba_iterations', 60, 'Max iterations of the refinement adjustment', lo=5, hi=500),
    P('gps_sigma_h', 3.0, 'GPS horizontal uncertainty in the refinement (m, loose: GPS only prevents drift)', lo=0.1, hi=100.0),
    P('gps_sigma_v', 3.0, 'GPS vertical uncertainty in the refinement (m)', lo=0.1, hi=100.0),
]


def _triangulate(rec, db, images, out, prm):
    import pycolmap
    po = pycolmap.IncrementalPipelineOptions(); t = po.triangulation; t.ignore_two_view_tracks = False; t.min_angle = prm['min_angle_deg']
    t.merge_max_reproj_error = t.complete_max_reproj_error = prm['track_px']; t.create_max_angle_error = t.continue_max_angle_error = prm['track_angle_deg']
    if os.path.exists(out): shutil.rmtree(out)
    os.makedirs(out)
    return pycolmap.triangulate_points(rec, db, images, out, clear_points=True, options=po, refine_intrinsics=False)


def run(p, prm, ctx):
    import pycolmap, torch
    names = p.names(); idx = {n: i for i, n in enumerate(names)}; g = p.gps()
    rec = pycolmap.Reconstruction(p.current_model); src = p.state['model']; ctx.log(f'cameras from models/{src}')
    fh = p.path('features_dense'); os.makedirs(fh, exist_ok=True)
    tag = f"{prm['detector']}_{prm['max_keypoints']}"; tf = os.path.join(fh, 'settings.txt')
    redo = os.path.exists(tf) and open(tf).read().strip() != tag
    todo = [n for n in names if redo or not os.path.exists(os.path.join(fh, n + '.kp.npy'))]
    for k, n in enumerate(todo):
        img = load_image(os.path.join(p.images_dir, n)); H, W = img.shape[:2]
        kp = detect(img, prm['detector'], max(H, W), prm['max_keypoints'])
        m = mask_for(p, n, (W, H))
        if m is not None: ix = np.clip(kp.astype(int), 0, [W - 1, H - 1]); kp = kp[m[ix[:, 1], ix[:, 0]]]
        np.save(os.path.join(fh, n + '.kp.npy'), kp.astype(np.float32))
        if ctx.live_due(): ctx.live_async(lambda img=img, kp=kp, m=m: live_keypoints(img, kp, m), f'dense keypoints, image {k + 1} / {len(todo)}: {n} - {len(kp)} keypoints at full resolution')
        if k % 100 == 0: ctx.progress(0.2 * k / len(todo), f'dense keypoints {k}/{len(todo)}'); ctx.log(f'{k}/{len(todo)} {n}: {len(kp)} keypoints')
    open(tf, 'w').write(tag)
    # pairs: verified in step 4 + repair pairs
    pi = np.load(p.path('pair_inliers.npz')); pairs = {tuple(x) for x, n in zip(pi['pairs'], pi['n']) if n >= prm['min_pair_inliers']}
    if os.path.exists(p.path('repair_pairs.txt')) and src == 'repaired':
        for l in open(p.path('repair_pairs.txt')):
            a, b = l.split()[:2]
            if a in idx and b in idx: pairs.add(tuple(sorted((idx[a], idx[b]))))
    pairs = sorted(pairs); ctx.log(f'{len(pairs)} pairs to re-match')
    fs = FeatureStore(p.path('features')); res = {}; t1 = time.time()
    for k, (i, j) in enumerate(pairs):
        m, _ = fg_match(fs, names[i], names[j], kp_folder=fh, snap_px=prm['snap_px'])
        if len(m) >= 15: res[(i, j)] = m
        if ctx.live_due(1.5):
            def _mk(na_=names[i], nb_=names[j], m_=m.copy()):
                from PIL import Image
                ia, ib = load_image(os.path.join(p.images_dir, na_), 1100), load_image(os.path.join(p.images_dir, nb_), 1100)
                sa = ia.shape[1] / Image.open(os.path.join(p.images_dir, na_)).size[0]; sb = ib.shape[1] / Image.open(os.path.join(p.images_dir, nb_)).size[0]
                return live_pair(ia, ib, np.load(os.path.join(fh, na_ + '.kp.npy')) * sa, np.load(os.path.join(fh, nb_ + '.kp.npy')) * sb, m_)
            ctx.live_async(_mk, f'dense matching, pair {k + 1} / {len(pairs)}: {names[i]} <-> {names[j]} - {len(m)} precise matches')
        if k % 1000 == 0:
            r = (k + 1) / (time.time() - t1); ctx.progress(0.2 + 0.4 * k / len(pairs), f'dense matching {k}/{len(pairs)}')
            ctx.log(f'matched {k}/{len(pairs)} ({r:.1f} pairs/s, ~{(len(pairs) - k) / r / 60:.0f} min left)')
    # database with the solved camera
    dd = p.path('dense'); os.makedirs(dd, exist_ok=True); db = os.path.join(dd, 'database.db')
    for f in (db, db + '-wal', db + '-shm'):
        if os.path.exists(f): os.remove(f)
    cam = rec.cameras[next(iter(rec.cameras))]
    opt = pycolmap.ImageReaderOptions(); opt.camera_model = cam.model.name; opt.camera_params = ','.join(f'{v:.10g}' for v in cam.params)
    pycolmap.Database.open(db).close()
    pycolmap.import_images(db, p.images_dir, camera_mode=pycolmap.CameraMode.SINGLE, image_names=names, options=opt)
    with pycolmap.Database.open(db) as d:
        ids = {im.name: im.image_id for im in d.read_all_images()}
        for n in names: d.write_keypoints(ids[n], np.load(os.path.join(fh, n + '.kp.npy')) + 0.5)
        for (i, j), m in res.items(): d.write_matches(ids[names[i]], ids[names[j]], m)
    pf = os.path.join(dd, 'pairs.txt'); open(pf, 'w').writelines(f'{names[i]} {names[j]}\n' for i, j in res)
    ctx.progress(0.62, 'verification'); vo = pycolmap.TwoViewGeometryOptions(); vo.ransac.max_error = prm['ransac_px']
    pycolmap.verify_matches(db, pf, vo)
    ctx.progress(0.66, 'triangulating'); ctx.log('triangulating with the solved cameras ...')
    raw = p.model_dir('dense_raw')
    rec2 = _triangulate(rec, db, p.images_dir, raw, prm); rec2.update_point_3d_errors()
    ctx.metric('points_first_triangulation', rec2.num_points3D()); ctx.metric('reprojection_px_before_refine', round(rec2.compute_mean_reprojection_error(), 3))
    ctx.log(f'{rec2.num_points3D()} points, mean error {rec2.compute_mean_reprojection_error():.3f} px')
    if prm['refine']:
        ctx.progress(0.7, 'refinement adjustment (CPU)')
        sub = pycolmap.Reconstruction(rec2)
        pts = sorted(sub.points3D.items(), key=lambda kv: (-kv[1].track.length(), kv[1].error))
        for pid, _ in pts[prm['refine_points']:]: sub.delete_point3D(pid)
        ctx.log(f'refinement on {sub.num_points3D()} best points ...')
        opt = pycolmap.BundleAdjustmentOptions(); opt.refine_principal_point = prm['refine_principal_point']
        opt.ceres.solver_options.max_num_iterations = prm['ba_iterations']; ba_logging(opt)
        sub.update_point_3d_errors(); e0 = sub.compute_mean_reprojection_error()
        if g is not None and src != 'initial':
            idn = {n: i for i, n in enumerate(names)}
            cov = np.diag([prm['gps_sigma_h'] ** 2] * 2 + [(1e4 if g['ref'].get('no_altitude') else prm['gps_sigma_v']) ** 2])
            pri = [pycolmap.PosePrior(corr_data_id=im.data_id, position=g['enu'][idn[im.name]], position_covariance=cov,
                                      coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN) for im in sub.images.values() if g['valid'][idn[im.name]]]
            cfg = pycolmap.BundleAdjustmentConfig()
            for i in sub.images: cfg.add_image(i)
            ctx.live_gps = {n_: g['enu'][i_] for i_, n_ in enumerate(names) if g['valid'][i_]}
            done, tot = 0, prm['ba_iterations']; ctx.live_scene(sub, 'refinement (lens + cameras, GPS priors)', 'start', force=True)
            while done < tot:                         # bursts of iterations so the app shows the refinement converging
                opt.ceres.solver_options.max_num_iterations = min(10, tot - done)
                s = pycolmap.create_pose_prior_bundle_adjuster(opt, pycolmap.PosePriorBundleAdjustmentOptions(), cfg, pri, sub).solve(); done += min(10, tot - done)
                ctx.live_scene(sub, 'refinement (lens + cameras, GPS priors)', f'iteration {done}/{tot}', force=True)
                ctx.progress(0.7 + 0.18 * done / tot, f'refinement: iteration {done}/{tot}')
                t_ = str(getattr(s, 'termination_type', s))
                if 'CONVERGENCE' in t_ and 'NO_' not in t_: break
        else:
            C0 = np.array([im.projection_center() for im in sub.images.values()])
            done, tot = 0, prm['ba_iterations']; ctx.live_scene(sub, 'refinement (lens + cameras)', 'start', force=True)
            while done < tot:
                opt.ceres.solver_options.max_num_iterations = min(10, tot - done); pycolmap.bundle_adjustment(sub, opt); done += min(10, tot - done)
                ctx.live_scene(sub, 'refinement (lens + cameras)', f'iteration {done}/{tot}', force=True)
            s = 'done'
            from .. import geom
            C1 = np.array([im.projection_center() for im in sub.images.values()]); sc, R, t = geom.umeyama(C1, C0); sub.transform(geom.sim3(sc, R, t))
        sub.update_point_3d_errors(); e1 = sub.compute_mean_reprojection_error(); ctx.log(f'refinement: {getattr(s, "termination_type", s)}; error {e0:.3f} -> {e1:.3f} px')
        ctx.metric('refine_error_px', f'{e0:.3f} -> {e1:.3f}')
        ctx.metric('camera_after_refine', np.round(sub.cameras[next(iter(sub.cameras))].params, 5).tolist())
        ref = p.model_dir('refined'); os.makedirs(ref, exist_ok=True)
        cam_only = pycolmap.Reconstruction(sub)
        for pid in list(cam_only.points3D): cam_only.delete_point3D(pid)
        cam_only.write(ref)
        ctx.progress(0.9, 're-triangulating'); rec2 = _triangulate(cam_only, db, p.images_dir, raw, prm); rec2.update_point_3d_errors()
        ctx.log(f'after refinement: {rec2.num_points3D()} points, mean error {rec2.compute_mean_reprojection_error():.3f} px')
        mv = [np.linalg.norm(sub.images[i].projection_center() - rec.images[i].projection_center()) for i in sub.images]
        ctx.metric('camera_moves_m', dict(median=round(float(np.median(mv)), 3), p90=round(float(np.percentile(mv, 90)), 3)))
    rec2.write(raw)
    ctx.metric('points', rec2.num_points3D()); ctx.metric('mean_reprojection_px', round(rec2.compute_mean_reprojection_error(), 3))
    e_all = np.array([q.error for q in rec2.points3D.values()]); tl = np.array([q.track.length() for q in rec2.points3D.values()])
    med = float(np.median(e_all)); ctx.metric('median_reprojection_px', round(med, 3))
    ctx.metric('median_error_by_track_length', {int(L): round(float(np.median(e_all[tl == L])), 3) for L in (2, 3, 4, 6, 10) if (tl == L).any()})
    ctx.check('dense points precise (sharp splats)', med < 0.8, f'{med:.2f} px median', '< 0.8 px (commercial photogrammetry on 4K video: 0.33)',
              'Imprecise points cause ghosting/doubled edges in splats: use detector sift, lower snap px (0.7) and track px (1.5).',
              level=None if med < 0.8 else 'warn')
    MS.summary(rec2, names, g, ctx, checks=False)
    # reprojection overlay sample
    im = rec2.images[sorted(rec2.images)[len(rec2.images) // 2]]
    obs = np.array([pp.xy for pp in im.points2D if pp.has_point3D()]); err = np.array([rec2.points3D[pp.point3D_id].error for pp in im.points2D if pp.has_point3D()])
    fig, ax = plt.subplots(figsize=(16, 9)); ax.imshow(load_image(os.path.join(p.images_dir, im.name)))
    sc = ax.scatter(obs[:, 0], obs[:, 1], c=np.clip(err, 0, 3), s=1, cmap='turbo', vmin=0, vmax=3); plt.colorbar(sc, ax=ax, fraction=0.03, label='point error (px)')
    ax.axis('off'); ax.set_title(f'{im.name}: {len(obs)} dense points'); ctx.fig(fig, 'dense_overlay', 'Dense points on a frame, coloured by error')
    ctx.log('next: step 10 (filter) to choose how many points to keep')
