"""Statistics, checks and figures for a reconstruction (shared by reconstruct / repair / densify / filter / diagnose)."""
import numpy as np
from .common import plt, plan_axes, route_plot, ba_logging


def per_image(rec, names):
    idx = {n: i for i, n in enumerate(names)}; N = len(names)
    reg = np.zeros(N, bool); err = np.full(N, np.nan); nobs = np.zeros(N, int); C = np.full((N, 3), np.nan)
    pts = rec.points3D
    for im in rec.images.values():
        if im.name not in idx or not im.has_pose: continue
        k = idx[im.name]; reg[k] = True; C[k] = im.projection_center()
        e = [pts[p.point3D_id].error for p in im.points2D if p.has_point3D()]
        nobs[k] = len(e); err[k] = float(np.mean(e)) if e else np.nan
    return dict(reg=reg, err=err, nobs=nobs, C=C)


def gps_residuals(C, g, reg):
    if g is None: return None
    ok = reg & g['valid']; d = np.full((len(C), 3), np.nan); d[ok] = C[ok] - g['enu'][ok]
    h = np.linalg.norm(d[:, :2], axis=1); v = np.abs(d[:, 2])
    no_alt = g['ref'].get('no_altitude', False)
    return dict(d=d, h=h, v=v, ok=ok, no_alt=no_alt,
                summary=dict(horiz_median_m=round(float(np.nanmedian(h)), 3), horiz_p90_m=round(float(np.nanpercentile(h, 90)), 3),
                             vert_median_m=None if no_alt else round(float(np.nanmedian(v)), 3)))


def summary(rec, names, g, ctx, prefix='', checks=True, gps_tol=2.0):
    """Metrics + standard sanity checks + standard figures for a model. Returns the per-image dict."""
    rec.update_point_3d_errors(); st = per_image(rec, names); N = len(names)
    cam = rec.cameras[next(iter(rec.cameras))]
    ctx.metric(prefix + 'registered', f'{int(st["reg"].sum())}/{N}'); ctx.metric(prefix + 'points', rec.num_points3D())
    ctx.metric(prefix + 'mean_reprojection_px', round(rec.compute_mean_reprojection_error(), 3))
    ctx.metric(prefix + 'mean_track_length', round(rec.compute_mean_track_length(), 2))
    ctx.metric(prefix + 'camera', f'{cam.model.name} {cam.width}x{cam.height} params {np.round(cam.params, 5).tolist()}')
    gr = gps_residuals(st['C'], g, st['reg'])
    if gr: ctx.metric(prefix + 'gps_residual', gr['summary'])
    if checks:
        r = st['reg'].mean()
        ctx.check('cameras placed', r > 0.98, f'{st["reg"].sum()}/{N}', '> 98 %',
                  'Unplaced frames: see the figure and log. Usually weak links (step 4) - add pairs or raise "sequential".',
                  level=None if r > 0.98 else ('fail' if r < 0.8 else 'warn'))
        me = rec.compute_mean_reprojection_error()
        ctx.check('reprojection error', me < 2.0, f'{me:.2f} px', '< 2 px', 'High error: check the camera model (step 4) or rolling shutter; run filter.',
                  level=None if me < 2.0 else 'warn')
        bad = np.nan_to_num(st['err']) > 3.0
        ctx.check('no badly fitting frames', bad.sum() == 0, f'{bad.sum()} frames > 3 px', '0',
                  'Frames that fit badly: see "error per frame". Often blur or moving objects - mask them or remove the frames.',
                  level=None if bad.sum() == 0 else 'warn')
        lo = (st['nobs'] < 100) & st['reg']
        ctx.check('every frame well tied', lo.sum() == 0, f'{lo.sum()} frames with < 100 points', '0',
                  'Weakly tied frames can be off. More keypoints (step 2) or pairs (step 3) help.', level=None if lo.sum() == 0 else 'warn')
        if gr:
            hm = gr['summary']['horiz_median_m']
            ctx.check('agrees with GPS', hm < gps_tol, f'{hm} m horizontal median', f'< {gps_tol} m',
                      'Large GPS disagreement: either the GPS is poor, or the model is bent/split. See "distance to GPS" in the per-frame figure.',
                      level=None if hm < gps_tol else 'warn')
        f0 = cam.mean_focal_length(); ctx.metric(prefix + 'focal_px', round(f0, 1))
    # figures
    fig, ax = plt.subplots(1, 2, figsize=(18, 6.5))
    C = st['C']; ok = st['reg']
    if g is not None:
        ax[0].plot(g['enu'][:, 0], g['enu'][:, 1], '-', c='#bbbbbb', lw=1, label='GPS')
        route_plot(ax[0], C[ok], gr['h'][ok] if gr else np.zeros(ok.sum()), 'horizontal distance to GPS (m)', vmin=0, vmax=max(1.0, np.nanpercentile(gr['h'], 95)))
        plan_axes(ax[0])
    else:
        route_plot(ax[0], C[ok][:, :2], np.flatnonzero(ok), 'frame'); ax[0].set_aspect('equal'); ax[0].grid(alpha=.3)
    if (~ok).any(): ax[0].plot([], [], ' ', label=f'{(~ok).sum()} frames not placed')
    ax[0].legend(fontsize=8); ax[0].set_title('camera positions (plan view)')
    X = np.array([p.xyz for p in rec.points3D.values()]) if rec.num_points3D() else np.zeros((1, 3))
    if len(X) > 400000: X = X[np.random.default_rng(0).choice(len(X), 400000, replace=False)]
    lo_, hi_ = np.percentile(X[:, :2], 1, axis=0), np.percentile(X[:, :2], 99, axis=0)
    ax[1].hexbin(X[:, 0], X[:, 1], gridsize=300, bins='log', cmap='viridis', extent=(lo_[0], hi_[0], lo_[1], hi_[1]))
    ax[1].set_aspect('equal'); ax[1].set_title('3D points seen from above (density) - doubled roads/edges show here')
    ctx.fig(fig, prefix + 'cameras_points', 'Cameras and points from above')
    fig, ax = plt.subplots(3 if gr else 2, 1, figsize=(16, 9 if gr else 6), sharex=True)
    ax[0].plot(st['err']); ax[0].axhline(3, c='r', ls='--', lw=.8); ax[0].set_ylabel('mean error (px)'); ax[0].set_title('reprojection error per frame'); ax[0].grid(alpha=.3)
    ax[1].plot(st['nobs']); ax[1].set_ylabel('3D points seen'); ax[1].grid(alpha=.3)
    if gr:
        ax[2].plot(gr['h'], label='horizontal');
        if not gr['no_alt']: ax[2].plot(gr['v'], label='vertical')
        ax[2].set_ylabel('distance to GPS (m)'); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
        if 'pass_id' in g:
            for b in np.flatnonzero(np.diff(g['pass_id'])):
                for a in ax: a.axvline(b, c='k', ls=':', lw=.7)
    ax[-1].set_xlabel('frame'); ctx.fig(fig, prefix + 'per_frame', 'Per-frame quality along the sequence (dotted = turn-backs)')
    if rec.num_points3D():
        e = np.array([p.error for p in rec.points3D.values()]); tl = np.array([p.track.length() for p in rec.points3D.values()])
        fig, ax = plt.subplots(1, 2, figsize=(14, 4))
        ax[0].hist(np.clip(e, 0, 6), bins=60); ax[0].set_title('point reprojection error (px)')
        ax[1].hist(np.clip(tl, 2, 40), bins=np.arange(2, 42)); ax[1].set_title('track length (frames seeing each point)')
        ctx.fig(fig, prefix + 'point_stats', 'Point statistics')
    return st


def gps_prior_ba(rec, names, g, sig_h, sig_v, iters, ctx=None, max_iter=50, refine_intrinsics=True):
    """Bundle adjustment with a position prior per camera (GPS, ENU metres)."""
    import pycolmap
    idx = {n: i for i, n in enumerate(names)}
    if g['ref'].get('no_altitude'): sig_v = 1e4
    cov = np.diag([sig_h ** 2, sig_h ** 2, sig_v ** 2])
    pri = [pycolmap.PosePrior(corr_data_id=im.data_id, position=g['enu'][idx[im.name]], position_covariance=cov,
                              coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN)
           for im in rec.images.values() if im.name in idx and g['valid'][idx[im.name]] and im.has_pose]
    for k in range(iters):
        cfg = pycolmap.BundleAdjustmentConfig()
        for i, im in rec.images.items():
            if im.has_pose: cfg.add_image(i)
        opt = pycolmap.BundleAdjustmentOptions(); opt.refine_principal_point = False
        opt.refine_focal_length = refine_intrinsics; opt.refine_extra_params = refine_intrinsics
        opt.ceres.solver_options.max_num_iterations = max_iter; ba_logging(opt)
        if ctx: ctx.log(f'GPS bundle adjustment pass {k + 1}/{iters} (CPU, up to {max_iter} iterations, shown live in bursts of {BURST}) ...')
        done = 0
        if ctx: ctx.live_scene(rec, f'GPS bundle adjustment {k + 1}/{iters}', 'start', force=True)
        while done < max_iter:                       # run in short bursts so the app can show cameras moving and the error falling
            n = min(BURST, max_iter - done); opt.ceres.solver_options.max_num_iterations = n
            ba = pycolmap.create_pose_prior_bundle_adjuster(opt, pycolmap.PosePriorBundleAdjustmentOptions(), cfg, pri, rec)
            s = ba.solve(); done += n; term = str(getattr(s, 'termination_type', s))
            if ctx:
                ctx.live_scene(rec, f'GPS bundle adjustment {k + 1}/{iters}', f'iteration {done}/{max_iter}', force=True)
                ctx.progress(min(0.99, (k + done / max_iter) / iters), f'GPS bundle adjustment {k + 1}/{iters}: iteration {done}/{max_iter}')
            if 'CONVERGENCE' in term and 'NO_' not in term: break
        if ctx: ctx.log(f'  -> {term} after {done} iterations')
    return rec


BURST = 10   # bundle-adjustment iterations between live snapshots


def plain_ba(rec, ctx=None, max_iter=50, refine_intrinsics=True):
    import pycolmap
    opt = pycolmap.BundleAdjustmentOptions(); opt.refine_principal_point = False
    opt.refine_focal_length = refine_intrinsics; opt.refine_extra_params = refine_intrinsics
    opt.ceres.solver_options.max_num_iterations = max_iter; ba_logging(opt)
    if ctx: ctx.log(f'bundle adjustment (CPU, up to {max_iter} iterations, shown live in bursts of {BURST}) ...'); ctx.live_scene(rec, 'bundle adjustment', 'start', force=True)
    done = 0
    while done < max_iter:
        n = min(BURST, max_iter - done); opt.ceres.solver_options.max_num_iterations = n; pycolmap.bundle_adjustment(rec, opt); done += n
        if ctx: ctx.live_scene(rec, 'bundle adjustment', f'iteration {done}/{max_iter}', force=True)
    return rec


def align_to_gps(rec, names, g, thr=5.0):
    """Similarity (RANSAC) taking the model onto GPS ENU. Without GPS altitude only the horizontal part is trusted."""
    from .geom import robust_umeyama, sim3
    idx = {n: i for i, n in enumerate(names)}
    ims = [im for im in rec.images.values() if im.has_pose and im.name in idx and g['valid'][idx[im.name]]]
    C = np.array([im.projection_center() for im in ims]); G = np.array([g['enu'][idx[im.name]] for im in ims])
    if g['ref'].get('no_altitude'):
        G = G.copy(); G[:, 2] = 0
        for _ in range(4):
            s, R, t, inl = robust_umeyama(C, G, thr); G[:, 2] = ((C @ R.T) * s + t)[:, 2]
    s, R, t, inl = robust_umeyama(C, G, thr)
    rec.transform(sim3(s, R, t))
    return s, int(inl.sum()), len(ims)
