"""Step 8 (optional): tie revisits / passes together (fixes doubled surfaces)."""
import json, os
import numpy as np
from ..common import P, plt, FeatureStore
from .. import modelstats as MS, revisit as RV, geom

TITLE = '8  Repair revisits (optional)'
WHAT = """Fixes doubled surfaces / ghosting where the capture passes the same place again but the model keeps the passes apart.
1. Finds image pairs from different passes looking at the same place (incl. opposite directions).
2. Matches them with FlexGlue (it handles large viewpoint changes far better than SIFT/LightGlue).
3. Adds those matches to the model as new tie points, then bundle-adjusts (with GPS priors if GPS exists).
4. Validates on pairs held out from the fix: their disagreement with the poses must drop. The result becomes the
   current model only if validation improves - otherwise the previous model is kept and the reason is reported.
The bundle adjustment runs on the CPU: expect ~1 hour for 2000-3000 frames."""
NEEDS = ['reconstruct']
PARAMS = [
    P('pairs_per_image', 4, 'Revisit partners per image', lo=1, hi=20),
    P('footprint_m', 4.0, 'Images whose viewed areas are this close count as seeing the same place (m)', lo=0.2, hi=50.0),
    P('time_gap', 60, 'Without GPS passes: frames further apart than this count as a revisit', lo=10),
    P('min_inliers', 25, 'Revisit pairs need at least this many verified matches', lo=8),
    P('gate_px', 60.0, 'Matches disagreeing with the current poses by more than this are treated as wrong', lo=5.0, hi=500.0),
    P('holdout', 0.2, 'Fraction of revisit pairs kept aside to validate the fix', lo=0.05, hi=0.5),
    P('gps_sigma_h', 1.5, 'GPS horizontal uncertainty in the bundle adjustment (m)', lo=0.05, hi=50.0),
    P('gps_sigma_v', 1.0, 'GPS vertical uncertainty (m)', lo=0.05, hi=100.0),
    P('ba_iterations', 50, 'Max bundle-adjustment iterations', lo=5, hi=500),
    P('refine_intrinsics', False, 'Also refine focal length / distortion', advanced=True),
]


def _validate(rec, D, fs):
    if not D: return None
    e = np.concatenate([geom.epipolar_px(rec, a, b, fs.kp(rec.images[a].name)[m[:, 0]], fs.kp(rec.images[b].name)[m[:, 1]]) for (a, b), (m, _) in D.items()])
    return dict(median_px=round(float(np.median(e)), 2), p90_px=round(float(np.percentile(e, 90)), 2)), e


def run(p, prm, ctx):
    import pycolmap
    src_name = p.state['model']; rec = pycolmap.Reconstruction(p.current_model); names = p.names(); g = p.gps(); idx = {n: i for i, n in enumerate(names)}
    ctx.log(f'starting from models/{src_name}')
    grp, kind = RV.groups_for(names, g, prm['time_gap'])
    cand, fp = RV.candidate_pairs(rec, names, grp, prm['footprint_m'], k=prm['pairs_per_image'])
    if not cand: ctx.log('no revisits found - nothing to repair'); ctx.check('no revisits to tie (nothing to repair)', True, '0 revisit pairs', '-', level='pass'); return
    rng = np.random.default_rng(0); hold = rng.random(len(cand)) < prm['holdout']
    ctx.log(f'{len(cand)} revisit pairs ({hold.sum()} held out for validation), grouping by {kind}')
    M = RV.match_and_check(p, rec, [(c[0], c[1]) for c, h in zip(cand, hold) if not h], ctx, min_inliers=prm['min_inliers'], gate_px=prm['gate_px'])
    H = RV.match_and_check(p, rec, [(c[0], c[1]) for c, h in zip(cand, hold) if h], None, min_inliers=prm['min_inliers'], gate_px=prm['gate_px'])
    id_of = {im.name: i for i, im in rec.images.items()}; ctrl = []
    for c in cand[::max(1, len(cand) // 200)]:
        k = idx[rec.images[c[0]].name]
        if k + 3 < len(names) and names[k + 3] in id_of and grp[k + 3] == grp[k]: ctrl.append((c[0], id_of[names[k + 3]]))
    Cm = RV.match_and_check(p, rec, ctrl, None)
    ctx.metric('pairs_used', len(M)); ctx.metric('pairs_held_out', len(H)); ctx.metric('control_pairs', len(Cm))
    if len(M) < 10: raise RuntimeError(f'only {len(M)} revisit pairs matched - cannot repair (try a larger "footprint m")')
    fs = FeatureStore(p.path('features'))
    vb_h, eb_h = _validate(rec, H, fs) or (None, None); vb_c, eb_c = _validate(rec, Cm, fs) or (None, None)
    ctx.metric('before_heldout_px', vb_h); ctx.metric('before_control_px', vb_c); ctx.log(f'before: held-out {vb_h}, control {vb_c}')
    with open(p.path('repair_pairs.txt'), 'w') as f: f.writelines(f'{rec.images[a].name} {rec.images[b].name}\n' for a, b in M)
    # ---- tracks (union-find), triangulate, add
    par = {}
    def fnd(x):
        while par.setdefault(x, x) != x: par[x] = par[par[x]]; x = par[x]
        return x
    for (a, b), (m, _) in M.items():
        for u, v in m: par[fnd((a, int(u)))] = fnd((b, int(v)))
    groups = {}
    for x in list(par): groups.setdefault(fnd(x), []).append(x)
    added = nobs = 0
    for gg in groups.values():
        ims = [x[0] for x in gg]
        if len(set(ims)) != len(ims): continue
        obs = [(i, fs.kp(rec.images[i].name)[k] + 0.5) for i, k in gg]
        X, err = geom.triangulate_track(rec, obs)
        if X is None or err > prm['gate_px']: continue
        tr = pycolmap.Track()
        for i, xy in obs:
            rec.images[i].points2D.append(pycolmap.Point2D(np.asarray(xy, float))); tr.add_element(i, rec.images[i].num_points2D() - 1)
        rec.add_point3D(X, tr, np.array([255, 0, 255], np.uint8)); added += 1; nobs += len(obs)
    ctx.metric('tie_points_added', added); ctx.metric('tie_observations', nobs); ctx.log(f'added {added} tie points ({nobs} observations)')
    C0 = {i: im.projection_center() for i, im in rec.images.items()}
    ctx.progress(0.5, 'bundle adjustment (CPU)')
    if g is not None and src_name != 'initial':
        MS.gps_prior_ba(rec, names, g, prm['gps_sigma_h'], prm['gps_sigma_v'], 1, ctx, prm['ba_iterations'], prm['refine_intrinsics'])
    else:
        MS.plain_ba(rec, ctx, prm['ba_iterations'], prm['refine_intrinsics'])
        A = np.array([rec.images[i].projection_center() for i in C0]); B = np.array([C0[i] for i in C0])
        s, R, t = geom.umeyama(A, B); rec.transform(geom.sim3(s, R, t))    # keep the original frame
    va_h, ea_h = _validate(rec, H, fs) or (None, None); va_c, ea_c = _validate(rec, Cm, fs) or (None, None)
    ctx.metric('after_heldout_px', va_h); ctx.metric('after_control_px', va_c); ctx.log(f'after: held-out {va_h}, control {va_c}')
    mv = np.array([np.linalg.norm(rec.images[i].projection_center() - C0[i]) for i in C0])
    ctx.metric('camera_moves_m', dict(median=round(float(np.median(mv)), 3), p90=round(float(np.percentile(mv, 90)), 3), max=round(float(mv.max()), 3)))
    better = va_h and vb_h and va_h['median_px'] < 0.8 * vb_h['median_px']
    control_ok = not (va_c and vb_c) or va_c['median_px'] < 1.3 * vb_c['median_px'] + 0.3
    already_ok = vb_h and vb_c and vb_h['median_px'] < max(1.5, 1.5 * vb_c['median_px'])     # revisits already agree like neighbours do
    if already_ok and not better:
        ctx.check('revisits already line up (no repair needed)', True, f'held-out {vb_h["median_px"]} px vs neighbours {vb_c["median_px"]} px before repair', 'similar',
                  level='pass'); ctx.log('revisits already line up as well as neighbouring frames - nothing to repair, model kept')
    else:
        ctx.check('held-out revisit pairs improved', bool(better), f'{vb_h and vb_h["median_px"]} -> {va_h and va_h["median_px"]} px', '>= 20 % lower',
                  'No improvement: the passes may need more ties (raise "pairs per image" / "footprint m") or GPS trust lower (raise sigma).')
    ctx.check('neighbour pairs not worse', control_ok, f'{vb_c and vb_c["median_px"]} -> {va_c and va_c["median_px"]} px', 'about equal',
              'The repair distorted the model - it was not adopted.')
    d = p.model_dir('repaired'); os.makedirs(d, exist_ok=True); rec.write(d)
    if better and control_ok:
        p.set_model('repaired'); ctx.log('repaired model adopted: models/repaired is now the current model')
    else:
        ctx.log(f'repair NOT adopted - current model stays models/{src_name} (the attempt is kept in models/repaired for inspection)')
    MS.summary(rec, names, g, ctx, checks=False)
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    for a, (b_, a_, t) in zip(ax[:2], ((eb_h, ea_h, 'held-out revisit pairs'), (eb_c, ea_c, 'neighbour pairs (control)'))):
        if b_ is not None:
            bins = np.linspace(0, np.percentile(b_, 98), 60); a.hist(b_, bins=bins, alpha=.6, label='before'); a.hist(a_, bins=bins, alpha=.6, label='after')
        a.set_title(t); a.set_xlabel('epipolar error (px)'); a.legend()
    fr = [idx[rec.images[i].name] for i in C0]; ax[2].scatter(fr, mv, s=2); ax[2].set_xlabel('frame'); ax[2].set_ylabel('camera moved (m)'); ax[2].set_title('how far the repair moved each camera')
    ctx.fig(fig, 'repair_validation', 'Validation: held-out revisit matches should agree much better after the repair')
