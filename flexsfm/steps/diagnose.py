"""Step 7: find problems in the current model and propose fixes."""
import json, os
import numpy as np
from ..common import P, plt, plan_axes
from .. import modelstats as MS, revisit as RV

TITLE = '7  Diagnose'
WHAT = """Looks for the problems Metashape does not show you, and proposes a fix for each (Issues tab):
- Revisits not tied (doubled surfaces): where the capture passes the same place again (other pass / opposite direction),
  real FlexGlue matches between those images are compared with the current camera poses. If they disagree much more
  than matches between neighbouring frames do, the passes are offset -> run 'Repair'.
- Pass offsets against GPS, frames not placed, frames that fit badly, weak ties.
The 'misalignment' numbers are in pixels and, roughly, in metres at the ground (pixels x distance / focal length)."""
NEEDS = ['reconstruct']
MIN_OFFSET_M = 0.10   # with GPS (metres): revisit offsets below this are not reported as doubled surfaces

PARAMS = [
    P('sample_pairs', 400, 'How many revisit pairs to test with real matches (more = slower, more reliable)', lo=20, hi=5000),
    P('footprint_m', 4.0, 'Images whose viewed areas are this close (m, or model units) count as seeing the same place', lo=0.2, hi=50.0),
    P('time_gap', 60, 'Without GPS passes: frames further apart than this count as a revisit', lo=10),
    P('ratio_limit', 2.0, 'Revisit error larger than this many times the neighbour error = misaligned', lo=1.0, hi=10.0),
]


def rolling_shutter_check(rec, ctx, n=120):
    """Rolling shutter makes the residuals depend on the image row and grow with speed. Measures how much of the
    residual a row-dependent shift explains, for fast vs slow frames."""
    ims = sorted(rec.images.values(), key=lambda im: im.name)
    if len(ims) < 10: return None
    C = np.array([im.projection_center() for im in ims]); cam = rec.cameras[ims[0].camera_id]; cy = cam.height / 2
    rng = np.random.default_rng(1); out = []
    for k in rng.choice(np.arange(1, len(ims) - 1), min(n, len(ims) - 2), replace=False):
        im = ims[k]; ob = [(q.xy, rec.points3D[q.point3D_id].xyz) for q in im.points2D if q.has_point3D()]
        if len(ob) < 50: continue
        xy = np.array([o[0] for o in ob]); pr = [im.project_point(o[1]) for o in ob]
        ok = np.array([q is not None for q in pr]); xy = xy[ok]; d = np.array([q for q in pr if q is not None]) - xy
        A = np.c_[np.ones(len(xy)), (xy[:, 1] - cy) / cy]
        res = sum(((d[:, c] - A @ np.linalg.lstsq(A, d[:, c], rcond=None)[0]) ** 2).sum() for c in (0, 1))
        res0 = sum(((d[:, c] - d[:, c].mean()) ** 2).sum() for c in (0, 1))
        out.append((np.linalg.norm(C[k + 1] - C[k - 1]) / 2, 1 - res / max(res0, 1e-9)))
    if not out: return None
    o = np.array(out); fast = o[:, 0] >= np.median(o[:, 0])
    ef, es = float(np.median(o[fast, 1])), float(np.median(o[~fast, 1]))
    ctx.metric('rolling_shutter_explained_fast_slow', [round(ef, 3), round(es, 3)])
    ctx.check('rolling shutter effect', ef < 0.15, f'{100 * ef:.1f} % of residual (fast frames), {100 * es:.1f} % (slow)', '< 15 %',
              'Rolling shutter is visible: fly slower / turn slower, or use a global-shutter or photo mode. COLMAP has no rolling-shutter model.',
              level=None if ef < 0.15 else 'warn')
    return ef, es


def run(p, prm, ctx):
    import pycolmap
    rec = pycolmap.Reconstruction(p.current_model); names = p.names(); g = p.gps(); idx = {n: i for i, n in enumerate(names)}
    ctx.log(f'model: {p.current_model}')
    st = MS.summary(rec, names, g, ctx)
    issues = []
    rs = rolling_shutter_check(rec, ctx)
    unreg = (~st['reg']).sum()
    if unreg: issues.append(dict(severity='warn', title=f'{unreg} frames not placed', detail='See reconstruct log for names. Usually weak links.',
                                 action=dict(step='pairs', params={'sequential': p.params('pairs')['sequential'] + 4}, label='More sequential pairs, then re-run 3-5')))
    grp, kind = RV.groups_for(names, g, prm['time_gap'])
    cand, fp = RV.candidate_pairs(rec, names, grp, prm['footprint_m'], k=3)
    ctx.metric('revisit_candidates', len(cand)); ctx.metric('revisit_grouping', kind)
    if not cand:
        ctx.log('no revisits: the sequence never sees the same place twice from another pass - nothing to tie')
        ctx.check('revisits tied', True, 'no revisits', '-', level='pass')
    else:
        shared = np.array([c[4] for c in cand]); zero = float((shared == 0).mean())
        ctx.metric('revisit_pairs_with_no_shared_points', f'{100 * zero:.0f} %')
        rng = np.random.default_rng(0); sel = [cand[k] for k in rng.choice(len(cand), min(prm['sample_pairs'], len(cand)), replace=False)]
        # controls: same group, a few frames apart
        id_of = {im.name: i for i, im in rec.images.items()}; ctrl = []
        for c in sel[: max(20, len(sel) // 2)]:
            k = idx[rec.images[c[0]].name]
            for d in (3, -3, 5, -5):
                if 0 <= k + d < len(names) and names[k + d] in id_of and grp[k + d] == grp[k]: ctrl.append((c[0], id_of[names[k + d]])); break
        ctx.log(f'testing {len(sel)} revisit pairs and {len(ctrl)} neighbour pairs with FlexGlue ...')
        mr = RV.match_and_check(p, rec, [(c[0], c[1]) for c in sel], ctx)
        mc = RV.match_and_check(p, rec, ctrl, None)
        er = {k: float(np.median(v[1])) for k, v in mr.items()}; ec = np.array([float(np.median(v[1])) for v in mc.values()]) if mc else np.array([1.0])
        r_med = float(np.median(list(er.values()))) if er else np.nan; c_med = float(np.median(ec))
        ctx.metric('revisit_pairs_matched', f'{len(mr)}/{len(sel)}'); ctx.metric('revisit_error_px_median', round(r_med, 2)); ctx.metric('neighbour_error_px_median', round(c_med, 2))
        # approx metres
        dist = {k: np.linalg.norm(fp[k[0]][0] - fp[k[0]][2]) for k in er}
        f = rec.cameras[next(iter(rec.cameras))].mean_focal_length()
        er_m = {k: er[k] * dist[k] / f for k in er}
        ctx.metric('revisit_misalignment_m_median', round(float(np.median(list(er_m.values()))), 3) if er_m else None)
        ratio = r_med / max(c_med, 0.3)
        m_med = float(np.median(list(er_m.values()))) if er_m else None
        tiny = g is not None and m_med is not None and m_med < MIN_OFFSET_M      # in GPS metres and only a few cm: not a doubled surface
        tied = ratio < prm['ratio_limit'] or tiny
        if tiny and ratio >= prm['ratio_limit']:
            ctx.log(f'revisits disagree x{ratio:.1f} more than neighbours, but only ~{m_med * 100:.0f} cm - below {MIN_OFFSET_M * 100:.0f} cm, not a doubled surface')
        ctx.check('revisits line up (no doubled lanes)', tied, f'revisit {r_med:.1f} px vs neighbours {c_med:.1f} px (x{ratio:.1f}), ~{np.median(list(er_m.values())) if er_m else 0:.2f} m',
                  f'< x{prm["ratio_limit"]}' + (f' or < {MIN_OFFSET_M * 100:.0f} cm' if g is not None else ''), 'Passes are offset (doubled lanes): run step 8 Repair.', level=None if tied else ('fail' if ratio > 3 else 'warn'))
        ctx.check('revisits share points', zero < 0.5, f'{100 * zero:.0f} % of revisit image pairs share no 3D point', '< 50 %',
                  'Passes are only held together by GPS: run step 8 Repair to tie them.', level=None if zero < 0.5 else 'warn')
        if not tied or zero >= 0.5:
            issues.append(dict(severity='fail' if not tied else 'warn', title='Passes not tied together (doubled surfaces)',
                               detail=f'revisit matches disagree with the poses by {r_med:.1f} px (neighbours {c_med:.1f} px); {100 * zero:.0f} % of revisit pairs share no points.',
                               action=dict(step='repair', params={}, label='Run Repair')))
        # figures
        fr = np.array([idx[rec.images[k[0]].name] for k in er]); ev = np.array(list(er.values())); em = np.array(list(er_m.values()))
        fig, ax = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
        ax[0].scatter(fr, ev, s=6, c='#d62728', label='revisit pairs'); fc = [idx[rec.images[k[0]].name] for k in mc]
        ax[0].scatter(fc, [np.median(v[1]) for v in mc.values()], s=6, c='#2ca02c', label='neighbour pairs (reference)')
        ax[0].set_yscale('log'); ax[0].set_ylabel('median epipolar error (px)'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        ax[0].set_title('real matches vs current poses: revisits far above neighbours = passes offset')
        cs = np.array([c[4] for c in cand]); cf = np.array([idx[rec.images[c[0]].name] for c in cand])
        ax[1].scatter(cf, np.maximum(cs, 0.8), s=3); ax[1].set_yscale('log'); ax[1].set_ylabel('3D points shared'); ax[1].set_xlabel('frame'); ax[1].grid(alpha=.3)
        ax[1].set_title('points the model shares between revisit images (bottom row = not tied at all)')
        ctx.fig(fig, 'revisit_errors', 'Revisit check along the sequence')
        fig, ax = plt.subplots(figsize=(11, 8))
        C = st['C']; ok = st['reg']; ax.plot(C[ok, 0], C[ok, 1], '-', c='#cccccc', lw=1)
        pts = np.array([fp[k[0]][0][:2] for k in er])
        if len(pts):
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=em, cmap='inferno_r', s=14, vmin=0, vmax=max(0.2, np.percentile(em, 95)))
            plt.colorbar(sc, ax=ax, label='approx. misalignment at the ground (m)')
        plan_axes(ax); ax.set_title('where revisits disagree (approximate metres)')
        ctx.fig(fig, 'revisit_map', 'Map of revisit misalignment')
    if g is not None and 'pass_id' in g:
        gr = MS.gps_residuals(st['C'], g, st['reg'])
        offs = {int(k): np.nanmean(gr['d'][(g['pass_id'] == k) & gr['ok']], 0).round(2).tolist() for k in np.unique(g['pass_id'])}
        ctx.metric('mean_offset_to_gps_per_pass_m', offs)
    json.dump(issues, open(p.path('issues.json'), 'w'), indent=1)
    ctx.metric('issues', len(issues))
    if not issues: ctx.log('no issues found')
    for it in issues: ctx.log(f'ISSUE [{it["severity"]}] {it["title"]}: {it["detail"]} -> {it["action"]["label"]}')
