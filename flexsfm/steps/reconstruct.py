"""Step 5: cameras + sparse points, aligned to GPS."""
import os, shutil, time
import numpy as np
from ..common import P
from .. import modelstats as MS

TITLE = '5  Reconstruct cameras'
WHAT = """Solves all camera positions/orientations and the lens from the verified matches, then (with GPS) places the
model in real metres with a similarity fitted to the GPS track, and optionally refines it with GPS inside the
bundle adjustment.
- global mapper (GLOMAP): solves all cameras at once - fast and less drift on long sequences. Recommended.
- incremental mapper: adds cameras one by one - slower, sometimes more robust on hard data.
- 'GPS in bundle adjustment' pulls each camera towards its GPS position with the given uncertainty. It removes slow
  bends over long distances. It runs on the CPU and takes long on big projects (about 1 hour per pass for 2000+ frames).
Check: all frames placed in ONE model, low error, agreement with GPS."""
NEEDS = ['match']
PARAMS = [
    P('mapper', 'global', 'global = GLOMAP (recommended); incremental = classic COLMAP', choices=['global', 'incremental']),
    P('gps_align', True, 'Put the model in GPS metres (similarity fit to the GPS track). Ignored without GPS.'),
    P('gps_ba_passes', 0, 'Bundle adjustment passes with GPS priors (0 = off). Each pass is slow on the CPU.', lo=0, hi=5),
    P('gps_sigma_h', 1.5, 'GPS horizontal uncertainty (m). Smaller = trust GPS more.', lo=0.05, hi=50.0),
    P('gps_sigma_v', 1.0, 'GPS vertical uncertainty (m)', lo=0.05, hi=100.0),
    P('ba_iterations', 50, 'Max iterations per bundle-adjustment pass', lo=5, hi=500, advanced=True),
    P('align_inlier_m', 5.0, 'Cameras further than this from GPS are ignored when fitting the similarity', lo=0.5, hi=100.0, advanced=True),
]


def run(p, prm, ctx):
    import pycolmap
    names = p.names(); g = p.gps(); db = p.path('database.db')
    out = p.path('models', '_mapper')
    if os.path.exists(out): shutil.rmtree(out)
    os.makedirs(out)
    n = len(names)
    ctx.progress(0.05, f'{prm["mapper"]} mapping of {n} images (stages are printed below)')
    ctx.log(f'{prm["mapper"]} mapping of {n} images ... COLMAP prints each stage below. One call does all of it, so there is no percentage;'
            + (f' with {n} images expect roughly {max(1, n // 300)}-{max(2, n // 100)} min - long quiet stretches (global positioning,'
               ' bundle adjustment) are normal.' if n > 500 else ''))
    lg = getattr(pycolmap, 'logging', None); old = None
    try:                                              # always show the mapper's stage messages, even in quiet mode
        if lg is not None: old = lg.minloglevel; lg.minloglevel = 0; lg.logtostderr = True
    except Exception: pass
    t0 = time.time()
    try:
        recs = _map_live(pycolmap, prm['mapper'], db, p.images_dir, out, ctx, n)
    finally:
        try:
            if lg is not None and old is not None: lg.minloglevel = old
        except Exception: pass
    ctx.log(f'mapping finished in {(time.time() - t0) / 60:.1f} min')
    if not recs: raise RuntimeError('mapping produced no model - check the match step (weak links / isolated frames)')
    sizes = sorted(((r.num_reg_images(), k) for k, r in recs.items()), reverse=True)
    ctx.metric('models_found', len(recs)); ctx.metric('model_sizes', [s for s, _ in sizes])
    ctx.check('one connected model', len(recs) == 1, f'{len(recs)} models: {[s for s, _ in sizes]} frames', '1',
              'The sequence broke into pieces at weak links. See step 4 link strength; raise "sequential" / lower "min covisibility", then re-run 4 and 5.',
              level=None if len(recs) == 1 else 'warn')
    rec = recs[sizes[0][1]]
    ini = p.model_dir('initial'); os.makedirs(ini, exist_ok=True); rec.write(ini); ctx.log(rec.summary())
    name = 'initial'
    if g is not None and prm['gps_align']:
        ctx.progress(0.5, 'aligning to GPS')
        s, ni, nt = MS.align_to_gps(rec, names, g, prm['align_inlier_m'])
        ctx.live_gps = {n_: g['enu'][i_] for i_, n_ in enumerate(names) if g['valid'][i_]}   # live 3D shows GPS dots + offsets from here on
        ctx.live_scene(rec, 'aligned to GPS', f'similarity fitted to {nt} GPS positions (scale {s:.4f}) - the model is now in metres', force=True)
        ctx.metric('similarity_scale', round(float(s), 6)); ctx.metric('align_inliers', f'{ni}/{nt}')
        ctx.check('GPS alignment inliers', ni > 0.8 * nt, f'{ni}/{nt}', '> 80 %',
                  'Many cameras far from GPS: the model may be bent or the GPS poor. Look at "distance to GPS".', level=None if ni > 0.8 * nt else 'warn')
        MS.summary(rec, names, g, ctx, prefix='before_gps_ba_', checks=False) if prm['gps_ba_passes'] else None
        if prm['gps_ba_passes']:
            ctx.progress(0.6, 'GPS bundle adjustment')
            MS.gps_prior_ba(rec, names, g, prm['gps_sigma_h'], prm['gps_sigma_v'], prm['gps_ba_passes'], ctx, prm['ba_iterations'])
        name = 'aligned'
    elif g is None:
        ctx.log('no GPS: model kept in its own arbitrary frame (use markers for scale/orientation)')
    d = p.model_dir(name); os.makedirs(d, exist_ok=True); rec.write(d); p.set_model(name)
    ctx.progress(0.9, 'report')
    MS.summary(rec, names, g, ctx)
    reg = {im.name for im in rec.images.values()}; miss = [n for n in names if n not in reg]
    if miss: ctx.log(f'{len(miss)} frames not placed: ' + ', '.join(miss[:60]))
    ctx.log(f'current model: models/{name}')


def _map_live(pycolmap, mapper, db_path, image_path, out, ctx, n_images):
    """Run the mapper through its pipeline object so every model update is shown live in the app (3D view + error curve).
    Falls back to the plain one-call API if the pipeline classes are not available."""
    try:
        rm = pycolmap.ReconstructionManager(); db = pycolmap.Database.open(db_path); state = {'k': 0}
        def latest():
            return rm.get(rm.size() - 1) if rm.size() else None
        if mapper == 'global':
            opt = pycolmap.GlobalPipelineOptions(); opt.image_path = image_path
            pipe = pycolmap.GlobalPipeline(opt, db, rm)
            names = ['global positioning (first camera positions from rotations + tracks)', 'bundle adjustment (rotations fixed)',
                     'bundle adjustment (joint)', 'bundle adjustment (joint, refined)', 'retriangulation + final refinement']
            def cb():
                r = latest(); k = state['k']; state['k'] += 1
                if r is not None:
                    st = names[k] if k < len(names) else f'model update {k + 1}'
                    ctx.live_scene(r, st, f'{r.num_reg_images()} cameras, {r.num_points3D()} points', force=True)
                    ctx.progress(min(0.6, 0.1 + 0.1 * (k + 1)), st); ctx.log(f'model update {k + 1}: {st} - {r.num_reg_images()} cameras, {r.num_points3D()} points')
            pipe.add_callback(pycolmap.GlobalPipelineCallback.MODEL_UPDATE_CALLBACK, cb)
            ctx.log('rotation averaging + track establishment (no positions yet - the 3D view starts at global positioning) ...')
        else:
            opt = pycolmap.IncrementalPipelineOptions(); opt.image_path = image_path
            pipe = pycolmap.IncrementalPipeline(opt, db, rm)
            def cb_init():
                r = latest()
                if r is not None: ctx.live_scene(r, 'initial pair', 'the first two cameras and their points', force=True)
            def cb_next():
                r = latest()
                if r is not None:
                    k = r.num_reg_images(); ctx.live_scene(r, 'adding cameras', f'{k} / {n_images} cameras placed, {r.num_points3D()} points')
                    ctx.progress(min(0.6, 0.05 + 0.55 * k / max(1, n_images)), f'incremental mapping: {k}/{n_images} cameras')
            pipe.add_callback(pycolmap.IncrementalPipelineCallback.INITIAL_IMAGE_PAIR_REG_CALLBACK, cb_init)
            pipe.add_callback(pycolmap.IncrementalPipelineCallback.NEXT_IMAGE_REG_CALLBACK, cb_next)
        pipe.run()
        recs = {k: rm.get(k) for k in range(rm.size())}
        if recs: rm.write(out)
        r = max(recs.values(), key=lambda x: x.num_reg_images()) if recs else None
        if r is not None: ctx.live_scene(r, 'mapping finished', f'{r.num_reg_images()} cameras, {r.num_points3D()} points', force=True)
        return recs
    except Exception as e:
        ctx.log(f'(live mapping view not available: {type(e).__name__}: {e} - running without it)')
        if mapper == 'global': return pycolmap.global_mapping(db_path, image_path, out, pycolmap.GlobalPipelineOptions())
        return pycolmap.incremental_mapping(db_path, image_path, out, pycolmap.IncrementalPipelineOptions())
