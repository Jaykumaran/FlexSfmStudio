"""Project folder, settings, step state, and the context object every step writes its results through.

Project layout (everything a step produces is kept, so every intermediate output can be inspected):
  project.yaml            settings of every step (edited in the GUI or by hand)
  state.json              status / time / settings hash / key numbers of every step
  frames/                 extracted frames (or the images folder is referenced in place)
  gps.npz                 per-frame GPS in local east/north/up metres (if any)
  features/               keypoints (.kp.npy) + FlexGlue descriptors (.pt) + global descriptors
  pairs.txt, matches.npz, database.db
  models/<name>/          COLMAP models: initial, aligned, repaired, dense_raw, dense ...
  reports/<step>/         figures (.png), checks.json, metrics.json for every step
  logs/<step>.log
  export/                 what goes to Gaussian-splat training
"""
import hashlib, json, os, time, traceback
import yaml

STEP_ORDER = ['ingest', 'features', 'pairs', 'match', 'reconstruct', 'markers', 'diagnose', 'repair', 'densify', 'filter', 'export']
OPTIONAL = {'markers', 'repair'}


def step_module(name):
    import importlib
    return importlib.import_module(f'flexsfm.steps.{name}')


class Project:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        if not os.path.exists(self.cfg_path):
            raise FileNotFoundError(f'{self.root} is not a FlexSfM Studio project (no project.yaml)')
        self.reload()

    # ------------------------------------------------------------------ creation / io
    @staticmethod
    def create(root, **ingest):
        root = os.path.abspath(root); os.makedirs(root, exist_ok=True)
        cfg = {s: {} for s in STEP_ORDER}; cfg['ingest'].update({k: v for k, v in ingest.items() if v not in (None, '')})
        with open(os.path.join(root, 'project.yaml'), 'w') as f: yaml.safe_dump(cfg, f, sort_keys=False)
        with open(os.path.join(root, 'state.json'), 'w') as f: json.dump({'steps': {}, 'model': None}, f)
        return Project(root)

    @property
    def cfg_path(self): return os.path.join(self.root, 'project.yaml')

    @property
    def state_path(self): return os.path.join(self.root, 'state.json')

    def reload(self):
        with open(self.cfg_path) as f: self.cfg = yaml.safe_load(f) or {}
        try:
            with open(self.state_path) as f: self.state = json.load(f)
        except Exception:
            self.state = {'steps': {}, 'model': None}

    def save_cfg(self):
        tmp = self.cfg_path + '.tmp'
        with open(tmp, 'w') as f: yaml.safe_dump(self.cfg, f, sort_keys=False)
        os.replace(tmp, self.cfg_path)

    def save_state(self):
        tmp = self.state_path + '.tmp'
        with open(tmp, 'w') as f: json.dump(self.state, f, indent=1, default=str)
        os.replace(tmp, self.state_path)

    # ------------------------------------------------------------------ paths
    def path(self, *p): return os.path.join(self.root, *p)

    def report_dir(self, step):
        d = self.path('reports', step); os.makedirs(d, exist_ok=True); return d

    def model_dir(self, name): return self.path('models', name)

    @property
    def images_dir(self): return self.state.get('images_dir') or self.path('frames')

    @property
    def masks_dir(self): return self.state.get('masks_dir') or ''

    @property
    def current_model(self):
        m = self.state.get('model'); return self.model_dir(m) if m else None

    def set_model(self, name):
        self.state['model'] = name; self.state.setdefault('model_history', []).append([name, time.strftime('%Y-%m-%d %H:%M:%S')]); self.save_state()

    def names(self):
        p = self.path('images.txt')
        return [l.strip() for l in open(p) if l.strip()] if os.path.exists(p) else []

    def gps(self):
        """dict(names, enu (N,3), lla (N,3) or None, valid (N,) bool, ref) or None when the project has no GPS."""
        p = self.path('gps.npz')
        if not os.path.exists(p): return None
        import numpy as np
        g = np.load(p, allow_pickle=True)
        d = {k: g[k] for k in g.files}
        d['names'] = list(d['names']); d['ref'] = d['ref'].item() if 'ref' in d else {}
        return d if d['valid'].any() else None

    # ------------------------------------------------------------------ parameters
    def params(self, step):
        mod = step_module(step); out = {p['key']: p['default'] for p in mod.PARAMS}
        out.update(self.cfg.get(step) or {}); return out

    def set_params(self, step, values):
        self.cfg.setdefault(step, {}); self.cfg[step] = dict(self.cfg[step] or {}); self.cfg[step].update(values); self.save_cfg()

    def params_hash(self, step):
        return hashlib.md5(json.dumps(self.params(step), sort_keys=True, default=str).encode()).hexdigest()[:10]

    # ------------------------------------------------------------------ status
    def step_state(self, step): return self.state.get('steps', {}).get(step, {})

    def status(self, step):
        """not run / running / failed / done / stale (settings changed or an earlier step was re-run)"""
        st = self.step_state(step); s = st.get('status', 'not run')
        if s == 'running' and st.get('pid') and not _alive(st['pid']): return 'interrupted'
        if s != 'done': return s
        if st.get('params_hash') != self.params_hash(step): return 'stale'
        i = STEP_ORDER.index(step)
        for prev in STEP_ORDER[:i]:
            ps = self.step_state(prev)
            if ps.get('status') == 'done' and ps.get('finished', 0) > st.get('finished', 0) and prev in step_module(step).NEEDS:
                return 'stale'
        return 'done'


def _alive(pid):
    try: os.kill(int(pid), 0); return True
    except ProcessLookupError: return False
    except Exception: return True


class Context:
    """Passed to every step: logging, progress, figures, metrics and sanity checks, all saved under reports/<step>/."""

    def __init__(self, project, step, echo=print):
        self.p, self.step, self.echo = project, step, echo
        self.dir = project.report_dir(step)
        for f in os.listdir(self.dir):                       # fresh report for this run
            if f.endswith(('.png', '.json')) or f.startswith('live_'): os.remove(os.path.join(self.dir, f))
        self._live_t, self._live_k = 0.0, 0
        os.makedirs(project.path('logs'), exist_ok=True)
        self.logf = open(project.path('logs', f'{step}.log'), 'w')
        self.metrics, self.checks, self.figs, self.t0 = {}, [], [], time.time()

    def log(self, msg):
        line = f'[{time.time() - self.t0:7.1f} s] {msg}'
        self.echo(line); self.logf.write(line + '\n'); self.logf.flush()

    def progress(self, frac, msg=''):
        self.echo(f'@@P {max(0.0, min(1.0, frac)):.4f} {msg}')

    def live_due(self, every=1.2):
        """True when it is time for a new live preview and the previous one is finished (previews never pile up)."""
        if os.environ.get('FLEXSFM_LIVE') == '0': return False     # previews off (e.g. headless batch runs)
        job = getattr(self, '_live_job', None)
        return time.time() - self._live_t >= every and (job is None or job.done())

    def live_async(self, make_image, caption):
        """Draw a preview in a background thread so the step itself does not slow down (decoding and drawing release the GIL)."""
        from concurrent.futures import ThreadPoolExecutor
        if not hasattr(self, '_live_pool'): self._live_pool = ThreadPoolExecutor(1, thread_name_prefix='live')
        self._live_t = time.time()
        def job():
            try: self.live(make_image(), caption)
            except Exception as e: self.echo(f'(live preview skipped: {e})')
        self._live_job = self._live_pool.submit(job)

    def live(self, img_bgr, caption=''):
        """Publish what the step is looking at right now: reports/<step>/live_NN.jpg (last 30 kept) + live.json, and '@@V' for the app."""
        try:
            import cv2
            self._live_t = time.time(); path = os.path.join(self.dir, f'live_{self._live_k % 30:02d}.jpg'); self._live_k += 1
            tmp = path + '.tmp.jpg'; cv2.imwrite(tmp, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85]); os.replace(tmp, path)
            info = dict(path=path, caption=caption, t=self._live_t, n=self._live_k)
            with open(os.path.join(self.dir, 'live_state.json.tmp'), 'w') as f: json.dump(info, f)
            os.replace(os.path.join(self.dir, 'live_state.json.tmp'), os.path.join(self.dir, 'live_state.json'))
            self.echo(f'@@V {path}\t{caption}')
        except Exception as e:
            self.echo(f'(live preview skipped: {e})')

    def live_scene(self, rec, stage, caption='', force=False, every=0.8, max_points=60000):
        """Publish the reconstruction as it is right now (cameras + a sample of points) for the app's live 3D view.
        Each call is one 'update' on the convergence plot (mean reprojection error)."""
        import numpy as np
        if os.environ.get('FLEXSFM_LIVE') == '0': return
        if not force and time.time() - getattr(self, '_scene_t', 0) < every: return
        try:
            ims = [im for im in rec.images.values() if im.has_pose]
            if not ims: return
            names = np.array([im.name for im in ims]); C = np.array([im.projection_center() for im in ims], np.float32)
            R = np.array([im.cam_from_world().rotation.matrix() for im in ims], np.float32)
            K = []
            for im in ims:
                cam = rec.cameras[im.camera_id]; f = cam.mean_focal_length(); K.append((cam.width, cam.height, f))
            pts = list(rec.points3D.values()); n = len(pts)
            sel = np.random.default_rng(len(ims)).choice(n, min(n, max_points), replace=False) if n > max_points else np.arange(n)
            X = np.array([pts[i].xyz for i in sel], np.float32).reshape(-1, 3); col = np.array([pts[i].color for i in sel], np.uint8).reshape(-1, 3)
            gps = getattr(self, 'live_gps', None)        # {image name: GPS position in the model's frame}, set once the model is in GPS metres
            G = np.array([gps.get(str(nm_), (np.nan,) * 3) for nm_ in names], np.float32).reshape(-1, 3) if gps else np.zeros((0, 3), np.float32)
            try: rec.update_point_3d_errors(); err = float(rec.compute_mean_reprojection_error())   # refresh: stored errors are stale after BA
            except Exception: err = float('nan')
            hist = getattr(self, '_scene_hist', []); hist.append((time.time() - self.t0, err, len(ims), n, stage)); self._scene_hist = hist
            path = os.path.join(self.dir, 'live_scene.npz'); tmp = os.path.join(self.dir, 'live_scene.tmp.npz')
            np.savez(tmp, names=names, C=C, R=R, K=np.array(K, np.float32), X=X, col=col, G=G, stage=stage, caption=caption,
                     hist_t=np.array([h[0] for h in hist]), hist_err=np.array([h[1] for h in hist]), hist_ncam=np.array([h[2] for h in hist]),
                     hist_npts=np.array([h[3] for h in hist]), hist_stage=np.array([h[4] for h in hist]))
            os.replace(tmp, path); self._scene_t = time.time()
            self.echo(f'@@S {path}\t{stage}: {caption}' if caption else f'@@S {path}\t{stage}')
        except Exception as e:
            self.echo(f'(live 3D skipped: {e})')

    def metric(self, key, value):
        self.metrics[key] = value

    def check(self, name, ok, value='', expect='', fix='', level=None):
        """Sanity check. level: 'pass' | 'warn' | 'fail' (default from ok). fix = what to do when it is not 'pass'."""
        lvl = level or ('pass' if ok else 'fail')
        self.checks.append(dict(name=name, level=lvl, value=str(value), expect=str(expect), fix=fix if lvl != 'pass' else ''))
        self.echo(f'@@C {lvl.upper():4s} {name}: {value} (expected {expect})')

    def fig(self, fig, name, caption=''):
        """Save a matplotlib figure into the report with a caption shown in the GUI."""
        import matplotlib.pyplot as plt
        path = os.path.join(self.dir, f'{len(self.figs) + 1:02d}_{name}.png')
        fig.savefig(path, dpi=100, bbox_inches='tight'); plt.close(fig)
        self.figs.append(dict(file=os.path.basename(path), caption=caption)); self.echo(f'@@F {path}')
        return path

    def save(self):
        json.dump(self.metrics, open(os.path.join(self.dir, 'metrics.json'), 'w'), indent=1, default=str)
        json.dump(self.checks, open(os.path.join(self.dir, 'checks.json'), 'w'), indent=1)
        json.dump(self.figs, open(os.path.join(self.dir, 'figures.json'), 'w'), indent=1)


def run_step(project, step, echo=print):
    """Run one step with its saved settings; records status, timing, metrics and checks in state.json."""
    mod = step_module(step)
    for need in mod.NEEDS:
        if project.status(need) not in ('done', 'stale') and need not in OPTIONAL:
            raise RuntimeError(f'"{step}" needs "{need}" to be run first')
    ctx = Context(project, step, echo)
    prm = project.params(step)
    st = project.state.setdefault('steps', {}).setdefault(step, {})
    st.update(status='running', started=time.time(), pid=os.getpid()); st.pop('error', None); project.save_state()
    ctx.log(f'=== {mod.TITLE} ===')
    import sys, platform
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ctx.log(f'process {os.getpid()}: python -m flexsfm run {project.root} {step}   (Python {platform.python_version()})')
    ctx.log(f'code: {os.path.relpath(mod.__file__, pkg_root)}')
    ctx.log('settings: ' + json.dumps(prm, default=str))
    try:
        mod.run(project, prm, ctx)
        project.reload()
        st = project.state['steps'].setdefault(step, {})
        n_fail = sum(c['level'] == 'fail' for c in ctx.checks); n_warn = sum(c['level'] == 'warn' for c in ctx.checks)
        st.update(status='done', finished=time.time(), params_hash=project.params_hash(step), metrics=ctx.metrics,
                  checks=dict(fail=n_fail, warn=n_warn, ok=len(ctx.checks) - n_fail - n_warn), seconds=round(time.time() - ctx.t0, 1))
        ctx.save(); project.save_state()
        ctx.log(f'done in {time.time() - ctx.t0:.0f} s - checks: {n_fail} failed, {n_warn} warnings')
    except BaseException as e:
        project.reload(); st = project.state['steps'].setdefault(step, {})
        st.update(status='failed', finished=time.time(), error=f'{type(e).__name__}: {e}'); project.save_state(); ctx.save()
        ctx.log('FAILED: ' + ''.join(traceback.format_exception(e)))
        raise
    finally:
        ctx.logf.close()
