"""Step 4: FlexGlue matching + geometric verification -> COLMAP database."""
import os, time
import numpy as np
from ..common import P, plt, FeatureStore, fg_match, load_image, draw_matches, live_pair

TITLE = '4  Matching'
WHAT = """Matches every pair with FlexGlue, then COLMAP checks each pair's matches against two-view geometry (RANSAC)
and keeps only the consistent ones ('inliers').
- 'link strength' below: for every point in the sequence, how many inlier matches connect frames before it with frames
  after it. A weak link is where a reconstruction can break into pieces or bend.
- Camera: one shared camera (same device, same zoom) is best. RADIAL = focal + 2 distortion terms, square pixels.
  The starting focal length comes from the images' EXIF when they have it; mixed image sizes get one camera per size.
Check: no frame without verified pairs, no weak links, most pairs verified."""
NEEDS = ['pairs']
PARAMS = [
    P('min_covisibility', 0.3, 'FlexGlue covisibility score below which a pair is dropped (overlap estimate, 0..1)', lo=0.0, hi=1.0),
    P('min_matches', 15, 'Pairs with fewer FlexGlue matches are dropped', lo=8),
    P('snap_px', 1.0, 'Match precision: max distance (px) between FlexGlue\'s predicted point and the keypoint it is snapped to. '
      'Smaller = more precise tracks (sharper), fewer matches. FlexGlue default is 3.', lo=0.3, hi=5.0),
    P('ransac_px', 2.0, 'Geometric verification tolerance (pixels at full resolution)', lo=0.5, hi=20.0),
    P('camera_model', 'RADIAL', 'COLMAP camera model. RADIAL suits most video and photos; OPENCV adds tangential + separate fx/fy; use OPENCV_FISHEYE-like models only for fisheye lenses.',
      choices=['SIMPLE_RADIAL', 'RADIAL', 'OPENCV', 'PINHOLE', 'SIMPLE_PINHOLE', 'FULL_OPENCV']),
    P('single_camera', True, 'All images share one camera (same device and zoom). Ignored when image sizes differ.'),
    P('focal_35mm', 24.0, 'Initial focal length, 35 mm equivalent (phones and drones are often 24-26). Used only when the images have no EXIF focal length. Refined during reconstruction.', lo=5.0, hi=500.0),
    P('focal_px', 0.0, 'Initial focal length in pixels; overrides the 35 mm value when > 0', lo=0.0, advanced=True),
    P('weak_link', 150, 'Link strength (inliers across a point in the sequence) below which a link is reported as weak', lo=1),
]


def _scaled(kp, name, p):
    """Keypoints in the size load_image(..., 1100) returns (live preview only)."""
    from PIL import Image
    W, H = Image.open(os.path.join(p.images_dir, name)).size; s = min(1.0, 1100 / max(W, H)); return kp * s


def run(p, prm, ctx):
    import pycolmap
    names = p.names(); N = len(names); idx = {n: i for i, n in enumerate(names)}
    pairs = [tuple(sorted((idx[l.split()[0]], idx[l.split()[1]]))) for l in open(p.path('pairs.txt')) if l.strip()]
    fs = FeatureStore(p.path('features')); res, cov = {}, np.zeros(len(pairs)); t1 = time.time()
    for k, (i, j) in enumerate(pairs):
        m, c = fg_match(fs, names[i], names[j], snap_px=prm['snap_px']); cov[k] = c
        if c >= prm['min_covisibility'] and len(m) >= prm['min_matches']: res[(i, j)] = m
        if ctx.live_due(1.5):
            ok = c >= prm['min_covisibility'] and len(m) >= prm['min_matches']
            na_, nb_, ka_, kb_, m_ = names[i], names[j], fs.kp(names[i]), fs.kp(names[j]), m.copy()
            ctx.live_async(lambda na_=na_, nb_=nb_, ka_=ka_, kb_=kb_, m_=m_: live_pair(load_image(os.path.join(p.images_dir, na_), 1100), load_image(os.path.join(p.images_dir, nb_), 1100),
                                                   _scaled(ka_, na_, p), _scaled(kb_, nb_, p), m_),
                     f'pair {k + 1} / {len(pairs)}: {names[i]} <-> {names[j]} ({j - i} frames apart) - {len(m)} FlexGlue matches, '
                     f'covisibility {c:.2f} - {"kept" if ok else "dropped"}')
        if k % 500 == 0:
            r = (k + 1) / (time.time() - t1); ctx.progress(0.8 * k / len(pairs), f'matching {k}/{len(pairs)}')
            ctx.log(f'matched {k}/{len(pairs)} ({r:.1f} pairs/s, ~{(len(pairs) - k) / r / 60:.0f} min left), kept {len(res)}')
    np.savez(p.path('matches.npz'), pairs=np.array(list(res.keys())), m=np.array(list(res.values()), dtype=object), covis=cov, all_pairs=np.array(pairs))
    ctx.log(f'{len(res)} of {len(pairs)} pairs kept after FlexGlue')
    # ---- database
    db = p.path('database.db')
    for f in (db, db + '-wal', db + '-shm'):
        if os.path.exists(f): os.remove(f)
    from PIL import Image
    im0 = Image.open(os.path.join(p.images_dir, names[0])); W, H = im0.size
    sizes = {Image.open(os.path.join(p.images_dir, n)).size for n in names[:: max(1, len(names) // 50)]}
    single = bool(prm['single_camera']) and len(sizes) == 1
    if prm['single_camera'] and not single: ctx.log(f'images have {len(sizes)} different sizes - using one camera per size')
    f35 = 0.0
    try: f35 = float(im0.getexif().get_ifd(0x8769).get(41989) or 0)        # EXIF FocalLengthIn35mmFilm
    except Exception: pass
    if prm['focal_px'] > 0: f0 = prm['focal_px']; src = 'setting focal_px'
    elif f35 > 0: f0 = f35 / 36.0 * max(W, H); src = f'EXIF ({f35:.0f} mm equivalent)'
    else: f0 = prm['focal_35mm'] / 36.0 * max(W, H); src = f'setting focal_35mm ({prm["focal_35mm"]} mm)'
    ctx.log(f'starting focal length {f0:.0f} px from {src}'); ctx.metric('start_focal_px', round(f0, 1))
    model = getattr(pycolmap.CameraModelId, prm['camera_model']); nparams = len(pycolmap.Camera(model=prm['camera_model'], width=W, height=H).params)
    base = {'SIMPLE_PINHOLE': [f0, W / 2, H / 2], 'PINHOLE': [f0, f0, W / 2, H / 2], 'SIMPLE_RADIAL': [f0, W / 2, H / 2, 0],
            'RADIAL': [f0, W / 2, H / 2, 0, 0], 'OPENCV': [f0, f0, W / 2, H / 2, 0, 0, 0, 0]}.get(prm['camera_model'])
    if base is None: base = [f0, f0, W / 2, H / 2] + [0] * (nparams - 4)
    opt = pycolmap.ImageReaderOptions(); opt.camera_model = prm['camera_model']; opt.camera_params = ','.join(f'{v:.6f}' for v in base)
    pycolmap.Database.open(db).close()
    pycolmap.import_images(db, p.images_dir, camera_mode=pycolmap.CameraMode.SINGLE if single else pycolmap.CameraMode.AUTO,
                           image_names=names, options=opt)
    with pycolmap.Database.open(db) as d:
        ids = {im.name: im.image_id for im in d.read_all_images()}
        for n in names: d.write_keypoints(ids[n], fs.kp(n) + 0.5)          # COLMAP pixel centres at +0.5
        for (i, j), m in res.items(): d.write_matches(ids[names[i]], ids[names[j]], m)
    with open(p.path('pairs_matched.txt'), 'w') as f: f.writelines(f'{names[i]} {names[j]}\n' for i, j in res)
    ctx.progress(0.85, 'geometric verification'); ctx.log('geometric verification')
    vo = pycolmap.TwoViewGeometryOptions(); vo.ransac.max_error = prm['ransac_px']
    pycolmap.verify_matches(db, p.path('pairs_matched.txt'), vo)
    inl = {}
    with pycolmap.Database.open(db) as d:
        iid = {im.image_id: im.name for im in d.read_all_images()}
        pids, tvgs = d.read_two_view_geometries()
        for pid, t in zip(pids, tvgs):
            a, b = pycolmap.pair_id_to_image_pair(pid); n = len(t.inlier_matches)
            if n: inl[tuple(sorted((idx[iid[a]], idx[iid[b]])))] = n
    np.savez(p.path('pair_inliers.npz'), pairs=np.array(list(inl.keys())), n=np.array(list(inl.values())))
    raw = {k: len(v) for k, v in res.items()}
    # ---- link strength across every cut of the sequence (local pairs only)
    link = np.zeros(max(N - 1, 1)); link_all = np.zeros(max(N - 1, 1))
    for (i, j), n in inl.items():
        link_all[i:j] += n                          # every pair crossing the cut (incl. revisits / neighbouring survey lines)
        if j - i <= 30: link[i:j] += n              # only nearby frames: shows local weak spots along a video
    deg = np.zeros(N, int)
    for (i, j) in inl: deg[i] += 1; deg[j] += 1
    broken = np.flatnonzero(link_all < prm['weak_link'])                 # weak even counting all pairs: can really split
    weak = np.flatnonzero((link < prm['weak_link']) & (link_all >= prm['weak_link']))   # weak locally only: held by longer pairs
    ctx.metric('pairs_matched', len(res)); ctx.metric('pairs_verified', len(inl)); ctx.metric('inliers_median', int(np.median(list(inl.values()))) if inl else 0)
    ctx.metric('isolated_frames', int((deg == 0).sum())); ctx.metric('weak_links', int(len(broken))); ctx.metric('locally_weak_links', int(len(weak)))
    ctx.metric('weakest_link', int(link_all.min()) if N > 1 else 0)
    ctx.metric('initial_focal_px', round(f0, 1))
    ctx.check('pairs verified', len(inl) > 0.6 * max(1, len(res)), f'{len(inl)}/{len(res)}', '> 60 %',
              'Many pairs fail verification: raise "ransac px" (rolling shutter / lens distortion) or check the camera model.',
              level=None if len(inl) > 0.6 * max(1, len(res)) else 'warn')
    ctx.check('no isolated frames', (deg == 0).sum() == 0, f'{(deg == 0).sum()} frames without any verified pair', '0',
              'Isolated frames cannot be placed. See the log for names; check them (blur, sky) or add pairs in pairs_add.txt.')
    ctx.check('no weak links in the sequence', len(broken) == 0, f'{len(broken)} weak cuts, weakest {int(link_all.min()) if N > 1 else 0} inliers', f'>= {prm["weak_link"]}',
              'Weak links can split or bend the model. Raise "sequential" in step 3, or lower "min covisibility".',
              level=None if len(broken) == 0 else ('fail' if (link_all == 0).any() else 'warn'))
    if len(weak) and not len(broken):
        ctx.log(f'{len(weak)} cuts are weak between nearby frames but held together by longer pairs (other passes / survey lines) - fine')
    if (deg == 0).any(): ctx.log('isolated: ' + ', '.join(np.array(names)[deg == 0][:50]))
    if len(broken): ctx.log('weak links after frames: ' + ', '.join(names[i] for i in broken[:50]))
    # ---- figures
    fig, ax = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    ax[0].semilogy(np.arange(N - 1) + 0.5, np.maximum(link_all, 1), label='all pairs'); ax[0].semilogy(np.arange(N - 1) + 0.5, np.maximum(link, 1), lw=.8, alpha=.7, label='nearby frames only')
    ax[0].axhline(prm['weak_link'], c='r', ls='--', lw=.8); ax[0].legend(fontsize=8)
    ax[0].set_ylabel('inliers across the cut'); ax[0].set_title('link strength along the sequence (dips = weak links)'); ax[0].grid(alpha=.3)
    ax[1].plot(deg); ax[1].set_ylabel('verified pairs'); ax[1].set_xlabel('frame'); ax[1].grid(alpha=.3)
    ctx.fig(fig, 'link_strength', 'Link strength: how strongly each point of the sequence is tied to its neighbours')
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    ax[0].hist(cov, bins=50); ax[0].axvline(prm['min_covisibility'], c='r', ls='--'); ax[0].set_title('FlexGlue covisibility of all pairs')
    v = np.array([inl.get(k, 0) for k in raw]); r = np.array(list(raw.values()))
    ax[1].hist(v, bins=60); ax[1].set_title('verified inliers per pair')
    ax[2].scatter(r, v, s=2); ax[2].plot([0, r.max() if len(r) else 1], [0, r.max() if len(r) else 1], 'k:'); ax[2].set_xlabel('FlexGlue matches'); ax[2].set_ylabel('inliers')
    ax[2].set_title('matches vs inliers (far below the line = rejected by geometry)')
    ctx.fig(fig, 'match_stats', 'Matching statistics')
    if inl:
        ks = sorted(inl, key=lambda k: inl[k]); sel = [ks[len(ks) // 2], ks[0]]
        with pycolmap.Database.open(db) as d:
            for tag, (i, j) in zip(('typical', 'weakest'), sel):
                a, b = ids[names[i]], ids[names[j]]; t = d.read_two_view_geometry(a, b); m = t.inlier_matches
                ka, kb = fs.kp(names[i]), fs.kp(names[j])
                fig, ax = plt.subplots(figsize=(10, 11)); draw_matches(ax, load_image(os.path.join(p.images_dir, names[i])), load_image(os.path.join(p.images_dir, names[j])), ka, kb, m, '#30ff60')
                ax.set_title(f'{tag} verified pair: {names[i]} - {names[j]}: {len(m)} inliers'); ctx.fig(fig, f'pair_{tag}', f'{tag.capitalize()} verified pair (green = inliers)')
