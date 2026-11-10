"""Step 1: frames + GPS."""
import glob, os, shutil, subprocess
import numpy as np
from ..common import P, plt, IMG_EXT, plan_axes, route_plot, load_image, live_keypoints
from .. import gps as G

TITLE = '1  Frames and GPS'
WHAT = """Collects the images and their positions.
- Source is a video (frames are extracted every 'step' frames) or a folder of images (used in place, not copied).
- GPS comes from the images' EXIF, from a DJI .SRT next to a drone video, from a CSV (name + lat/lon/alt or x/y/z), or there is none.
  Without GPS the reconstruction still works; it is then in arbitrary scale and orientation unless you add markers.
- Masks (optional): a folder with one PNG per image, black = ignore (sky, parts of the rig, moving people / cars).
- Passes: with GPS the trajectory is split where it turns back, so later steps can check that passes flown in
  opposite directions line up (the 'doubled lane' problem)."""
NEEDS = []
PARAMS = [
    P('source', '', 'Video file (.MP4) or folder of images', kind='path', label='video or image folder'),
    P('step', 15, 'Video only: keep every Nth frame (15 at 30 fps = 2 frames/s). Also used to map SRT lines to frames of an image folder.', lo=1, hi=300),
    P('max_frames', 0, 'Use only the first N frames (0 = all). Handy for a quick trial run.', lo=0),
    P('gps_source', 'auto', 'Where positions come from. auto = SRT next to the video / given SRT, else EXIF, else CSV, else none.',
      choices=['auto', 'srt', 'exif', 'csv', 'none']),
    P('srt', '', 'DJI .SRT file (auto-detected next to the video)', kind='file'),
    P('csv', '', 'CSV with image name and lat/lon/alt (or x/y/z metres), e.g. metashape_reference.csv', kind='file'),
    P('ignore_gps_altitude', False, 'Set when the altitude column is unusable (e.g. all zeros): heights then come from the images only.'),
    P('masks', '', 'Folder of masks (PNG, black = ignore). Leave empty for none. "<images>/masks" is used automatically if present.', kind='dir'),
    P('jpeg_quality', 2, 'Video extraction quality for ffmpeg -q:v (2 = best, 31 = worst)', lo=1, hi=31, advanced=True),
]


def ffmpeg_exe():
    """System ffmpeg if installed, else the one bundled with the imageio-ffmpeg package."""
    import shutil
    exe = shutil.which('ffmpeg')
    if exe: return exe
    try:
        import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError('ffmpeg not found: install it (e.g. sudo apt install ffmpeg) or pip install imageio-ffmpeg')


def _extract(video, out, step, max_frames, q, ctx):
    os.makedirs(out, exist_ok=True)
    have = sorted(glob.glob(os.path.join(out, 'frame_*.jpg')))
    if have and (not max_frames or len(have) >= max_frames):
        ctx.log(f'{len(have)} frames already extracted - reusing (delete frames/ to extract again)'); return
    cmd = [ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-stats', '-i', video, '-vf', f"select=not(mod(n\\,{step}))",
           '-vsync', 'vfr', '-q:v', str(q), '-start_number', '1']
    if max_frames: cmd += ['-frames:v', str(max_frames)]
    cmd += [os.path.join(out, 'frame_%06d.jpg')]
    ctx.log('extracting frames: ' + ' '.join(cmd))
    pr = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, bufsize=1)
    for line in iter(pr.stderr.readline, ''):
        if 'frame=' in line: ctx.log(line.strip()[:120])
        if ctx.live_due(1.5):                       # show the newest extracted frame
            fs_ = sorted(glob.glob(os.path.join(out, 'frame_*.jpg')))
            if len(fs_) > 1:
                try: ctx.live_async(lambda f_=fs_[-2]: live_keypoints(load_image(f_, 1280), np.zeros((0, 2))), f'extracting frames: {len(fs_) - 1} so far - {os.path.basename(fs_[-2])}')
                except Exception: pass
    if pr.wait() != 0: raise RuntimeError('ffmpeg failed')


def run(p, prm, ctx):
    src = os.path.expanduser(prm['source'])
    if not src or not os.path.exists(src): raise FileNotFoundError(f'source not found: "{src}" - set "video or image folder"')
    is_video = os.path.isfile(src)
    if is_video:
        img_dir = p.path('frames'); _extract(src, img_dir, prm['step'], prm['max_frames'], prm['jpeg_quality'], ctx)
    else:
        img_dir = src
    names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXT) and not f.startswith('.'))
    if prm['max_frames']: names = names[:prm['max_frames']]
    if len(names) < 3: raise RuntimeError(f'only {len(names)} images in {img_dir}')
    with open(p.path('images.txt'), 'w') as f: f.writelines(n + '\n' for n in names)
    masks = os.path.expanduser(prm['masks']) or (os.path.join(img_dir, 'masks') if os.path.isdir(os.path.join(img_dir, 'masks')) else '')
    p.state['images_dir'] = img_dir; p.state['masks_dir'] = masks; p.save_state()
    ctx.log(f'{len(names)} images in {img_dir}' + (f', masks from {masks}' if masks else ', no masks'))
    from PIL import Image
    sizes = {}
    for n in names[:: max(1, len(names) // 50)]:
        sizes[Image.open(os.path.join(img_dir, n)).size] = sizes.get(Image.open(os.path.join(img_dir, n)).size, 0) + 1
    W, H = max(sizes, key=sizes.get); ctx.metric('images', len(names)); ctx.metric('image_size', f'{W}x{H}')
    ctx.check('all images the same size', len(sizes) == 1, ', '.join(f'{a}x{b}' for a, b in sizes), 'one size',
              'Mixed sizes: set "single camera" off in step 4 (match) so each size gets its own camera.', level=None if len(sizes) == 1 else 'warn')
    if masks:
        nm = sum(os.path.exists(os.path.join(masks, os.path.splitext(n)[0] + '.png')) for n in names)
        ctx.metric('masks_found', nm)
        ctx.check('masks found for the images', nm >= 0.9 * len(names), f'{nm}/{len(names)}', '>= 90 %',
                  'Mask files must be named like the images, with .png. Check the masks folder.', level=None if nm >= 0.9 * len(names) else 'warn')

    # ---------------------------------------------------------------- GPS
    gs = prm['gps_source']; srt = os.path.expanduser(prm['srt'])
    if not srt and is_video:
        for c in (os.path.splitext(src)[0] + e for e in ('.SRT', '.srt')):
            if os.path.exists(c): srt = c
    csvp = os.path.expanduser(prm['csv'])
    if not csvp and not is_video and os.path.exists(os.path.join(img_dir, 'metashape_reference.csv')): csvp = os.path.join(img_dir, 'metashape_reference.csv')
    lla = None; valid = np.zeros(len(names), bool); used = 'none'; geo = True
    order = {'auto': ['srt', 'exif', 'csv'], 'none': []}.get(gs, [gs])
    for s in order:
        if s == 'srt' and srt and os.path.exists(srt):
            lla, valid = G.srt_for_frames(names, srt, prm['step']); used = f'SRT {os.path.basename(srt)}'
        elif s == 'exif':
            v = [G.exif_gps(os.path.join(img_dir, n)) for n in names]
            if sum(x is not None for x in v) > 0.5 * len(names):
                lla = np.array([x if x else (np.nan,) * 3 for x in v], float); valid = ~np.isnan(lla[:, 0]); used = 'EXIF'
        elif s == 'csv' and csvp and os.path.exists(csvp):
            lla, valid, geo = G.read_csv(csvp, names); used = f'CSV {os.path.basename(csvp)}'
        if lla is not None and valid.sum() >= 3: break
        lla = None; valid = np.zeros(len(names), bool)
    ctx.metric('gps_source', used)
    if lla is None:
        if os.path.exists(p.path('gps.npz')): os.remove(p.path('gps.npz'))
        ctx.log('no GPS: the model will be in arbitrary scale/orientation (add markers to fix scale and orientation)')
        ctx.check('GPS available', True, 'none', 'optional', level='warn',
                  fix='Without GPS: scale and orientation are arbitrary. Use step 6 (markers) with 3+ known points or a known distance, if you need metres.')
        fig, ax = plt.subplots(1, 4, figsize=(18, 3.2))
        for a, n in zip(ax, names[:: max(1, len(names) // 4)][:4]): a.imshow(load_image(os.path.join(img_dir, n), 640)); a.set_title(n, fontsize=8); a.axis('off')
        ctx.fig(fig, 'sample_frames', 'Sample frames')
        return
    if geo:
        enu, ref = G.lla_to_enu(lla)
    else:
        enu, ref = lla - np.nanmin(lla, 0) * [1, 1, 0], dict(metric_csv=True)
    if prm['ignore_gps_altitude'] or (valid.any() and np.nanstd(lla[valid, 2]) < 1e-6):
        if not prm['ignore_gps_altitude']: ctx.log('GPS altitude is constant - treated as unknown')
        ref['no_altitude'] = True
    pid, ext = G.detect_passes(enu, valid)
    np.savez(p.path('gps.npz'), names=np.array(names), lla=lla, enu=enu, valid=valid, ref=np.array(ref, dtype=object), pass_id=pid)
    step_len = np.linalg.norm(np.diff(enu[valid][:, :2], axis=0), axis=1)
    jump_k = 5 if is_video else 8             # photo surveys have bigger, more irregular gaps (line changes) than video
    jumps = int((step_len > max(jump_k * np.median(step_len), 3.0)).sum()) if len(step_len) else 0
    ctx.metric('gps_valid', int(valid.sum())); ctx.metric('track_length_m', round(float(step_len.sum()), 1))
    ctx.metric('passes', len(ext) - 1); ctx.metric('pass_ranges', [[names[ext[k]], names[ext[k + 1] - 1]] for k in range(len(ext) - 1)])
    ctx.metric('median_spacing_m', round(float(np.median(step_len)), 3) if len(step_len) else 0)
    ctx.check('GPS for every frame', valid.mean() > 0.98, f'{valid.sum()}/{len(names)}', '> 98 %',
              'Missing GPS: check the SRT matches the video, and "step" equals the extraction step.')
    ctx.check('no GPS jumps', jumps == 0, f'{jumps} jumps', '0', 'GPS jumps (multipath/glitches) are down-weighted later; check the track figure.',
              level=None if jumps == 0 else 'warn')
    if is_video:                             # only meaningful for video: photos of a survey are normally tens of metres apart
        ctx.check('frames not too far apart', np.median(step_len) < 3.0 if len(step_len) else True, f'{np.median(step_len):.2f} m median', '< 3 m',
                  'Frames far apart overlap little: use a smaller "step".', level=None if not len(step_len) or np.median(step_len) < 3.0 else 'warn')
    still = float((step_len < (0.05 if is_video else 0.5)).mean()) if len(step_len) else 0
    ctx.check('camera moving', still < 0.3, f'{100 * still:.0f} % of frames nearly static', '< 30 %',
              'Long hovers add many near-identical frames; fine, but you can raise "step".', level=None if still < 0.3 else 'warn')
    # figures
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.5))
    route_plot(ax[0], enu[valid], np.flatnonzero(valid), 'frame index'); plan_axes(ax[0])
    for k in range(0, len(names), max(1, len(names) // 15)): ax[0].annotate(str(k + 1), enu[k, :2], fontsize=7)
    ax[0].set_title(f'GPS track ({used}), colour = frame order')
    for k in range(len(ext) - 1):
        s = slice(ext[k], ext[k + 1]); ax[1].plot(enu[s, 0], enu[s, 1], '-', lw=2, label=f'pass {k + 1}: frames {ext[k] + 1}-{ext[k + 1]}')
    plan_axes(ax[1]); ax[1].legend(fontsize=8); ax[1].set_title('passes (split where the camera turns back)')
    ctx.fig(fig, 'gps_track', 'GPS track and passes. Passes in opposite directions must line up later (diagnose step).')
    fig, ax = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    ax[0].plot(enu[:, 2]); ax[0].set_ylabel('height (m)'); ax[0].grid(alpha=.3)
    ax[1].plot(np.r_[0, np.linalg.norm(np.diff(enu[:, :2], axis=0), axis=1)]); ax[1].set_ylabel('m between frames'); ax[1].set_xlabel('frame index'); ax[1].grid(alpha=.3)
    for b in ext[1:-1]:
        for a in ax: a.axvline(b, c='r', ls='--', lw=.8)
    ctx.fig(fig, 'gps_height_speed', 'Height and spacing between frames along the sequence (red = turn-backs)')
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.2))
    for a, n in zip(ax, names[:: max(1, len(names) // 4)][:4]): a.imshow(load_image(os.path.join(img_dir, n), 640)); a.set_title(n, fontsize=8); a.axis('off')
    ctx.fig(fig, 'sample_frames', 'Sample frames')
