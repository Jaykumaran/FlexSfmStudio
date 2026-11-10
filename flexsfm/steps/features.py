"""Step 2: keypoints + FlexGlue descriptors."""
import os
import numpy as np
from ..common import P, plt, device, load_image, mask_for, live_keypoints

TITLE = '2  Keypoints'
WHAT = """Finds distinctive points in every image and computes FlexGlue's dense descriptors used for matching.
- detector: SIFT at full resolution places points most precisely (about 0.5 px; ALIKED about 0.75-0.85 px, measured with
  FlexGlue on 4K video frames). Precise points = sharper splats (less ghosting on signs/edges). ALIKED finds more points on weak
  texture; 'sift+aliked' uses both.
- 'long side' is the resolution keypoints are found at (full resolution recommended for SIFT).
- Keypoints on masked (black) pixels are dropped.
- A small global descriptor per image is kept to find places seen again (loop closures) when there is no GPS.
Check: every frame should have plenty of keypoints; frames with few are blurred, over-exposed, sky or mostly masked."""
NEEDS = ['ingest']
PARAMS = [
    P('detector', 'sift', 'sift = most precise (recommended); aliked = most robust on weak texture; sift+aliked = both', choices=['sift', 'aliked', 'sift+aliked']),
    P('long_side', 3840, 'Resolution (longest side, px) keypoints are detected at. Full resolution gives the most precise points.', lo=320, hi=8192),
    P('max_keypoints', 6000, 'Keypoints kept per image (matching time grows with this)', lo=256, hi=32768),
    P('threshold', 0.01, 'ALIKED detection threshold (lower = more, weaker points)', lo=0.0, hi=1.0, advanced=True),
    P('redo', False, 'Recompute features that already exist (otherwise existing ones are reused)'),
]


def run(p, prm, ctx):
    import torch
    from ..common import flexglue, detect
    names = p.names(); fd = p.path('features'); os.makedirs(fd, exist_ok=True); dev = device()
    ctx.log(f'device: {dev}' + ('' if dev == 'cuda' else ' (no GPU - this will be slow)'))
    tag = f"{prm['detector']}_{prm['long_side']}_{prm['max_keypoints']}_{prm['threshold']}"
    tagf = os.path.join(fd, 'settings.txt')
    if os.path.exists(tagf) and open(tagf).read().strip() != tag: prm = dict(prm, redo=True); ctx.log('settings changed - recomputing all features')
    todo = [n for n in names if prm['redo'] or not os.path.exists(os.path.join(fd, n + '.pt'))]
    fg = flexglue(True) if todo else None
    masked_frac = []
    for k, n in enumerate(todo):
        img = load_image(os.path.join(p.images_dir, n)); H, W = img.shape[:2]
        kp = detect(img, prm['detector'], prm['long_side'], prm['max_keypoints'], prm['threshold'])
        t = torch.from_numpy(img.copy()).permute(2, 0, 1).float().div(255)[None].to(dev)
        with torch.no_grad():
            f = fg.descriptor({'image': t}, resize=fg.conf.describe_resize)['features']
        m = mask_for(p, n, (W, H))
        if m is not None:
            ix = np.clip(kp.astype(int), 0, [W - 1, H - 1]); keep = m[ix[:, 1], ix[:, 0]]; masked_frac.append(1 - keep.mean()); kp = kp[keep]
        g = torch.nn.functional.normalize(f.float().mean((2, 3)), dim=1)[0].cpu().numpy()
        np.save(os.path.join(fd, n + '.kp.npy'), kp.astype(np.float32)); np.save(os.path.join(fd, n + '.g.npy'), g.astype(np.float32))
        torch.save({'features': f.half().cpu(), 'size': (W, H)}, os.path.join(fd, n + '.pt'))
        if ctx.live_due(): ctx.live_async(lambda img=img, kp=kp, m=m: live_keypoints(img, kp, m), f'image {k + 1} / {len(todo)}: {n} - {len(kp)} keypoints ({prm["detector"]})'
                                    + (f', {100 * (1 - m.mean()):.0f} % masked (red)' if m is not None else ''))
        if k % 50 == 0: ctx.progress(k / len(todo), f'keypoints {k}/{len(todo)}'); ctx.log(f'{k}/{len(todo)} {n}: {len(kp)} keypoints')
        del t
    open(tagf, 'w').write(tag)
    cnt = np.array([len(np.load(os.path.join(fd, n + '.kp.npy'))) for n in names])
    gl = np.stack([np.load(os.path.join(fd, n + '.g.npy')) for n in names]); np.save(p.path('globals.npy'), gl)
    low = cnt < 0.25 * prm['max_keypoints']
    ctx.metric('keypoints_median', int(np.median(cnt))); ctx.metric('keypoints_min', int(cnt.min())); ctx.metric('weak_frames', int(low.sum()))
    ctx.check('every frame has keypoints', low.mean() < 0.02, f'{low.sum()} frames below 25 % of max', '< 2 % of frames',
              'Weak frames: blur, sky, over-exposure or large masks. Lower "threshold" or check those frames (list in the log).',
              level=None if low.mean() < 0.02 else 'warn')
    if low.any(): ctx.log('weak frames: ' + ', '.join(np.array(names)[low][:50]))
    if masked_frac: ctx.metric('masked_keypoint_fraction', round(float(np.mean(masked_frac)), 3))
    # figures
    fig, ax = plt.subplots(figsize=(16, 3.5)); ax.plot(cnt, lw=.8); ax.axhline(0.25 * prm['max_keypoints'], c='r', ls='--', lw=.8)
    ax.set_xlabel('frame index'); ax.set_ylabel('keypoints'); ax.grid(alpha=.3)
    ctx.fig(fig, 'keypoints_per_frame', 'Keypoints per frame (red = weak-frame limit)')
    idx = [int(np.argmin(cnt)), len(names) // 2]
    fig, ax = plt.subplots(1, 2, figsize=(18, 5.5))
    for a, i in zip(ax, idx):
        img = load_image(os.path.join(p.images_dir, names[i])); kp = np.load(os.path.join(fd, names[i] + '.kp.npy'))
        a.imshow(img); a.scatter(kp[:, 0], kp[:, 1], s=1, c='#00ff66'); a.axis('off'); a.set_title(f'{names[i]}: {len(kp)} keypoints' + (' (weakest frame)' if i == idx[0] else ''))
        m = mask_for(p, names[i], (img.shape[1], img.shape[0]))
        if m is not None: a.imshow(np.ma.masked_where(m, np.ones_like(m)), cmap='autumn', alpha=.35)
    ctx.fig(fig, 'keypoints_sample', 'Keypoints on the weakest frame and a middle frame (orange = masked out)')
    sub = np.linspace(0, len(names) - 1, min(len(names), 400)).astype(int)
    fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(gl[sub] @ gl[sub].T, cmap='magma', vmin=0.5, vmax=1, extent=(sub[0], sub[-1], sub[-1], sub[0]))
    plt.colorbar(im, ax=ax); ax.set_title('image similarity (global descriptor)'); ax.set_xlabel('frame'); ax.set_ylabel('frame')
    ctx.fig(fig, 'similarity', 'Bright off-diagonal bands = the same place seen again (used to find revisits without GPS)')
