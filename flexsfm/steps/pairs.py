"""Step 3: which images to match."""
import os
import numpy as np
from ..common import P, plt

TITLE = '3  Image pairs'
WHAT = """Decides which image pairs get matched. Matching every pair is too slow, so pairs come from:
- sequence: each frame with its next N frames (video order),
- GPS: frames within a radius, including other passes over the same place (if GPS exists),
- similarity: the most similar images by global descriptor, far apart in time - this finds revisits without GPS
  and pairs looking at the same place from opposite directions.
- your own additions/removals: pairs_add.txt / pairs_remove.txt in the project folder (one 'nameA nameB' per line).
Check: every frame needs pairs; revisits (other passes) should appear as off-diagonal bands in the pair matrix."""
NEEDS = ['features']
PARAMS = [
    P('sequential', 8, 'Pair each frame with the next N frames', lo=0, hi=100),
    P('gps_radius_m', 30.0, 'GPS: pair frames closer than this (metres). 0 = off', lo=0.0),
    P('gps_neighbours', 20, 'GPS: at most this many nearest frames per frame', lo=0, hi=200),
    P('similar', 10, 'Similarity: pair each frame with its N most similar frames that are far apart in time (0 = off)', lo=0, hi=100),
    P('similar_min_gap', 30, 'Similarity: only frames at least this many frames apart (revisits, not neighbours)', lo=1),
    P('similar_min_score', 0.75, 'Similarity: minimum cosine score of the global descriptors', lo=0.0, hi=1.0, advanced=True),
]


def _names_to_pairs(path, idx):
    out = set()
    if os.path.exists(path):
        for l in open(path):
            t = l.split()
            if len(t) >= 2 and t[0] in idx and t[1] in idx: out.add(tuple(sorted((idx[t[0]], idx[t[1]]))))
    return out


def run(p, prm, ctx):
    from scipy.spatial import cKDTree
    names = p.names(); N = len(names); idx = {n: i for i, n in enumerate(names)}
    src = {}
    def add(i, j, s):
        if i != j: src.setdefault((min(i, j), max(i, j)), s)
    for i in range(N):
        for d in range(1, prm['sequential'] + 1):
            if i + d < N: add(i, i + d, 'seq')
    g = p.gps()
    if g is not None and prm['gps_radius_m'] > 0 and prm['gps_neighbours'] > 0:
        ok = np.flatnonzero(g['valid']); tr = cKDTree(g['enu'][ok, :2])
        for k, i in enumerate(ok):
            d, j = tr.query(g['enu'][i, :2], k=min(len(ok), prm['gps_neighbours'] + 2 * prm['sequential'] + 1), distance_upper_bound=prm['gps_radius_m'])
            c = 0
            for jj in j[np.isfinite(d)]:
                jj = ok[jj]
                if abs(jj - i) > prm['sequential'] and c < prm['gps_neighbours']: add(i, jj, 'gps'); c += 1
    if prm['similar'] > 0 and os.path.exists(p.path('globals.npy')):
        gl = np.load(p.path('globals.npy')); S = gl @ gl.T
        for i in range(N):
            s = S[i].copy(); s[max(0, i - prm['similar_min_gap']):i + prm['similar_min_gap'] + 1] = -1
            for j in np.argsort(-s)[:prm['similar']]:
                if s[j] >= prm['similar_min_score']: add(i, int(j), 'similar')
    for pr in _names_to_pairs(p.path('pairs_add.txt'), idx): src[pr] = 'manual'
    rem = _names_to_pairs(p.path('pairs_remove.txt'), idx)
    for pr in rem: src.pop(pr, None)
    pairs = sorted(src)
    with open(p.path('pairs.txt'), 'w') as f: f.writelines(f'{names[i]} {names[j]} {src[(i, j)]}\n' for i, j in pairs)
    kinds = {k: sum(v == k for v in src.values()) for k in ('seq', 'gps', 'similar', 'manual')}
    per = np.bincount(np.array(pairs).ravel(), minlength=N) if pairs else np.zeros(N, int)
    revisit = sum(1 for (i, j) in pairs if abs(i - j) > max(prm['sequential'], prm['similar_min_gap']))
    ctx.metric('pairs', len(pairs)); ctx.metric('by_source', kinds); ctx.metric('removed_manual', len(rem)); ctx.metric('revisit_pairs', revisit)
    ctx.metric('min_pairs_per_frame', int(per.min()))
    ctx.check('every frame has pairs', per.min() >= 2, f'min {per.min()} pairs per frame', '>= 2', 'Raise "sequential".')
    if g is not None and int(g['pass_id'].max()) > 0:
        pid = g['pass_id']; cross = sum(1 for (i, j) in pairs if pid[i] != pid[j])
        ctx.metric('cross_pass_pairs', cross)
        ctx.check('pairs between passes', cross > 0.2 * N, f'{cross}', f'> {int(0.2 * N)}',
                  'Few pairs link the passes: raise "gps radius" or "similar". Without them passes can end up offset (doubled lanes).',
                  level=None if cross > 0.2 * N else 'warn')
    ctx.log(f'{len(pairs)} pairs: {kinds}')
    fig, ax = plt.subplots(1, 2, figsize=(17, 7))
    col = {'seq': '#888888', 'gps': '#1f77b4', 'similar': '#d62728', 'manual': '#2ca02c'}
    for k, c in col.items():
        pp = np.array([q for q in pairs if src[q] == k])
        if len(pp): ax[0].scatter(pp[:, 0], pp[:, 1], s=0.5 if N > 500 else 3, c=c, label=f'{k} ({len(pp)})')
    ax[0].set_xlim(0, N); ax[0].set_ylim(N, 0); ax[0].set_aspect('equal'); ax[0].legend(markerscale=10, fontsize=8)
    ax[0].set_xlabel('frame'); ax[0].set_ylabel('frame'); ax[0].set_title('pair matrix: dots off the diagonal = revisits')
    ax[1].plot(per); ax[1].set_xlabel('frame'); ax[1].set_ylabel('pairs'); ax[1].grid(alpha=.3); ax[1].set_title('pairs per frame')
    ctx.fig(fig, 'pairs', 'Which frames will be matched, and by which rule')
