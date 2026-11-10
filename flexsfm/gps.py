"""GPS sources (DJI SRT, image EXIF, CSV), conversion to local east/north/up metres, pass detection (where the camera turns back)."""
import csv, math, os, re
import numpy as np


def parse_srt(path):
    """DJI .SRT -> {video frame index (0-based): (lat, lon, abs_alt)}. Handles '[latitude: x]' and 'GPS(lon,lat,alt)' styles."""
    rec = {}
    txt = open(path, encoding='utf-8', errors='ignore').read().replace('\r', '')
    for k, blk in enumerate(re.split(r'\n\s*\n', txt.strip())):
        m = re.search(r'FrameCnt\s*:\s*(\d+)', blk)
        idx = int(m.group(1)) - 1 if m else k
        la = re.search(r'latitude\s*:\s*([-\d.]+)', blk); lo = re.search(r'longitude\s*:\s*([-\d.]+)', blk)
        al = re.search(r'abs_alt\s*:\s*([-\d.]+)', blk) or re.search(r'altitude\s*:\s*([-\d.]+)', blk)
        if la and lo:
            rec[idx] = (float(la.group(1)), float(lo.group(1)), float(al.group(1)) if al else 0.0); continue
        g = re.search(r'GPS\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)', blk)   # older DJI: GPS(lon, lat, alt)
        if g: rec[idx] = (float(g.group(2)), float(g.group(1)), float(g.group(3)))
    return rec


def srt_for_frames(names, srt, step):
    """frame_000001.jpg <-> video frame 0, frame_000002 <-> video frame step, ... (frames extracted every `step` frames)."""
    rec = parse_srt(srt); keys = np.array(sorted(rec)); lla, ok = [], []
    for n in names:
        m = re.search(r'(\d+)(?!.*\d)', os.path.splitext(n)[0]); k = (int(m.group(1)) - 1) * step if m else 0
        if k not in rec and len(keys):
            j = int(keys[np.abs(keys - k).argmin()]); ok.append(abs(j - k) <= step); k = j
        else:
            ok.append(k in rec)
        lla.append(rec.get(k, (np.nan, np.nan, np.nan)))
    return np.array(lla, float), np.array(ok)


def exif_gps(path):
    from PIL import Image
    try:
        ex = Image.open(path).getexif(); g = ex.get_ifd(0x8825)
        if not g: return None
        def dms(v): return float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        lat = dms(g[2]) * (-1 if g.get(1) in ('S', b'S') else 1); lon = dms(g[4]) * (-1 if g.get(3) in ('W', b'W') else 1)
        alt = float(g.get(6, 0.0)) * (-1 if g.get(5) in (1, b'\x01') else 1)
        return lat, lon, alt
    except Exception:
        return None


def read_csv(path, names):
    """CSV with a name column and either lat/lon/alt or x/y/z (already metric). Column names are detected."""
    rows = []
    with open(path, newline='') as f:
        txt = f.read()
    lines = [l for l in txt.splitlines() if l.strip()]
    hdr = [h.strip().lstrip('#').strip().lower() for h in next(csv.reader([lines[0]]))]
    body = list(csv.reader(lines[1:]))
    def col(*opts):
        for o in opts:
            for i, h in enumerate(hdr):
                if h == o or h.startswith(o): return i
        return None
    ni = col('label', 'name', 'image', 'file') or 0
    la, lo, al = col('latitude', 'lat'), col('longitude', 'lon', 'lng'), col('altitude', 'alt', 'height', 'z')
    x, y = col('x', 'easting', 'east'), col('y', 'northing', 'north')
    geo = la is not None and lo is not None
    d = {}
    for r in body:
        try:
            if geo: d[os.path.basename(r[ni])] = (float(r[la]), float(r[lo]), float(r[al]) if al is not None and r[al] else 0.0)
            else: d[os.path.basename(r[ni])] = (float(r[x]), float(r[y]), float(r[al]) if al is not None else 0.0)
        except (ValueError, IndexError, TypeError):
            continue
    v = np.array([d.get(n, (np.nan,) * 3) for n in names], float)
    return v, ~np.isnan(v[:, 0]), geo


def lla_to_enu(lla, ref=None):
    """Local tangent plane at ref (lat0, lon0, h0) - accurate to mm over a few km."""
    ok = ~np.isnan(lla[:, 0])
    if ref is None:
        lat0, lon0, h0 = lla[ok][0]
    else:
        lat0, lon0, h0 = ref['lat0'], ref['lon0'], ref['h0']
    a, f = 6378137.0, 1 / 298.257223563; e2 = f * (2 - f); p = math.radians(lat0)
    M = a * (1 - e2) / (1 - e2 * math.sin(p) ** 2) ** 1.5; N = a / math.sqrt(1 - e2 * math.sin(p) ** 2)
    enu = np.c_[np.radians(lla[:, 1] - lon0) * N * math.cos(p), np.radians(lla[:, 0] - lat0) * M, lla[:, 2] - h0]
    return enu, dict(lat0=float(lat0), lon0=float(lon0), h0=float(h0), M=M, N=N)


def enu_to_lla(enu, ref):
    lat = ref['lat0'] + np.degrees(enu[:, 1] / ref['M']); lon = ref['lon0'] + np.degrees(enu[:, 0] / (ref['N'] * math.cos(math.radians(ref['lat0']))))
    return np.c_[lat, lon, enu[:, 2] + ref['h0']]


def detect_passes(enu, valid, w=100, min_gap=150):
    """Split a trajectory where it turns back (U-turns). Returns pass id per frame (0, 1, ...) and the boundary indices."""
    N = len(enu); e = enu.copy()
    if (~valid).any():
        idx = np.flatnonzero(valid)
        for k in range(3): e[:, k] = np.interp(np.arange(N), idx, enu[idx, k])
    w = max(10, min(w, N // 6))
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(e[:, :2], axis=0), axis=1))]; ratio = np.full(N, np.inf)
    for i in range(w, N - w):
        fl = cum[i + w] - cum[i - w]
        if fl > 10: ratio[i] = np.linalg.norm(e[i + w, :2] - e[i - w, :2]) / fl
    ext = [0]
    for i in np.argsort(ratio):
        if ratio[i] > 0.3: break
        if all(abs(i - b) > min_gap for b in ext): ext.append(int(i))
    ext = sorted(ext) + [N]; pid = np.zeros(N, int)
    for k in range(len(ext) - 1): pid[ext[k]:ext[k + 1]] = k
    return pid, ext
