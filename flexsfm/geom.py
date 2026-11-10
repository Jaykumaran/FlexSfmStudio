"""Geometry helpers shared by the steps."""
import numpy as np


def umeyama(src, dst, with_scale=True):
    """Similarity dst ~ s R src + t (least squares)."""
    ms, md = src.mean(0), dst.mean(0); S, D = src - ms, dst - md
    U, s, Vt = np.linalg.svd(D.T @ S / len(src)); d = np.sign(np.linalg.det(U @ Vt))
    E = np.diag([1, 1, d]); R = U @ E @ Vt
    c = (s * np.diag(E)).sum() / (S ** 2).sum(1).mean() if with_scale else 1.0
    return c, R, md - c * R @ ms


def robust_umeyama(src, dst, thr=3.0, iters=500, seed=0):
    """RANSAC + refit similarity; returns (s, R, t, inlier mask)."""
    rng = np.random.default_rng(seed); n = len(src); best = None
    if n < 3: raise ValueError('need >= 3 points for a similarity')
    for _ in range(iters):
        k = rng.choice(n, 3, replace=False)
        try:
            s, R, t = umeyama(src[k], dst[k])
        except np.linalg.LinAlgError:
            continue
        inl = np.linalg.norm((src @ R.T) * s + t - dst, axis=1) < thr
        if best is None or inl.sum() > best.sum(): best = inl
    if best.sum() < 3: best = np.ones(n, bool)
    s, R, t = umeyama(src[best], dst[best])
    for _ in range(3):
        inl = np.linalg.norm((src @ R.T) * s + t - dst, axis=1) < thr
        if inl.sum() >= 3: s, R, t = umeyama(src[inl], dst[inl]); best = inl
    return s, R, t, best


def sim3(s, R, t):
    import pycolmap
    return pycolmap.Sim3d(float(s), pycolmap.Rotation3d(np.asarray(R, float)), np.asarray(t, float))


def normalized(cam, xy):
    """Pixel coordinates -> normalized image-plane coordinates (removes distortion)."""
    return np.asarray(cam.cam_from_img(np.asarray(xy, float)), float).reshape(-1, 2)


def essential_from_poses(rec, ia, ib):
    A, B = rec.images[ia], rec.images[ib]; T = B.cam_from_world() * A.cam_from_world().inverse()
    R, t = T.rotation.matrix(), T.translation
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]) @ R


def epipolar_px(rec, ia, ib, xa, xb):
    """Sampson distance of matches (pixel coords) against the current poses, in pixels (via the focal length)."""
    ca, cb = rec.cameras[rec.images[ia].camera_id], rec.cameras[rec.images[ib].camera_id]
    p = normalized(ca, xa); q = normalized(cb, xb); E = essential_from_poses(rec, ia, ib)
    p = np.c_[p, np.ones(len(p))]; q = np.c_[q, np.ones(len(q))]; Ep = p @ E.T; Etq = q @ E
    d = np.abs((q * Ep).sum(1)) / np.sqrt(Ep[:, 0] ** 2 + Ep[:, 1] ** 2 + Etq[:, 0] ** 2 + Etq[:, 1] ** 2)
    return d * 0.5 * (ca.mean_focal_length() + cb.mean_focal_length())


def triangulate_track(rec, obs):
    """obs: list of (image_id, xy pixel). Multi-view DLT in normalized coords. Returns xyz, max reprojection error (px)."""
    rows = []
    for i, xy in obs:
        im = rec.images[i]; P = im.cam_from_world().matrix(); u = normalized(rec.cameras[im.camera_id], [xy])[0]
        rows += [u[0] * P[2] - P[0], u[1] * P[2] - P[1]]
    X = np.linalg.svd(np.array(rows))[2][-1]
    if abs(X[3]) < 1e-12: return None, np.inf
    X = X[:3] / X[3]; err = 0.0
    for i, xy in obs:
        im = rec.images[i]; pc = im.cam_from_world() * X
        if pc[2] <= 0: return None, np.inf
        q = rec.cameras[im.camera_id].img_from_cam(np.asarray(pc, float).reshape(1, 3))
        err = max(err, float(np.linalg.norm(np.asarray(q).ravel()[:2] - np.asarray(xy))))
    return X, err


def footprints(rec):
    """Per image: median of its 3D points (where the camera looks), viewing direction, centre."""
    out = {}
    for i, im in rec.images.items():
        ids = [p.point3D_id for p in im.points2D if p.has_point3D()]
        X = np.array([rec.points3D[k].xyz for k in ids]) if ids else np.array([im.projection_center()])
        out[i] = (np.median(X, 0), im.viewing_direction(), im.projection_center(), len(ids))
    return out


def tri_angle_deg(rec, p):
    C = np.array([rec.images[e.image_id].projection_center() for e in p.track.elements]) - p.xyz
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return float(np.degrees(np.arccos(np.clip((C @ C.T).min(), -1, 1))))
