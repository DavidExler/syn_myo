import argparse
import os

import tifffile
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing, binary_fill_holes, gaussian_filter
import numpy as np


def parse_shape(shape_str):
    parts = [part.strip() for part in shape_str.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be three integers separated by commas")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must contain integers") from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic myocyte volumes")
    parser.add_argument("--output-folder", default="annotations", help="Target output folder")
    parser.add_argument("--output-stem", default="syn_", help="Filename stem for generated images")
    parser.add_argument("--num-images", "--num-pictures", dest="num_images", type=int, default=15, help="Number of images to generate")
    parser.add_argument("--num-polys", "--num-myos", dest="num_polys", type=int, default=64, help="Number of myos per image")
    parser.add_argument("--max-tries", dest="max_tries", type=int, default=5, help="Max failed image placement attempts before skipping")
    parser.add_argument("--place-poly-max-tries", type=int, default=20, help="Max attempts to place a single poly")
    parser.add_argument("--shape", type=parse_shape, default=(1024, 1024, 128), help="Output volume shape as H,W,Z")
    parser.add_argument("--min-thickness", type=int, default=8, help="Minimum xy thickness for myos")
    parser.add_argument("--max-thickness", type=int, default=30, help="Maximum xy thickness for myos")
    parser.add_argument("--branch-prob", type=float, default=0.15, help="Probability a myo gets a secondary branch")
    parser.add_argument("--wiggle-amp", type=float, default=150.0, help="Scale factor applied to xy wiggle amplitude")
    parser.add_argument("--z-wiggle-scale", type=float, default=0.3, help="Scale factor applied to z wiggle amplitude relative to xy")
    parser.add_argument("--xy-max-wiggles", type=float, default=5.0, help="Max wiggle periods used for xy thickness variation")
    parser.add_argument("--xy-sin-influence", type=float, default=0.15, help="Sine intensity for xy thickness variation")
    parser.add_argument("--z-max-wiggles", type=float, default=5.0, help="Max wiggle periods used for z thickness variation")
    parser.add_argument("--z-sin-influence", type=float, default=0.15, help="Sine intensity for z thickness variation")
    parser.add_argument("--xy-degree", type=int, default=10, help="Polynomial degree for xy centerline")
    parser.add_argument("--xy-bound", type=float, default=0.75, help="Bound for xy centerline sampling")
    parser.add_argument("--xy-max-wiggle", type=float, default=0.095, help="Max mean square wiggle for xy centerline")
    parser.add_argument("--xy-coeff-damper", type=float, default=1.35, help="Coefficient damper for xy centerline")
    parser.add_argument("--z-degree", type=int, default=6, help="Polynomial degree for z centerline")
    parser.add_argument("--z-bound", type=float, default=0.25, help="Bound for z centerline sampling")
    parser.add_argument("--z-max-wiggle", type=float, default=0.005, help="Max mean square wiggle for z centerline")
    parser.add_argument("--z-coeff-damper", type=float, default=1.35, help="Coefficient damper for z centerline")
    parser.add_argument("--xy-straight-prob", type=float, default=0.4, help="Probability to insert a straight segment in xy")
    parser.add_argument("--xy-straight-max-length", type=int, default=500, help="Max straight segment length in xy")
    parser.add_argument("--z-straight-prob", type=float, default=0.5, help="Probability to insert a straight segment in z")
    parser.add_argument("--z-straight-max-length", type=int, default=50, help="Max straight segment length in z")
    parser.add_argument("--branch-min-len", type=int, default=40, help="Min length of a secondary branch")
    parser.add_argument("--branch-max-len", type=int, default=220, help="Max length of a secondary branch")
    parser.add_argument("--branch-min-end-dist", type=int, default=10, help="Min end distance for a branch endpoint")
    parser.add_argument("--branch-max-end-dist", type=int, default=100, help="Max end distance for a branch endpoint")
    parser.add_argument("--length-min", type=int, default=400, help="Minimum total poly length")
    parser.add_argument("--length-max", type=int, default=1025, help="Maximum total poly length (exclusive)")
    parser.add_argument("--z-mean-thickness-min", type=int, default=5, help="Minimum z mean thickness")
    parser.add_argument("--z-mean-thickness-max", type=int, default=15, help="Maximum z mean thickness")
    parser.add_argument("--z-max-thickness-extra-min", type=int, default=3, help="Minimum extra thickness added to z mean thickness")
    parser.add_argument("--z-max-thickness-extra-max", type=int, default=5, help="Maximum extra thickness added to z mean thickness")
    parser.add_argument("--xy-local-max-thickness-min", type=int, default=3, help="Minimum extra thickness added to xy mean thickness")
    parser.add_argument("--xy-local-max-thickness-max", type=int, default=8, help="Maximum extra thickness added to xy mean thickness")
    return parser.parse_args()


def sample_wiggly_polynomial(degree, size, bound, max_wiggle=0.01, coeff_damper=1.35):
    """
    Create a smooth, mildly wiggly 1D curve in [-bound, bound].

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.
    size : int
        Number of sample points.
    bound : float
        Output is guaranteed to stay within [-bound, bound].
    rng : np.random.Generator | None
        Optional random generator.

    Returns
    -------
    x : np.ndarray
        Sample locations in [-1, 1].
    y : np.ndarray
        Polynomial values in [-bound, bound].
    coeffs : np.ndarray
        Coefficients for Chebyshev terms T_1 ... T_degree.
        No constant term is used.
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")
    if size < 2:
        raise ValueError("size must be >= 2")
    if bound <= 0:
        raise ValueError("bound must be > 0")

    rng = np.random.default_rng()

    x = np.linspace(0, 1.0, size, dtype=np.float64)

    d = np.arange(1, degree + 1, dtype=np.float64)

    stop = False
    iterations = 0
    while not stop:
        iterations += 1
        coeffs = rng.uniform(-1.0, 1.0, size=degree) / (d ** coeff_damper)
        
        tail = np.clip((d / degree), 0.0, 1.0)
        coeffs *= (1.0 - 0.35 * tail**2)

        y = np.zeros_like(x)
        Tkm2 = np.ones_like(x)   # T_0
        Tkm1 = x.copy()          # T_1

        y += coeffs[0] * Tkm1

        for k in range(2, degree + 1):
            Tk = 2.0 * x * Tkm1 - Tkm2
            y += coeffs[k - 1] * Tk
            Tkm2, Tkm1 = Tkm1, Tk

        peak = np.max(np.abs(y))
        if peak > 0:
            y = y * (bound / peak)

        y = y - y[0]
        y_true = np.zeros_like(y)
        for i in range(len(y_true)):
            y_true[i] = y[0] + (y[-1] - y[0]) * (np.float16(i) / (len(y_true) - 1))
        if _get_MSE(y_true, y, max_wiggle) or iterations > 10:
            stop = True
        #else:
        #    print("Warning: MSE exceeds max_wiggle")

    return x, y, coeffs, y_true

def _get_MSE(y_true, y_pred, max_wiggle):
    wiggle = np.mean((y_true - y_pred) ** 2)
    #print(f"Wiggle: {wiggle}, max_wiggle: {max_wiggle}")
    return wiggle < max_wiggle

def _insert_straight_line(poly, prob, max_length):
    rng = np.random.default_rng()
    p = rng.uniform(0, 1)
    if p < prob:
        splines = np.zeros_like(poly)
        length = np.int16(rng.uniform(5, max_length))
        max_pos = (len(poly) - length) / len(poly) - 0.05
        pos = np.int16(rng.uniform(0.05,max_pos) * len(poly))
        #print(f"got length {length} at pixel {pos}, height {poly[pos]}")
        drv = np.gradient(poly)[pos]
        if length + pos < len(poly):
            for i in range(pos, pos + length):
                splines[i] += (i - pos) * drv + poly[pos]
                #print(f"x: {i - pos}, y: {splines[i]}")
            for i in range(0, pos):
                splines[i] = poly[i]

            scale = (len(poly) - (pos + length)) / (len(poly) - pos)
            tail = poly[pos:]
            x_tail = np.linspace(0, len(tail) - 1, len(poly) - (pos + length))
            splines[pos + length:] = ((length - 1) * drv + poly[pos]) + scale * (np.interp(x_tail, np.arange(len(tail)), tail) - poly[pos])
        else:
            print("line does not fit")
            return poly, False, None 
        return splines, True, (pos, poly[pos])
    else: 
        print("random skip of straight line")
        return poly,True, None

def _sample_thickness_curve(mean_thickness, max_thickness, max_wiggles, sin_influence, size):
    rng = np.random.default_rng()
    _, y, _, _ = sample_wiggly_polynomial(
        degree=5,
        size=size,
        bound=1,
        max_wiggle=0.01,
        coeff_damper=1.1,
    )
    y = np.asarray(y, dtype=np.float32)

    line = np.linspace(y[0], y[-1], size, dtype=np.float32)
    env = y - line
    env /= max(np.max(np.abs(env)), 1e-6)
    edge = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    edge_fade = 1.0 - (1.0 - edge**2)**2  
    poly_strength = rng.uniform(0.8, 5.0)
    env = 1.0 + poly_strength * env * (1.0 - edge_fade)

    env = 0.5 + (env - env.min()) * 0.5# / max(env.max() - env.min(), 1e-6)

    base_amp = sin_influence * (max_thickness - mean_thickness)

    # sine
    t0 = rng.uniform(0, 2 * np.pi)
    periods = rng.uniform(1.5, max_wiggles)
    phase = t0 + np.linspace(0, 2 * np.pi * periods, size, dtype=np.float32)
    wave = np.sin(phase)

    thickness = mean_thickness + ((base_amp * env) + (wave * sin_influence))

    thickness -= thickness.min()
    thickness /= max(thickness.max(), 1e-6)

    shrink = rng.uniform(0.45, 0.75)
    thickness *= shrink
    low = rng.integers(max(1, mean_thickness - 2), mean_thickness + 2)

    if mean_thickness + 2 >= max_thickness + 1:
        max_thickness = mean_thickness + 4
    high = rng.integers(mean_thickness + 2, max_thickness + 1)
    thickness = low + thickness * (high - low)

    return np.rint(thickness).astype(np.int16), wave, env

def _sample_secondary_branch(
    xy,
    z,
    min_len=40,
    max_len=220,
    prob=0.15,
    min_end_dist=10,
    max_end_dist=100,
):
    rng = np.random.default_rng()

    n = len(xy)
    if n < 30:
        return None, False

    if rng.uniform(0, 1) >= prob:
        print("random skip of secondary branch")
        return None, False

    join_idx = int(rng.integers(max(5, n // 10), max(6, n - n // 5)))

    y0 = float(xy[join_idx])
    z0 = float(z[join_idx])

    dy0 = float(np.gradient(xy)[join_idx])
    dz0 = float(np.gradient(z)[join_idx])

    branch_len = int(rng.integers(min_len, min(max_len, n)))
    end_dist = float(rng.uniform(min_end_dist, max_end_dist))

    # target endpoint offset in yz-plane relative to original endpoint
    theta = rng.uniform(0, 2 * np.pi)
    y_end = float(xy[-1]) + end_dist * np.cos(theta)
    z_end = float(z[-1]) + 0.3 * end_dist * np.sin(theta)

    dy_end = rng.uniform(-1.0, 1.0) * max(1.0, abs(dy0))
    dz_end = rng.uniform(-0.5, 0.5) * max(1.0, abs(dz0) + 1e-3)

    t = np.linspace(0.0, 1.0, branch_len)
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2

    L = branch_len - 1

    xy_branch = h00 * y0 + h10 * (dy0 * L) + h01 * y_end + h11 * (dy_end * L)
    z_branch  = h00 * z0 + h10 * (dz0 * L) + h01 * z_end + h11 * (dz_end * L)

    return {
        "join_idx": join_idx,
        "xy": xy_branch,
        "z": z_branch,
    }, True

def _insert_branch(obj, xy, z, thickness_xy, thickness_z, angle, start_xyz, branch):
    if branch is None:
        print("skip branch - branch is none")
        return obj

    join_idx = int(branch["join_idx"])
    xy_b = np.asarray(branch["xy"], dtype=np.float32).copy()
    z_b = np.asarray(branch["z"], dtype=np.float32).copy()

    if len(xy_b) == 0 or len(z_b) == 0:
        print("skip branch - branch length 0")
        return obj
    if join_idx < 0 or join_idx >= len(xy):
        print("skip branch - join idx outside frame")
        return obj

    # --- reproduce trunk straightening exactly like in _poly_to_3d_object ---
    xy_main = np.asarray(xy, dtype=np.float32).copy()
    n_main = min(len(xy_main), len(z), len(thickness_xy), len(thickness_z))
    xy_main = xy_main[:n_main]
    z_main = np.asarray(z, dtype=np.float32)[:n_main]
    thickness_xy = np.ravel(np.asarray(thickness_xy, dtype=np.int16))[:n_main]
    thickness_z = np.ravel(np.asarray(thickness_z, dtype=np.int16))[:n_main]

    for i in range(n_main):
        xy_main[i] = (xy_main[0] - xy_main[-1]) * (i / n_main) + xy_main[i] - xy_main[0]

    if join_idx >= n_main:
        print("skip branch - join idx too high")
        return obj

    c, s = np.cos(angle), np.sin(angle)
    c_p, s_p = -s, c
    x0, y0, z0 = map(float, start_xyz)

    x_line = x0 + join_idx * c
    y_line = y0 + join_idx * s

    x_join = x_line + c_p * xy_main[join_idx]
    y_join = y_line + s_p * xy_main[join_idx]
    z_join = z0 + z_main[join_idx]

    x_join = float(x_join)
    y_join = float(y_join)
    z_join = float(z_join)

    dxy = float(np.gradient(xy_main)[join_idx])
    dz = float(np.gradient(z_main)[join_idx])

    xy_b -= xy_b[0]
    z_b -= z_b[0]

    tx = c + c_p * dxy
    ty = s + s_p * dxy
    norm_xy = np.hypot(tx, ty)
    if norm_xy < 1e-8:
        tx, ty = c, s
        norm_xy = np.hypot(tx, ty)

    tx /= norm_xy
    ty /= norm_xy

    # perpendicular to tangent in xy plane
    px, py = -ty, tx

    # thickness starts from trunk thickness at join and tapers
    txy0 = max(2, int(round(thickness_xy[join_idx])))
    tz0 = max(2, int(round(thickness_z[join_idx])))

    n_branch = min(len(xy_b), len(z_b))
    xy_b = xy_b[:n_branch]
    z_b = z_b[:n_branch]

    txy_branch = np.linspace(txy0, max(2, int(round(0.6 * txy0))), n_branch)
    tz_branch = np.linspace(tz0, max(2, int(round(0.6 * tz0))), n_branch)

    # --- rasterize branch ---
    #print(f"placing {n_branch} positions")
    for i in range(n_branch):
        # branch centerline point
        x_core = x_join + i * tx + px * xy_b[i]
        y_core = y_join + i * ty + py * xy_b[i]
        z_core = z_join + i * dz + z_b[i]

        x_core = int(round(x_core))
        y_core = int(round(y_core))
        z_core = int(round(z_core))

        t_xy = max(2, int(round(txy_branch[i])))
        t_z = max(2, int(round(tz_branch[i])))

        start_xy = -(t_xy // 2)
        stop_xy = start_xy + t_xy
        start_z = -(t_z // 2)
        stop_z = start_z + t_z

        for k in range(start_xy, stop_xy):
            for kz in range(start_z, stop_z):
                if t_xy == 1 and t_z == 1:
                    inside = True
                elif t_xy == 1:
                    inside = abs(kz) <= (t_z - 1) / 2
                elif t_z == 1:
                    inside = abs(k) <= (t_xy - 1) / 2
                else:
                    inside = (k / ((t_xy - 1) / 2)) ** 2 + (kz / ((t_z - 1) / 2)) ** 2 <= 1

                if inside:
                    xx = int(round(x_core + k * px))
                    yy = int(round(y_core + k * py))
                    zz = int(round(z_core + kz))

                    if 0 <= xx < obj.shape[0] and 0 <= yy < obj.shape[1] and 0 <= zz < obj.shape[2]:
                        #print("branch voxel")
                        obj[xx, yy, zz] = 1

    return obj

def _sample_capilars(xy, z, thickness_xy, thickness_z, rng=None):
    """
    Sample 0..5 capillaries attached to a main centerline, but return only
    their properties so they can be inserted later.

    Returns
    -------
    capilars : list[dict]
        Each dict contains:
            - "join_idx": int
            - "angle_xy": float         # relative xy angle offset in radians
            - "angle_z": float          # z slope / tilt term
            - "thickness_xy": int
            - "thickness_z": int
            - "length": int
    """
    if rng is None:
        rng = np.random.default_rng()

    n = min(len(xy), len(z), len(thickness_xy), len(thickness_z))
    if n < 10:
        return []

    xy = np.asarray(xy)[:n]
    z = np.asarray(z)[:n]
    thickness_xy = np.asarray(thickness_xy)[:n]
    thickness_z = np.asarray(thickness_z)[:n]

    # 0..5, more = less likely
    # probabilities sum to 1
    n_cap = int(rng.choice([0, 1, 2, 3, 4, 5], p=[0.33, 0.27, 0.18, 0.11, 0.07, 0.04]))
    print(f"place {n_cap} capilars")
    if n_cap == 0:
        return []

    capilars = []
    used_positions = set()

    # avoid very start/end
    low = max(15, n // 10)
    high = min(n - 15, n - n // 10)
    if high <= low:
        high = low + 5

    for _ in range(n_cap):
        # try to avoid duplicate / too-close attachment points
        join_idx = None
        for _try in range(20):
            cand = int(rng.integers(low, high))
            if all(abs(cand - u) > 40 for u in used_positions):
                join_idx = cand
                used_positions.add(cand)
                break
        if join_idx is None:
            join_idx = int(rng.integers(low, high))

        base_txy = max(1, float(thickness_xy[join_idx]))
        base_tz = max(1, float(thickness_z[join_idx]))

        cap = {
            "join_idx": join_idx,
            "angle_xy": float(rng.uniform(0.95, 1.05)),
            "angle_z": float(rng.uniform(0.975, 1.025)),
            "r_xy": float(rng.uniform(0.9, 1.15)),
            "r_z": float(rng.uniform(0.95, 1.2)),
            "r_along": float(rng.uniform(1.1, 1.5)),
        }
        capilars.append(cap)

    return capilars

def _insert_capilars(obj, xy, z, thickness_xy, thickness_z, angle, start_xyz, capilars):
    if capilars is None or len(capilars) == 0:
        return obj

    xy_main = np.asarray(xy, dtype=np.float32).copy()
    z_main = np.asarray(z, dtype=np.float32).copy()
    thickness_xy = np.ravel(np.asarray(thickness_xy, dtype=np.int16))
    thickness_z = np.ravel(np.asarray(thickness_z, dtype=np.int16))

    n_main = min(len(xy_main), len(z_main), len(thickness_xy), len(thickness_z))
    if n_main < 3:
        return obj

    xy_main = xy_main[:n_main]
    z_main = z_main[:n_main]
    thickness_xy = thickness_xy[:n_main]
    thickness_z = thickness_z[:n_main]

    # same straightening as in _poly_to_3d_object
    for i in range(n_main):
        xy_main[i] = (xy_main[0] - xy_main[-1]) * (i / n_main) + xy_main[i] - xy_main[0]

    c, s = np.cos(angle), np.sin(angle)
    c_p, s_p = -s, c
    x0, y0, z0 = map(float, start_xyz)

    grad_xy = np.gradient(xy_main)
    grad_z = np.gradient(z_main)

    sx, sy, sz = obj.shape

    for cap in capilars:
        join_idx = int(cap["join_idx"])
        if not (0 <= join_idx < n_main):
            continue

        base_txy = max(2, int(round(thickness_xy[join_idx])))
        base_tz = max(2, int(round(thickness_z[join_idx])))

        # factors -> absolute radii in pixels
        r_xy = max(2, int(round(base_txy * float(cap["r_xy"]))))
        r_z = max(2, int(round(base_tz * float(cap["r_z"]))))
        r_along = max(2, int(round(base_txy * float(cap["r_along"]))))

        if r_xy > r_along:
            r_xy, r_along = r_along, r_xy

        # shell thickness in pixels
        shell_xy = max(2, int(round(0.35 * r_xy)))
        shell_z = max(2, int(round(0.35 * r_z)))
        shell_along = max(2, int(round(0.35 * r_along)))

        inner_r_xy = max(0, r_xy - shell_xy)
        inner_r_z = max(0, r_z - shell_z)
        inner_r_along = max(0, r_along - shell_along)

        # exact trunk core at join point
        x_line = x0 + join_idx * c
        y_line = y0 + join_idx * s
        x_core = x_line + c_p * xy_main[join_idx]
        y_core = y_line + s_p * xy_main[join_idx]
        z_core = z0 + z_main[join_idx]

        # local trunk direction from slope at this index
        dxy = float(grad_xy[join_idx])
        dz_local = float(grad_z[join_idx])

        tx = c + c_p * dxy
        ty = s + s_p * dxy
        tnorm = np.hypot(tx, ty)
        if tnorm < 1e-8:
            tx, ty = c, s
            tnorm = np.hypot(tx, ty)
        tx /= tnorm
        ty /= tnorm

        local_angle = np.arctan2(ty, tx)
        scaled_angle = local_angle * float(cap["angle_xy"])

        ux = np.cos(scaled_angle)
        uy = np.sin(scaled_angle)
        vx = -uy
        vy = ux
        uz = dz_local * float(cap["angle_z"])

        # tighter local bounding box
        pad_x = int(np.ceil(abs(r_along * ux) + abs(r_xy * vx))) + 2
        pad_y = int(np.ceil(abs(r_along * uy) + abs(r_xy * vy))) + 2
        pad_z = int(np.ceil(abs(r_along * uz) + r_z)) + 2

        xc = int(round(x_core))
        yc = int(round(y_core))
        zc = int(round(z_core))

        x_min = max(0, xc - pad_x)
        x_max = min(sx, xc + pad_x + 1)
        y_min = max(0, yc - pad_y)
        y_max = min(sy, yc + pad_y + 1)
        z_min = max(0, zc - pad_z)
        z_max = min(sz, zc + pad_z + 1)

        if x_min >= x_max or y_min >= y_max or z_min >= z_max:
            continue

        X, Y, Z = np.meshgrid(
            np.arange(x_min, x_max, dtype=np.float32),
            np.arange(y_min, y_max, dtype=np.float32),
            np.arange(z_min, z_max, dtype=np.float32),
            indexing="ij",
        )

        DX = X - x_core
        DY = Y - y_core
        DZ = Z - z_core

        # coordinates in ellipsoid frame
        A = DX * ux + DY * uy
        B = DX * vx + DY * vy
        C = DZ - A * uz

        outer = (A / r_along) ** 2 + (B / r_xy) ** 2 + (C / r_z) ** 2 <= 1.0

        if inner_r_along > 0 and inner_r_xy > 0 and inner_r_z > 0:
            inner = (
                (A / inner_r_along) ** 2
                + (B / inner_r_xy) ** 2
                + (C / inner_r_z) ** 2
                <= 1.0
            )
        else:
            inner = np.zeros_like(outer, dtype=bool)

        shell = outer & (~inner)

        sub = obj[x_min:x_max, y_min:y_max, z_min:z_max]

        # outer shell becomes 1
        sub[shell] = 1
        # hollow interior overwrites everything to 0
        sub[inner] = 0

        obj[x_min:x_max, y_min:y_max, z_min:z_max] = sub

    return obj

def _place_endpoint(x_core, y_core, z_core, t_xy, t_z, c, s, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # Unit vectors in xy
    u_x, u_y = c, s          # along line direction
    v_x, v_y = -s, c         # perpendicular in xy

    # Random radii: biased to be thicker than the line at the endpoint
    r_along = max(2, int(round(t_xy * rng.uniform(1.1, 2.0))))   # longest radius
    r_perp  = max(2, int(round(t_xy * rng.uniform(1.0, 1.35))))   # wider than line
    r_z     = max(2, int(round(t_z  * rng.uniform(1.0, 1.4))))   # z thickness

    pts = []

    # Filled 3D ellipsoid aligned with:
    # - major axis: line direction (c, s, 0)
    # - minor axis: perpendicular in xy (-s, c, 0)
    # - third axis: z
    for a in range(-r_along, r_along + 1):
        for b in range(-r_perp, r_perp + 1):
            for dz in range(-r_z, r_z + 1):
                if (a / r_along) ** 2 + (b / r_perp) ** 2 + (dz / r_z) ** 2 <= 1.0:
                    x = int(round(x_core + a * u_x + b * v_x))
                    y = int(round(y_core + a * u_y + b * v_y))
                    z = int(round(z_core + dz))
                    pts.append((x, y, z))

    return pts

def _poly_to_3d_object(shape, xy, z, thickness_xy, thickness_z, start_xyz=(0, 0, 0), angle=0.0, idx=None):
    out = np.zeros(shape)
    if idx is None:
        idx = 1
        #print(f"placing index {idx}")

    #xy = np.ravel(np.asarray(xy, dtype=np.float32))
    #z = np.ravel(np.asarray(z, dtype=np.float32))
    thickness_xy = np.ravel(np.asarray(thickness_xy, dtype=np.int16))
    thickness_z = np.ravel(np.asarray(thickness_z, dtype=np.int16))

    n = min(len(xy), len(z), len(thickness_xy), len(thickness_z))
    print(f"found length of poly {n}")
    xy = xy[:n]
    z = z[:n]
    thickness_xy = thickness_xy[:n]
    thickness_z = thickness_z[:n]

    for i in range(n):
        xy[i] = (xy[0] - xy[-1]) * (i / n) + xy[i] - xy[0]
    #plt.plot(xy)
    #plt.title("straigt xy")
    #plt.show()
    # base centerline before rotation


    x = np.arange(n, dtype=np.float32)
    y = xy

    c, s = np.cos(angle), np.sin(angle)
    c_p, s_p = -s, c
    x0, y0, z0 = map(float, start_xyz)
    for i in range(n):
        x_line = x0 + i * c
        y_line = y0 + i * s

        x_core = x_line + c_p * xy[i]
        y_core = y_line + s_p * xy[i]
        z_core = z0 + z[i]

        x_core = int(round(x_core))
        y_core = int(round(y_core))
        z_core = int(round(z_core))

        t_xy = max(2, int(round(thickness_xy[i])))
        t_z  = max(2, int(round(thickness_z[i])))
        if i == n - 1 or i == 0:
            for xp, yp, zp in _place_endpoint(x_core, y_core, z_core, t_xy, t_z, c, s):
                if 0 <= xp < out.shape[0] and 0 <= yp < out.shape[1] and 0 <= zp < out.shape[2]:
                    out[xp, yp, zp] = 1

        start_xy = -(t_xy // 2)
        stop_xy  = start_xy + t_xy
        start_z  = -(t_z // 2)
        stop_z   = start_z + t_z

        for k in range(start_xy, stop_xy):
            for kz in range(start_z, stop_z):
                if t_xy == 1 and t_z == 1:
                    inside = True
                elif t_xy == 1:
                    inside = abs(kz) <= (t_z - 1) / 2
                elif t_z == 1:
                    inside = abs(k) <= (t_xy - 1) / 2
                else:
                    inside = (k / ((t_xy - 1) / 2)) ** 2 + (kz / ((t_z - 1) / 2)) ** 2 <= 1

                if inside:
                    x = int(round(x_core + k * c_p))
                    y = int(round(y_core + k * s_p))
                    zz = z_core + kz

                    if 0 <= x < out.shape[0] and 0 <= y < out.shape[1] and 0 <= zz < out.shape[2]:
                        out[x, y, zz] = 1
    

    return out


def _test_fit(synth, candidate, thrsh = 150):
    _and = np.logical_and(synth > 0, candidate > 0)
    if np.sum(_and) > thrsh:
        return False
    return True

def _place_poly(synth, min_thickness, max_thickness, max_tries, params):
    rng = np.random.default_rng()

    length = int(rng.integers(params.length_min, params.length_max))
    new_idx = int(synth.max()) + 1
    print("-" * 50)
    print(f"placing idx {new_idx}")
    print("-" * 50)

    _, xy, _, _ = sample_wiggly_polynomial(
        degree=params.xy_degree,
        size=length,
        bound=params.xy_bound,
        max_wiggle=params.xy_max_wiggle,
        coeff_damper=params.xy_coeff_damper,
    )
    print(np.max(xy), np.min(xy))
    xy, _, _ = _insert_straight_line(
        xy,
        params.xy_straight_prob,
        min(params.xy_straight_max_length, length // 2),
    )
    xy = xy * params.wiggle_amp

    mean_thickness = int(rng.uniform(min_thickness, max_thickness))
    local_max_thickness = int(rng.uniform(params.xy_local_max_thickness_min, params.xy_local_max_thickness_max)) + mean_thickness
    thickness_xy, _, _ = _sample_thickness_curve(
        mean_thickness=mean_thickness,
        max_thickness=local_max_thickness,
        max_wiggles=params.xy_max_wiggles,
        sin_influence=params.xy_sin_influence,
        size=length,
    )

    _, z, _, _ = sample_wiggly_polynomial(
        degree=params.z_degree,
        size=length,
        bound=params.z_bound,
        max_wiggle=params.z_max_wiggle,
        coeff_damper=params.z_coeff_damper,
    )
    z, _, _ = _insert_straight_line(
        z,
        params.z_straight_prob,
        min(params.z_straight_max_length, length // 4),
    )
    z = z * params.wiggle_amp * params.z_wiggle_scale
    mean_thickness = np.int16(rng.uniform(params.z_mean_thickness_min, params.z_mean_thickness_max))
    z_max_thickness = np.int16(rng.uniform(params.z_max_thickness_extra_min, params.z_max_thickness_extra_max)) + mean_thickness
    thickness_z, _, _ = _sample_thickness_curve(
        mean_thickness=mean_thickness,
        max_thickness=z_max_thickness,
        max_wiggles=params.z_max_wiggles,
        sin_influence=params.z_sin_influence,
        size=length,
    )

    branch, branch_sampled = _sample_secondary_branch(
        xy,
        z,
        min_len=params.branch_min_len,
        max_len=params.branch_max_len,
        prob=params.branch_prob,
        min_end_dist=params.branch_min_end_dist,
        max_end_dist=params.branch_max_end_dist,
    )

    capilars = _sample_capilars(
        xy=xy,
        z=z,
        thickness_xy=thickness_xy,
        thickness_z=thickness_z,
        rng=rng,
    )

    shape = synth.shape
    placed = False
    tries = 0
    previous_start_xyz = tuple(rng.integers(0, s) for s in shape)
    previous_angle = rng.uniform(0, 2.0 * np.pi)
    force = False
    while not placed and tries < max_tries:
        tries += 1
        p_parallel = rng.uniform(0, 1)
        if p_parallel > 0.9:
            angle = previous_angle
            c, s = np.cos(angle), np.sin(angle)
            c_p, s_p = -s, c

            dx = int(round(10 * c_p))
            dy = int(round(10 * s_p))

            x0, y0, z0 = previous_start_xyz
            start_xyz = (
                np.clip(x0 + dx, 0, shape[0] - 1),
                np.clip(y0 + dy, 0, shape[1] - 1),
                z0,
            )
            force = True
        else:
            start_xyz = tuple(rng.integers(0, s) for s in shape)
            previous_start_xyz = start_xyz
            angle = rng.uniform(0, 2.0 * np.pi)
            previous_angle = angle
            force = False
        obj = _poly_to_3d_object(
            shape,
            xy,
            z,
            thickness_xy=thickness_xy,
            thickness_z=thickness_z,
            angle=angle,
            start_xyz=start_xyz,
        )

        if branch_sampled:
            obj = _insert_branch(
                obj,
                xy,
                z,
                thickness_xy,
                thickness_z,
                angle,
                start_xyz,
                branch,
            )
        if len(capilars) > 0:
            obj = _insert_capilars(
                obj,
                xy,
                z,
                thickness_xy,
                thickness_z,
                angle,
                start_xyz,
                capilars,
            )
        if _test_fit(synth=synth, candidate=obj) or force:
            synth[obj > 0] = new_idx
            return synth, True

    return synth, False

def _dilate_synth(synth):
    for z in range(synth.shape[2]):
        slice_ = synth[..., z].copy()

        mask = binary_dilation(slice_ > 0)
        new_pixels = np.logical_xor(slice_ > 0, mask)

        #print(f"dilating {np.count_nonzero(new_pixels)} pixels in slice {z}")

        coords = np.argwhere(new_pixels)

        for x, y in coords:
            x0 = max(0, x - 1)
            x1 = min(slice_.shape[0], x + 2)
            y0 = max(0, y - 1)
            y1 = min(slice_.shape[1], y + 2)

            neighborhood = slice_[x0:x1, y0:y1]
            values = neighborhood[neighborhood > 0]

            if values.size == 0:
                continue

            labels, counts = np.unique(values, return_counts=True)
            max_count = counts.max()
            candidates = labels[counts == max_count]

            if len(candidates) == 1:
                chosen = candidates[0]
            else:
                chosen = candidates.min()

            synth[x, y, z] = chosen

    return synth


def generate_synth_tif(num_polys, max_tries, shape, name="syn.tif", params=None):
    if not name.endswith(".tif"):
        name = name + ".tif"

    if params is None:
        params = argparse.Namespace(
            min_thickness=8,
            max_thickness=30,
            place_poly_max_tries=20,
            branch_prob=0.15,
            wiggle_amp=150.0,
            z_wiggle_scale=0.3,
            xy_max_wiggles=5.0,
            xy_sin_influence=0.15,
            z_max_wiggles=5.0,
            z_sin_influence=0.15,
            xy_degree=10,
            xy_bound=0.75,
            xy_max_wiggle=0.095,
            xy_coeff_damper=1.35,
            z_degree=6,
            z_bound=0.25,
            z_max_wiggle=0.005,
            z_coeff_damper=1.35,
            xy_straight_prob=0.4,
            xy_straight_max_length=500,
            z_straight_prob=0.5,
            z_straight_max_length=50,
            branch_min_len=40,
            branch_max_len=220,
            branch_min_end_dist=10,
            branch_max_end_dist=100,
            length_min=400,
            length_max=1025,
            z_mean_thickness_min=5,
            z_mean_thickness_max=15,
            z_max_thickness_extra_min=3,
            z_max_thickness_extra_max=5,
            xy_local_max_thickness_min=3,
            xy_local_max_thickness_max=8,
        )

    synth = np.zeros(shape)
    tries = 0
    n = 0
    while tries < max_tries and n < num_polys:
        tries += 1
        synth, success = _place_poly(
            synth=synth,
            min_thickness=params.min_thickness,
            max_thickness=params.max_thickness,
            max_tries=params.place_poly_max_tries,
            params=params,
        )
        if success:
            print(f"placed poly number {n}")
            n += 1
            tries = 0
    synth = _dilate_synth(synth)
    save = np.transpose(synth, (2, 1, 0)).astype(np.float32)
    tifffile.imwrite(
        name,
        save.astype(np.float32),
        imagej=True,
    )
    return synth
def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    for i in range(args.num_images):
        print("*" * 50)
        print(f"image {i} / {args.num_images}")
        print("*" * 50)
        output_file = os.path.join(args.output_folder, f"{args.output_stem}{i}.tif")
        generate_synth_tif(
            num_polys=args.num_polys,
            max_tries=args.max_tries,
            shape=args.shape,
            name=output_file,
            params=args,
        )


if __name__ == "__main__":
    main()