import tifffile
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing, binary_fill_holes, gaussian_filter
import numpy as np

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

def _place_poly(synth, min_thickness, max_thickness, max_tries, wiggle_amp=150, branch_prob=0.15):
    rng = np.random.default_rng()

    length = int(rng.integers(400, 1025))
    new_idx = int(synth.max()) + 1
    print("-"*50)
    print(f"placing idx {new_idx}")
    print("-"*50)

    _, xy, _, _ = sample_wiggly_polynomial(
        degree=10, size=length, bound=0.75, max_wiggle=0.095, coeff_damper=1.35
    ) 
    print(np.max(xy), np.min(xy))
    xy, _, _ = _insert_straight_line(xy, 0.4, min(500, length // 2))
    xy = xy * wiggle_amp

    mean_thickness = int(rng.uniform(min_thickness, max_thickness))
    local_max_thickness = int(rng.uniform(3, 8)) + mean_thickness
    thickness_xy, _, _ = _sample_thickness_curve(
        mean_thickness=mean_thickness,
        max_thickness=local_max_thickness,
        max_wiggles=5,
        sin_influence=0.15,
        size=length,
    )

    _, z, _, _ = sample_wiggly_polynomial(
        degree=6, size=length, bound=0.25, max_wiggle=0.005, coeff_damper=1.35
    )
    z, _, _ = _insert_straight_line(z, 0.5, min(50, length // 4))
    z = z * wiggle_amp * 0.3
    mean_thickness = np.int16(rng.uniform(5,15)) 
    max_thickness = np.int16(rng.uniform(3,5)) + mean_thickness
    thickness_z, _, _ = _sample_thickness_curve(
        mean_thickness=mean_thickness,
        max_thickness=local_max_thickness,
        max_wiggles=5,
        sin_influence=0.15,
        size=length,
    )

    branch, branch_sampled = _sample_secondary_branch(
        xy,
        z,
        min_len=40,
        max_len=220,
        prob=branch_prob,
        min_end_dist=10,
        max_end_dist=100,
    )
    
    shape = synth.shape
    placed = False
    tries = 0
    previous_start_xyz = tuple(rng.integers(0, s) for s in shape)
    previous_angle = rng.uniform(0, 2.0 * np.pi)
    force = False
    while not placed and tries < max_tries:
        tries += 1
        p_parallel = rng.uniform(0,1)
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
                z0
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
        #r = 1
        #grid = np.indices((2*r+1, 2*r+1, 2*r+1)) - r
        #kernel = (grid**2).sum(0) <= r**2
        #closed = binary_fill_holes(binary_closing(obj, structure=kernel))
        if _test_fit(synth=synth, candidate=obj) or force:
            synth[obj > 0] = new_idx
            return synth, True
    #closed = gaussian_filter(closed.astype(np.float32), sigma=1.0) > 0.5
    
    return synth, False

def generate_synth_tif(num_polys, max_tries, shape, name="syn.tif", branch_prob=0.15):
    if not name.endswith(".tif"):
        name = name + ".tif" 

    synth = np.zeros(shape)
    tries = 0
    n = 0
    while tries < max_tries and n < num_polys:
        tries += 1
        synth, success = _place_poly(synth=synth, min_thickness=8, max_thickness=30, max_tries=20, branch_prob=branch_prob)
        if success:
            print(f"placed poly number {n}")
            n += 1
            tries = 0
            save=np.transpose(synth, (2, 1, 0)).astype(np.float32)
            tifffile.imwrite(
                name,
                save.astype(np.float32),
                imagej=True
            )
    return save
if __name__ == "__main__":
    for i in range(30):
        print("*"*50)
        print(f"image {i} / 30")
        print("*"*50)
        syn_full = generate_synth_tif(num_polys=64, max_tries=5, shape=(1024,1024,128), name=f"syn_{i}.tif")