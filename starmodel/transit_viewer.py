"""
transit_viewer.py — Five-panel epoch visualization for a planetary transit.

Layout
------
Row 0 (top):    [Brightness disk map]  [Velocity disk map]
Row 1 (middle): [Normalized flux LC]   [Radial velocity anomaly]
Row 2 (bottom): [Spectrum of occulted region]  (full width)

All panels share a fixed figure size regardless of the planet's sky position.
The stellar disk axes always show exactly the same [-disk_window, +disk_window]
R★ window.

Key improvements over the original:
  * Smooth disk rendering via scipy griddata interpolation (no granulation).
  * Latitude/longitude grid lines projected onto the sky-plane disk.
  * ``pole_tilt_deg`` parameter: tilts the stellar rotation axis in the
    sky plane so the pole can point at any angle relative to the observer's
    North direction.
"""

from __future__ import annotations

import math
import numpy as np


# ─────────────────────────────────────────────────────────────────────────── #
#  Internal geometry helpers                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def _sky_frame(line_of_sight):
    """Return orthonormal (ex, ey) sky-plane basis for the given LOS."""
    lx, ly, lz = line_of_sight

    def cross(a, b):
        return (a[1]*b[2]-a[2]*b[1],
                a[2]*b[0]-a[0]*b[2],
                a[0]*b[1]-a[1]*b[0])

    def norm(v):
        n = math.sqrt(sum(c**2 for c in v))
        return tuple(c/n for c in v)

    ref = (1., 0., 0.) if abs(lx) < 0.9 else (0., 1., 0.)
    ey = norm(cross(line_of_sight, ref))
    ex = norm(cross(ey, line_of_sight))
    return ex, ey


def _rotation_pole_from_star(star):
    """
    Return the RHR North pole unit vector consistent with the star's
    Velocity model.

    The Velocity formula uses the *angular-momentum* vector:
        pole_AM = (sin i · sin λ,  −sin i · cos λ,  cos i)
    so that  v_los = v_eq · (pole_AM × n)_z  gives prograde motion
    (East limb approaching for v_eq > 0).

    Under the right-hand rule, the *North* pole is the direction from
    which the rotation looks **counter-clockwise**.  For prograde rotation
    (East approaching), that direction is the *negative* of pole_AM:

        North_RHR = −pole_AM = (−sin i · sin λ,  +sin i · cos λ,  −cos i)

    Example: i = 90°, λ = 0° → North_RHR = (0, +1, 0) = sky North (+y),
    which is the conventional astronomical North.

    Falls back to (0, 0, −1) if no Velocity model is set (pole-on, i = 0°).
    """
    if star._velocity is None:
        return np.array([0., 0., -1.])
    pole_am = np.array(star._velocity.rotation_axis, dtype=float)
    norm = np.linalg.norm(pole_am)
    if norm > 0:
        pole_am /= norm
    return -pole_am          # RHR North = −(angular-momentum South pole)


def _rotate_around_axis(vec, axis, angle_deg):
    """
    Rotate ``vec`` around ``axis`` by ``angle_deg`` degrees
    using Rodrigues' rotation formula.
    """
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    vec  = np.array(vec,  dtype=float)
    a = math.radians(angle_deg)
    return (vec * math.cos(a)
            + np.cross(axis, vec) * math.sin(a)
            + axis * np.dot(axis, vec) * (1 - math.cos(a)))


# ─────────────────────────────────────────────────────────────────────────── #
#  Smooth disk rasterisation                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def _build_disk_image_smooth(grid, quantity, line_of_sight, resolution=400):
    """
    Project visible surface elements onto the sky plane and interpolate
    smoothly onto a regular pixel grid using scipy griddata (linear, then
    nearest-neighbour fill for any interior gaps).

    Returns
    -------
    img : np.ndarray (resolution x resolution), NaN outside stellar limb
    r   : float  stellar radius
    xs  : np.ndarray  projected x-coords of visible elements
    ys  : np.ndarray  projected y-coords of visible elements
    """
    from scipy.interpolate import griddata

    ex, ey = _sky_frame(line_of_sight)
    r = grid.radius

    vis  = [e for e in grid if e.is_visible(line_of_sight)]
    vals = np.array([e.get(quantity) for e in vis], dtype=float)
    xs   = np.array([sum(e.cartesian[k]*ex[k] for k in range(3)) for e in vis])
    ys   = np.array([sum(e.cartesian[k]*ey[k] for k in range(3)) for e in vis])

    xi = np.linspace(-r, r, resolution)
    yi = np.linspace(-r, r, resolution)
    xg, yg = np.meshgrid(xi, yi)

    # Only interpolate inside the stellar disk (speed + avoids border artefacts)
    disk_mask = xg**2 + yg**2 <= (r * 0.999)**2

    pts = np.column_stack([xs, ys])
    img = np.full((resolution, resolution), np.nan)

    interp = griddata(pts, vals,
                      (xg[disk_mask], yg[disk_mask]),
                      method="linear")
    img[disk_mask] = interp

    # Fill residual NaNs in the disk interior with nearest-neighbour
    interior_nans = disk_mask & np.isnan(img)
    if interior_nans.any():
        fill = griddata(pts, vals,
                        (xg[interior_nans], yg[interior_nans]),
                        method="nearest")
        img[interior_nans] = fill

    return img, r, xs, ys


# ─────────────────────────────────────────────────────────────────────────── #
#  Latitude / longitude grid lines                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def _draw_latlon_grid(ax, line_of_sight, radius=1.0,
                      pole_axis=(0., 0., 1.),
                      dlat=30., dlon=30.,
                      color="#00ffcc", alpha=0.55, lw=1.1,
                      n_pts=300):
    """
    Draw latitude / longitude lines on the visible stellar hemisphere.

    Lines are broken whenever they pass behind the star so that only the
    visible arc is drawn.  The equator is drawn with double weight to make
    the rotational frame immediately obvious.  Pole markers (⊕N / ⊕S) are
    shown when visible.

    Parameters
    ----------
    ax : matplotlib Axes
    line_of_sight : (3,) unit vector toward the observer.
    radius : float  stellar radius.
    pole_axis : (3,) stellar rotation pole (unit vector in world coords).
        Must be the vector returned by ``Velocity.rotation_axis`` so that
        the grid frame is exactly aligned with the velocity pattern.
    dlat, dlon : float  parallel / meridian spacing in degrees.
    color : str
        Line colour.  Default cyan-green (#00ffcc) contrasts well against
        both the ``afmhot`` brightness map and the ``RdBu_r`` velocity map.
    alpha : float  line opacity (0–1).  Default 0.55.
    lw : float     line width in points.  Default 1.1.
    n_pts : int    sample points per line arc.
    """
    ex, ey = _sky_frame(line_of_sight)
    los = np.array(line_of_sight, dtype=float)

    pole = np.array(pole_axis, dtype=float)
    pole /= np.linalg.norm(pole)

    # Build the stellar equatorial frame using the same convention as
    # surface_features._stellar_frame: eq_x = LOS projected onto equatorial
    # plane so that lon=0 faces the observer (sub-observer longitude = 0°).
    los_eq = los - np.dot(los, pole) * pole
    los_eq_norm = np.linalg.norm(los_eq)
    if los_eq_norm > 1e-6:
        eq_x = los_eq / los_eq_norm
    else:
        ref2 = np.array([1., 0., 0.]) if abs(pole[0]) < 0.9 else np.array([0., 1., 0.])
        eq_x = np.cross(pole, ref2); eq_x /= np.linalg.norm(eq_x)
    eq_y = np.cross(pole, eq_x); eq_y /= np.linalg.norm(eq_y)

    def sphere_pt(lat_d, lon_d):
        lat = math.radians(lat_d)
        lon = math.radians(lon_d)
        return (math.cos(lat) * math.cos(lon) * eq_x
              + math.cos(lat) * math.sin(lon) * eq_y
              + math.sin(lat) * pole)

    def project(p):
        return (float(np.dot(p, ex)) * radius,
                float(np.dot(p, ey)) * radius)

    def draw_seg(sx_list, sy_list, lw_use, alpha_use):
        if sx_list:
            ax.plot(sx_list, sy_list,
                    color=color, alpha=alpha_use, lw=lw_use,
                    solid_capstyle="round", zorder=7)

    lons_deg = np.arange(0., 360., dlon)
    lats_deg = np.arange(-90. + dlat, 90., dlat)
    lat_arr  = np.linspace(-90., 90., n_pts)
    lon_arr  = np.linspace(0., 360., n_pts)

    # ── Meridians ─────────────────────────────────────────────────────────────
    for lon_d in lons_deg:
        seg_x, seg_y = [], []
        for lat_d in lat_arr:
            p   = sphere_pt(lat_d, lon_d)
            vis = float(np.dot(p, los)) > -0.01
            sx, sy = project(p)
            if vis:
                seg_x.append(sx); seg_y.append(sy)
            else:
                draw_seg(seg_x, seg_y, lw, alpha)
                seg_x, seg_y = [], []
        draw_seg(seg_x, seg_y, lw, alpha)

    # ── Parallels (equator drawn heavier) ─────────────────────────────────────
    for lat_d in lats_deg:
        is_equator = abs(lat_d) < 0.1          # lat_d is never exactly 0
        lw_use    = lw * 2.0 if is_equator else lw
        alpha_use = min(1.0, alpha * 1.4) if is_equator else alpha

        seg_x, seg_y = [], []
        for lon_d in lon_arr:
            p   = sphere_pt(lat_d, lon_d)
            vis = float(np.dot(p, los)) > -0.01
            sx, sy = project(p)
            if vis:
                seg_x.append(sx); seg_y.append(sy)
            else:
                draw_seg(seg_x, seg_y, lw_use, alpha_use)
                seg_x, seg_y = [], []
        draw_seg(seg_x, seg_y, lw_use, alpha_use)

    # Draw the equator separately (lat = 0 is skipped by lats_deg above)
    seg_x, seg_y = [], []
    for lon_d in lon_arr:
        p   = sphere_pt(0., lon_d)
        vis = float(np.dot(p, los)) > -0.01
        sx, sy = project(p)
        if vis:
            seg_x.append(sx); seg_y.append(sy)
        else:
            draw_seg(seg_x, seg_y, lw * 2.0, min(1.0, alpha * 1.4))
            seg_x, seg_y = [], []
    draw_seg(seg_x, seg_y, lw * 2.0, min(1.0, alpha * 1.4))

    # ── Pole markers ──────────────────────────────────────────────────────────
    for sign, label in [(1, "N"), (-1, "S")]:
        p_pole = sign * pole
        if float(np.dot(p_pole, los)) > 0:
            sx, sy = project(p_pole)
            ax.plot(sx, sy, "+",
                    color=color, ms=8, mew=1.6,
                    alpha=min(alpha * 1.8, 1.0), zorder=8)
            ax.text(sx + 0.05 * radius, sy + 0.05 * radius, label,
                    color=color, fontsize=8, fontweight="bold",
                    alpha=min(alpha * 1.8, 1.0), zorder=8, va="bottom")


# ─────────────────────────────────────────────────────────────────────────── #
#  Occulted-region spectrum                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def _occluded_spectrum(star, planet_x, planet_y, planet_radius,
                       brightness_key="brightness"):
    """Brightness-weighted mean spectrum of the occulted stellar region."""
    if star._spectrum is None:
        return None, None

    from .transit import occultation_mask
    mask = occultation_mask(star, planet_x, planet_y, planet_radius)

    wl           = star._spectrum.wavelengths
    total_flux   = np.zeros_like(wl)
    total_weight = 0.0
    los          = star.line_of_sight

    for idx, elem in enumerate(star.grid):
        if not mask[idx]:
            continue
        mu = elem.mu(los)
        if mu <= 0 or not elem.has("spectrum"):
            continue
        w = elem.get(brightness_key) * elem.area * mu
        total_flux   += np.asarray(elem.get("spectrum"), dtype=float) * w
        total_weight += w

    if total_weight == 0:
        return wl, np.zeros_like(wl)
    return wl, total_flux / total_weight


# ─────────────────────────────────────────────────────────────────────────── #
#  Linear interpolation in result arrays                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def _interpolate_result(result, t_epoch):
    t   = np.asarray(result.times)
    idx = np.clip(np.searchsorted(t, t_epoch), 1, len(t)-1)
    i0, i1 = idx-1, idx
    frac = (t_epoch - t[i0]) / (t[i1] - t[i0]) if t[i1] != t[i0] else 0.

    def lerp(arr): return float(arr[i0] + frac*(arr[i1]-arr[i0]))

    return lerp(result.flux), lerp(result.delta_rv), \
           lerp(result.planet_x), lerp(result.planet_y), lerp(result.planet_z)


# ─────────────────────────────────────────────────────────────────────────── #
#  Main public function                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def plot_transit_epoch(
    star,
    transit_model,
    result,
    t_epoch: float,
    # ── quantity keys
    brightness_key: str = "brightness",
    velocity_key:   str = "velocity",
    # ── disk rendering
    disk_resolution: int   = 400,
    disk_window:     float = 1.55,
    # ── pole tilt in the sky plane (degrees, CCW from North)
    pole_tilt_deg:   float = 0.,
    # ── lat/lon grid
    latlon_grid:   bool  = True,
    latlon_dlat:   float = 30.,
    latlon_dlon:   float = 30.,
    latlon_color:  str   = "#00ffcc",   # cyan-green: contrasts on afmhot + RdBu_r
    latlon_alpha:  float = 0.55,
    latlon_lw:     float = 1.1,
    # ── figure geometry
    figsize:   tuple = (14, 16),
    flux_ylim: tuple | None = None,
    rv_ylim:   tuple | None = None,
    # ── colormaps
    cmap_brightness: str = "afmhot",
    cmap_velocity:   str = "RdBu_r",
    # ── time-series options
    contact_lines: bool = True,
    time_format:   str  = "hours",
) -> "matplotlib.figure.Figure":
    """
    Five-panel transit epoch figure with smooth disk rendering,
    latitude/longitude grid lines aligned with the stellar rotation axis,
    and full spin-orbit obliquity support.

    The lat/lon grid pole is taken directly from
    ``star._velocity.rotation_axis``, which encodes both the stellar
    inclination *i* and the spin-orbit obliquity *λ* via:

        pole = (sin i · sin λ,  −sin i · cos λ,  cos i)

    so the parallels and meridians are always aligned with the actual
    velocity pattern regardless of λ.

    Parameters
    ----------
    star : Star
    transit_model : TransitModel
    result : TransitResult
    t_epoch : float
        Epoch time in days at which to render the snapshot.

    Disk rendering
    --------------
    disk_resolution : int
        Pixel resolution for each disk image (default 400).
    disk_window : float
        Half-width of disk panel axes in R★.

    Pole tilt (additional, on top of obliquity)
    -------------------------------------------
    pole_tilt_deg : float
        Extra sky-plane rotation of the grid pole in degrees (CCW).
        Use this only when you want to display the grid at a position
        angle that differs from the one implied by the velocity model,
        e.g. to show a reference orientation.  Normally leave at 0.

    Lat/lon grid
    ------------
    latlon_grid : bool     Draw parallels and meridians (default True).
    latlon_dlat, latlon_dlon : float  Spacing in degrees (default 30°).
    latlon_color : str     Default ``"#00ffcc"`` (cyan-green), which
                           contrasts on both ``afmhot`` and ``RdBu_r``.
    latlon_alpha : float   Opacity 0–1 (default 0.55).
    latlon_lw    : float   Line width in points (default 1.1).
                           The equator is drawn at 2× this weight.

    Returns
    -------
    matplotlib.figure.Figure  (fixed size = figsize)
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as mpl_cm
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Circle
    from scipy.ndimage import gaussian_filter

    matplotlib.rcParams.update({
        "font.size": 10, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    })

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(
        3, 2, figure=fig,
        hspace=0.46, wspace=0.38,
        top=0.93, bottom=0.06,
        left=0.09, right=0.96,
        height_ratios=[1.8, 1.0, 0.9],
    )
    ax_bright = fig.add_subplot(gs[0, 0])
    ax_vel    = fig.add_subplot(gs[0, 1])
    ax_flux   = fig.add_subplot(gs[1, 0])
    ax_rv     = fig.add_subplot(gs[1, 1])
    ax_ld     = fig.add_subplot(gs[2, 0])   # NEW: limb-darkening contrast map
    ax_ccf    = fig.add_subplot(gs[2, 1])   # NEW: CCF residual map (half width)

    for ax in (ax_bright, ax_vel, ax_flux, ax_rv, ax_ld, ax_ccf):
        ax.set_facecolor("#111111")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444444")
        ax.tick_params(colors="#bbbbbb")
        ax.xaxis.label.set_color("#bbbbbb")
        ax.yaxis.label.set_color("#bbbbbb")
        ax.title.set_color("#eeeeee")

    # ── Geometry ──────────────────────────────────────────────────────────────
    flux_ep, rv_ep, px_ep, py_ep, pz_ep = _interpolate_result(result, t_epoch)
    los = star.line_of_sight
    rp  = transit_model.orbit.planet_radius
    r   = star.radius
    in_transit = (pz_ep > 0 and
                  math.sqrt(px_ep**2 + py_ep**2) < (1. + rp))

    # Pole axis: start from what the Velocity model actually uses,
    # then apply the sky-plane tilt (rotation around the LOS).
    pole_base  = _rotation_pole_from_star(star)           # aligned with velocity field
    pole_world = _rotate_around_axis(pole_base, los, pole_tilt_deg)  # additional sky tilt

    # ── Smooth disk images ────────────────────────────────────────────────────
    img_bright, _, _, _ = _build_disk_image_smooth(
        star.grid, brightness_key, los, disk_resolution)
    img_vel,    _, _, _ = _build_disk_image_smooth(
        star.grid, velocity_key,   los, disk_resolution)

    # Light Gaussian to remove any residual interpolation noise
    sig = max(0.5, disk_resolution / 150.)
    img_bright = gaussian_filter(np.nan_to_num(img_bright, nan=0.), sig)
    img_vel    = gaussian_filter(np.nan_to_num(img_vel,    nan=0.), sig)

    # Re-apply limb mask (Gaussian may have leaked outside the limb)
    res_ = disk_resolution
    yg_, xg_ = np.mgrid[0:res_, 0:res_]
    cx_ = xg_/(res_-1)*2 - 1
    cy_ = yg_/(res_-1)*2 - 1
    outside = cx_**2 + cy_**2 > 1.005
    img_bright[outside] = np.nan
    img_vel   [outside] = np.nan

    # ── Disk imshow helper ────────────────────────────────────────────────────
    def _show(ax, img, cmap, cbar_label):
        cm_obj = mpl_cm.get_cmap(cmap).copy()
        cm_obj.set_bad("#0d0d0d")
        im = ax.imshow(
            img, origin="lower",
            extent=(-r, r, -r, r),
            cmap=cm_obj, aspect="equal",
            interpolation="bilinear",
        )
        ax.set_xlim(-disk_window, disk_window)
        ax.set_ylim(-disk_window, disk_window)
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046, aspect=20)
        cb.ax.yaxis.set_tick_params(color="#bbbbbb", labelsize=7)
        cb.outline.set_edgecolor("#444444")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#bbbbbb")
        cb.set_label(cbar_label, color="#bbbbbb", fontsize=8)
        # Limb circle
        th = np.linspace(0, 2*math.pi, 600)
        ax.plot(r*np.cos(th), r*np.sin(th),
                color="#aaaaaa", lw=0.9, alpha=0.85, zorder=5)
        return im

    _show(ax_bright, img_bright, cmap_brightness, "Intensity / I₀")
    _show(ax_vel,    img_vel,    cmap_velocity,   "v_los  (km/s)")

    # ── Lat/lon grid lines ────────────────────────────────────────────────────
    if latlon_grid:
        for ax in (ax_bright, ax_vel):
            _draw_latlon_grid(
                ax, los, radius=r,
                pole_axis=pole_world,
                dlat=latlon_dlat, dlon=latlon_dlon,
                color=latlon_color, alpha=latlon_alpha, lw=latlon_lw,
            )

    # ── Transit chord ─────────────────────────────────────────────────────────
    in_front = result.planet_z > 0
    for ax in (ax_bright, ax_vel):
        ax.plot(result.planet_x[in_front], result.planet_y[in_front],
                color="#00ccff", lw=1.0, ls="-", alpha=0.55, zorder=6)
        ax.plot(result.planet_x[~in_front], result.planet_y[~in_front],
                color="#00ccff", lw=0.6, ls=":", alpha=0.22, zorder=6)

    # ── Planet disk ───────────────────────────────────────────────────────────
    for ax in (ax_bright, ax_vel):
        face  = "#111111" if in_transit else "#1a1a2e"
        alpha_p = 0.93 if in_transit else 0.55
        ax.add_patch(Circle(
            (px_ep, py_ep), rp,
            facecolor=face, edgecolor="#4488ff",
            linewidth=1.2, alpha=alpha_p, zorder=10,
        ))
        ax.add_patch(Circle(
            (px_ep, py_ep), rp * 1.05,
            facecolor="none", edgecolor="#3366cc",
            linewidth=0.6, alpha=0.5, zorder=11,
        ))
        ch = rp * 0.35
        ax.plot([px_ep-ch, px_ep+ch], [py_ep, py_ep],
                color="#aaccff", lw=0.6, alpha=0.7, zorder=12)
        ax.plot([px_ep, px_ep], [py_ep-ch, py_ep+ch],
                color="#aaccff", lw=0.6, alpha=0.7, zorder=12)

    ax_bright.set_xlabel("x / R★"); ax_bright.set_ylabel("y / R★")
    ax_bright.set_title("Brightness map")
    ax_vel.set_xlabel("x / R★");   ax_vel.set_ylabel("y / R★")
    ax_vel.set_title("Velocity map  (Rossiter field)")

    # ── Time-series axes ──────────────────────────────────────────────────────
    t0_ref = transit_model.orbit.t0
    if time_format == "hours":
        t_plot  = (result.times - t0_ref) * 24.
        ep_x    = (t_epoch - t0_ref) * 24.
        ct_sc   = 24.
        t_label = "Time from mid-transit (h)"
    else:
        t_plot  = result.times - t0_ref
        ep_x    = t_epoch - t0_ref
        ct_sc   = 1.
        t_label = "Time from mid-transit (days)"

    # Light curve
    ax_flux.plot(t_plot, result.flux, color="#5599ff", lw=1.4, zorder=3)
    ax_flux.axvline(ep_x, color="#ffcc00", lw=1.2, ls="--", zorder=5)
    ax_flux.scatter([ep_x], [flux_ep], color="#ffcc00", s=40, zorder=6,
                    label=f"t = {t_epoch:.4f} d")
    ax_flux.set_xlabel(t_label); ax_flux.set_ylabel("Normalised flux")
    ax_flux.set_title("Transit light curve")
    ax_flux.grid(True, alpha=0.15, color="#555555")
    ax_flux.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#444444",
                   labelcolor="#eeeeee", loc="lower center")
    if flux_ylim: ax_flux.set_ylim(*flux_ylim)

    # RM anomaly
    ax_rv.plot(t_plot, result.delta_rv, color="#ff6655", lw=1.4, zorder=3)
    ax_rv.axhline(0, color="#555555", lw=0.7, ls="--")
    ax_rv.axvline(ep_x, color="#ffcc00", lw=1.2, ls="--", zorder=5)
    ax_rv.scatter([ep_x], [rv_ep], color="#ffcc00", s=40, zorder=6)
    ax_rv.set_xlabel(t_label); ax_rv.set_ylabel("ΔRV (km/s)")
    ax_rv.set_title("RM anomaly")
    ax_rv.grid(True, alpha=0.15, color="#555555")
    if rv_ylim: ax_rv.set_ylim(*rv_ylim)

    # Contact lines on both time-series panels
    if contact_lines:
        styles = {"T1":("#44ff88","-"), "T2":("#44ff88","--"),
                  "T3":("#44ff88","--"),"T4":("#44ff88","-")}
        for lbl, (col, ls) in styles.items():
            tv = result.contact_times.get(lbl, float("nan"))
            if not math.isnan(tv):
                xv = (tv - t0_ref) * ct_sc
                for ax_ts in (ax_flux, ax_rv):
                    ax_ts.axvline(xv, color=col, lw=0.7, ls=ls, alpha=0.5)
                    ax_ts.text(xv, ax_ts.get_ylim()[0], f" {lbl}",
                               color=col, fontsize=6, va="bottom", alpha=0.7)

    # ── Differential colour light curves (bottom-left) ───────────────────────
    has_color = (result.color_lcs is not None and result.color_names is not None)

    t_axis_bot = (result.times - transit_model.orbit.t0) * (
        24. if time_format == "hours" else 1.
    )

    if has_color:
        color_palette = ["#66aaff", "#ffaa33", "#cc55ff"]   # g-r, r-i, i-z
        lw_c = 1.5

        # Check if there is any meaningful signal (above noise floor ~1e-10 mag)
        max_signal = float(np.max(np.abs(result.color_lcs)))
        has_signal = max_signal > 1e-9

        for idx, (name, col) in enumerate(zip(result.color_names, color_palette)):
            ax_ld.plot(
                t_axis_bot, result.color_lcs[idx] * 1000.,  # → millimag
                color=col, lw=lw_c, label=name, zorder=3 + idx,
            )

        ax_ld.axhline(0., color="#555555", lw=0.7, ls="--", alpha=0.7)
        ax_ld.axvline(ep_x, color="#ffcc00", lw=1.2, ls="--", zorder=6)

        # Contact lines
        if contact_lines:
            for lbl, (col_c, ls_c) in {
                "T1": ("#44ff88", "-"), "T2": ("#44ff88", "--"),
                "T3": ("#44ff88", "--"), "T4": ("#44ff88", "-"),
            }.items():
                tv = result.contact_times.get(lbl, float("nan"))
                if not math.isnan(tv):
                    xv_col = (tv - transit_model.orbit.t0) * (
                        24. if time_format == "hours" else 1.
                    )
                    ax_ld.axvline(xv_col, color=col_c, lw=0.7, ls=ls_c, alpha=0.5)

        ax_ld.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#444444",
                     labelcolor="#eeeeee", loc="lower center", ncol=3)

        if not has_signal:
            ax_ld.text(0.98, 0.97,
                       "No T_eff gradient → zero colour signal\n"
                       "Set star.set_temperature_map() to see colour variations",
                       ha="right", va="top", color="#777777", fontsize=7,
                       transform=ax_ld.transAxes,
                       bbox=dict(facecolor="#1a1a1a", edgecolor="none",
                                 alpha=0.7, pad=3))

        ax_ld.set_title("Differential colour light curves  (SDSS Blackbody)")
        ax_ld.set_xlabel(t_label)
        ax_ld.set_ylabel("Δ(colour)  (mmag)")
        ax_ld.grid(True, alpha=0.15, color="#555555")
        ax_ld.tick_params(colors="#bbbbbb")

    else:
        ax_ld.text(
            0.5, 0.5,
            "Colour light curves unavailable.",
            ha="center", va="center", color="#888888",
            fontsize=9, transform=ax_ld.transAxes,
        )
        ax_ld.set_title("Differential colour light curves")
        ax_ld.set_xlabel(t_label)
        ax_ld.set_ylabel("Δ(colour)  [mag]")
        ax_ld.grid(True, alpha=0.15, color="#555555")

    # ── CCF residual map (bottom-right) ──────────────────────────────────────
    has_ccf = (result.ccf_map is not None
               and result.rv_grid is not None
               and result.ccf_oot is not None)

    if has_ccf:
        rv         = result.rv_grid
        t_axis_ccf = t_axis_bot   # same time axis as the colour LC panel

        # Residual map: CCF(t) − CCF_oot  highlights the planet shadow
        residual   = result.ccf_map - result.ccf_oot[np.newaxis, :]
        in_tr_mask = result.flux < 1.0 - 1e-5
        if in_tr_mask.any():
            sig  = float(np.std(residual[in_tr_mask]))
            vlim = max(3. * sig, 1e-4)
        else:
            vlim = float(np.max(np.abs(residual))) or 1e-4

        cmap_ccf_obj = mpl_cm.get_cmap("RdBu_r").copy()
        cmap_ccf_obj.set_bad("#0d0d0d")

        im_ccf = ax_ccf.imshow(
            residual.T,                  # (n_rv, n_times) → y = RV, x = time
            origin="lower",
            aspect="auto",
            extent=[t_axis_ccf[0], t_axis_ccf[-1], rv[0], rv[-1]],
            cmap=cmap_ccf_obj,
            vmin=-vlim, vmax=vlim,
            interpolation="bilinear",
        )

        # OOT CCF profile as faint vertical hairlines across the full time span
        # (scaled to fit the RV axis so the line shape is visible)
        ccf_norm = result.ccf_oot - result.ccf_oot.min()
        if ccf_norm.max() > 0:
            ccf_norm = ccf_norm / ccf_norm.max()
        rv_span   = rv[-1] - rv[0]
        ccf_scaled = rv[0] + ccf_norm * rv_span   # map 0-1 → rv_min:rv_max
        for t_val in t_axis_ccf[::max(1, len(t_axis_ccf) // 10)]:
            ax_ccf.plot([t_val] * len(rv), ccf_scaled,
                        color="#ffcc66", lw=0.5, alpha=0.18, zorder=2)

        # Epoch marker and zero-velocity line
        ax_ccf.axvline(ep_x, color="#ffcc00", lw=1.2, ls="--", zorder=5)
        ax_ccf.axhline(0.,   color="#888888", lw=0.7, ls=":",  alpha=0.5)

        # Contact lines
        if contact_lines:
            for lbl, (col_c, ls_c) in {
                "T1": ("#44ff88", "-"),  "T2": ("#44ff88", "--"),
                "T3": ("#44ff88", "--"), "T4": ("#44ff88", "-"),
            }.items():
                tv = result.contact_times.get(lbl, float("nan"))
                if not math.isnan(tv):
                    xv_c = (tv - transit_model.orbit.t0) * (
                        24. if time_format == "hours" else 1.
                    )
                    ax_ccf.axvline(xv_c, color=col_c, lw=0.7, ls=ls_c, alpha=0.5)

        # Colourbar
        cb_ccf = fig.colorbar(im_ccf, ax=ax_ccf, pad=0.02, fraction=0.046, aspect=20)
        cb_ccf.set_label("ΔCCF  (transit − OOT)", color="#bbbbbb", fontsize=8)
        cb_ccf.ax.yaxis.set_tick_params(color="#bbbbbb", labelsize=7)
        cb_ccf.outline.set_edgecolor("#444444")
        plt.setp(cb_ccf.ax.yaxis.get_ticklabels(), color="#bbbbbb")

        # Title: note template source
        has_ext  = (star._spectrum is not None
                    and getattr(star._spectrum, "has_template", False))
        tmpl_str = "external template" if has_ext else "Gaussian mask"
        ax_ccf.set_title(f"CCF residual map  │  {tmpl_str}")
        ax_ccf.set_xlabel(t_label)
        ax_ccf.set_ylabel("Radial velocity  (km/s)")
        ax_ccf.tick_params(colors="#bbbbbb")

    else:
        ax_ccf.text(
            0.5, 0.5,
            "No CCF map available.\n"
            "Pass compute_ccf=True to TransitModel.compute()\n"
            "and ensure the Star has a Spectrum model with lines.",
            ha="center", va="center", color="#888888",
            fontsize=9, transform=ax_ccf.transAxes,
        )
        ax_ccf.set_title("CCF residual map")
        ax_ccf.set_xlabel(t_label)
        ax_ccf.set_ylabel("Radial velocity  (km/s)")
        ax_ccf.grid(True, alpha=0.15, color="#555555")

    # ── Super-title ───────────────────────────────────────────────────────────
    phase = ((t_epoch - transit_model.orbit.t0)
             / transit_model.orbit.period + 0.5) % 1. - 0.5
    ep_h  = (t_epoch - transit_model.orbit.t0) * 24.
    lam_deg = (math.degrees(star._velocity.obliquity_rad)
               if star._velocity is not None else 0.0)
    tilt_str = f"pole tilt = {pole_tilt_deg:+.1f}°" if pole_tilt_deg != 0 else ""
    obl_str  = f"λ = {lam_deg:+.1f}°"
    geom_str = ",  ".join(s for s in [obl_str, tilt_str] if s)
    status   = "IN TRANSIT" if in_transit else "out of transit"

    fig.suptitle(
        f"{star.name}  —  Transit epoch viewer  │  {geom_str}\n"
        f"t = {t_epoch:.5f} d  │  Δt = {ep_h:+.3f} h  │  "
        f"φ = {phase:+.4f}  │  {status}",
        color="#eeeeee", fontsize=10.5, fontweight="bold", y=0.975,
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────── #
#  Animation helper                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def animate_transit(
    star, transit_model, result,
    epochs: np.ndarray,
    output_path: str = "transit_animation.gif",
    fps: int = 15,
    dpi: int = 100,
    **epoch_kwargs,
):
    """
    Save an animated sequence of :func:`plot_transit_epoch` frames.

    Parameters
    ----------
    epochs : array_like  — time values (days).
    output_path : str    — .gif or .mp4.
    fps, dpi : int
    **epoch_kwargs       — passed to :func:`plot_transit_epoch`.

    Returns
    -------
    str : path to saved file.
    """
    import tempfile, os
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    epochs = np.asarray(epochs)
    tmpdir = tempfile.mkdtemp()
    paths  = []

    for i, t_ep in enumerate(epochs):
        f = plot_transit_epoch(star, transit_model, result,
                               float(t_ep), **epoch_kwargs)
        fp = os.path.join(tmpdir, f"frame_{i:04d}.png")
        f.savefig(fp, dpi=dpi, bbox_inches="tight",
                  facecolor=f.get_facecolor())
        plt.close(f)
        paths.append(fp)

    if output_path.endswith(".mp4"):
        try:
            import imageio
            with imageio.get_writer(output_path, fps=fps) as w:
                for p in paths: w.append_data(imageio.imread(p))
        except ImportError:
            output_path = output_path.replace(".mp4", ".gif")
            print("imageio not found; saving as GIF.")

    if output_path.endswith(".gif"):
        imgs = [PILImage.open(p) for p in paths]
        imgs[0].save(output_path, save_all=True, append_images=imgs[1:],
                     duration=int(1000/fps), loop=0)

    return output_path
