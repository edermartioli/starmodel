"""
transit_overview.py — Compact 4-panel transit overview plot.

Layout
------
Row 0 (top):    [Brightness disk map]  [Velocity disk map]
Row 1 (bottom): [Normalised flux LC]   [RM radial velocity]

This is a lightweight complement to the full 5-panel
:func:`~starmodel.transit_viewer.plot_transit_epoch`.  It omits the
colour LC and CCF panels so the result is easy to read at a glance,
suitable for quick inspection, publications, or presentations.

Usage
-----
    from starmodel import PlanetarySystem, TransitModel
    from starmodel.transit_overview import plot_transit_overview
    import numpy as np

    sys   = PlanetarySystem("WASP-108.json")
    star  = sys.build_star(wavelengths=np.linspace(6300, 6900, 400))
    star.add_spectral_line(6563., depth=0.65, width=1.5)
    star.compute()

    orbit  = sys.build_orbit()
    model  = TransitModel(star, orbit)
    result = model.compute(n_times=500)

    fig = plot_transit_overview(star, model, result, t_epoch=orbit.t0)
    fig.savefig("overview.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
"""

from __future__ import annotations

import math
import numpy as np

# Reuse all geometry / rendering helpers from the main viewer
from .transit_viewer import (
    _sky_frame,
    _rotation_pole_from_star,
    _rotate_around_axis,
    _build_disk_image_smooth,
    _draw_latlon_grid,
    _interpolate_result,
)


def plot_transit_overview(
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
    # ── pole tilt (additional, on top of obliquity already encoded in star)
    pole_tilt_deg:   float = 0.,
    # ── lat/lon grid
    latlon_grid:  bool  = True,
    latlon_dlat:  float = 30.,
    latlon_dlon:  float = 30.,
    latlon_color: str   = "#00ffcc",
    latlon_alpha: float = 0.55,
    latlon_lw:    float = 1.1,
    # ── figure
    figsize:   tuple = (12, 10),
    flux_ylim: tuple | None = None,
    rv_ylim:   tuple | None = None,
    # ── colormaps
    cmap_brightness: str = "afmhot",
    cmap_velocity:   str = "RdBu_r",
    # ── time-series
    contact_lines: bool = True,
    time_format:   str  = "hours",
) -> "matplotlib.figure.Figure":
    """
    Four-panel transit overview figure.

    Panels
    ------
    Top-left    : Stellar disk brightness map with transit chord and planet.
    Top-right   : Stellar disk velocity map (Rossiter field) with transit chord.
    Bottom-left : Full normalised photometric light curve with epoch marker.
    Bottom-right: Rossiter-McLaughlin radial-velocity anomaly with epoch marker.

    Parameters
    ----------
    star : Star
        Configured and computed :class:`~starmodel.Star`.
    transit_model : TransitModel
        Transit model for the system.
    result : TransitResult
        Pre-computed result from ``TransitModel.compute()``.
    t_epoch : float
        Epoch time in days to snapshot on the disk maps.

    Disk rendering
    --------------
    disk_resolution : int
        Pixel resolution of each rasterised disk image (default 400).
    disk_window : float
        Half-width of disk panel axes in R★ (default 1.55, must be > 1 + Rp).

    Pole tilt
    ---------
    pole_tilt_deg : float
        Extra sky-plane rotation of the lat/lon grid pole in degrees (CCW).
        Normally 0; the obliquity already encoded in ``star.set_rotation()``
        is applied automatically.

    Lat/lon grid
    ------------
    latlon_grid : bool
        Draw parallels and meridians (default True).
    latlon_dlat, latlon_dlon : float
        Grid spacing in degrees (default 30°).
    latlon_color : str
        Grid line colour (default ``"#00ffcc"``).
    latlon_alpha : float
        Grid line opacity (default 0.55).
    latlon_lw : float
        Grid line width in points (default 1.1).

    Figure
    ------
    figsize : (width, height) in inches.  Default (12, 10).
    flux_ylim, rv_ylim : optional y-axis limits for the time-series panels.
    cmap_brightness, cmap_velocity : colormaps.
    contact_lines : bool
        Draw T1–T4 dashed lines on the time-series panels.
    time_format : {"hours", "days"}
        Time axis unit for the LC and RM panels.

    Returns
    -------
    matplotlib.figure.Figure
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

    # ── Layout: 2 rows × 2 columns ───────────────────────────────────────────
    fig = plt.figure(figsize=figsize, facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.38, wspace=0.35,
        top=0.92, bottom=0.08,
        left=0.09, right=0.96,
        height_ratios=[1.7, 1.0],
    )
    ax_bright = fig.add_subplot(gs[0, 0])
    ax_vel    = fig.add_subplot(gs[0, 1])
    ax_flux   = fig.add_subplot(gs[1, 0])
    ax_rv     = fig.add_subplot(gs[1, 1])

    for ax in (ax_bright, ax_vel, ax_flux, ax_rv):
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

    pole_base  = _rotation_pole_from_star(star)
    pole_world = _rotate_around_axis(pole_base, los, pole_tilt_deg)

    # ── Smooth disk images ────────────────────────────────────────────────────
    img_bright, _, _, _ = _build_disk_image_smooth(
        star.grid, brightness_key, los, disk_resolution)
    img_vel,    _, _, _ = _build_disk_image_smooth(
        star.grid, velocity_key,   los, disk_resolution)

    sig = max(0.5, disk_resolution / 150.)
    img_bright = gaussian_filter(np.nan_to_num(img_bright, nan=0.), sig)
    img_vel    = gaussian_filter(np.nan_to_num(img_vel,    nan=0.), sig)

    # Re-apply limb mask after smoothing
    res_ = disk_resolution
    yg_, xg_ = np.mgrid[0:res_, 0:res_]
    cx_ = xg_ / (res_ - 1) * 2 - 1
    cy_ = yg_ / (res_ - 1) * 2 - 1
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
        # Stellar limb circle
        th = np.linspace(0, 2 * math.pi, 600)
        ax.plot(r * np.cos(th), r * np.sin(th),
                color="#aaaaaa", lw=0.9, alpha=0.85, zorder=5)
        return im

    _show(ax_bright, img_bright, cmap_brightness, "Intensity / I₀")
    _show(ax_vel,    img_vel,    cmap_velocity,   "v_los  (km/s)")

    # ── Lat/lon grid ──────────────────────────────────────────────────────────
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
        ax.plot(result.planet_x[in_front],  result.planet_y[in_front],
                color="#00ccff", lw=1.0, ls="-", alpha=0.55, zorder=6)
        ax.plot(result.planet_x[~in_front], result.planet_y[~in_front],
                color="#00ccff", lw=0.6, ls=":", alpha=0.22, zorder=6)

    # ── Planet disk at epoch ──────────────────────────────────────────────────
    for ax in (ax_bright, ax_vel):
        face    = "#111111" if in_transit else "#1a1a2e"
        alpha_p = 0.93     if in_transit else 0.55
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
        ax.plot([px_ep - ch, px_ep + ch], [py_ep, py_ep],
                color="#aaccff", lw=0.6, alpha=0.7, zorder=12)
        ax.plot([px_ep, px_ep], [py_ep - ch, py_ep + ch],
                color="#aaccff", lw=0.6, alpha=0.7, zorder=12)

    ax_bright.set_xlabel("x / R★"); ax_bright.set_ylabel("y / R★")
    ax_bright.set_title("Brightness map")
    ax_vel.set_xlabel("x / R★");    ax_vel.set_ylabel("y / R★")
    ax_vel.set_title("Velocity map  (Rossiter field)")

    # ── Time-series setup ─────────────────────────────────────────────────────
    t0_ref = transit_model.orbit.t0
    if time_format == "hours":
        t_plot  = (result.times - t0_ref) * 24.
        ep_x    = (t_epoch   - t0_ref) * 24.
        ct_sc   = 24.
        t_label = "Time from mid-transit (h)"
    else:
        t_plot  = result.times - t0_ref
        ep_x    = t_epoch   - t0_ref
        ct_sc   = 1.
        t_label = "Time from mid-transit (days)"

    # ── Light curve ───────────────────────────────────────────────────────────
    ax_flux.plot(t_plot, result.flux, color="#5599ff", lw=1.4, zorder=3)
    ax_flux.axvline(ep_x, color="#ffcc00", lw=1.2, ls="--", zorder=5)
    ax_flux.scatter([ep_x], [flux_ep], color="#ffcc00", s=40, zorder=6,
                    label=f"t = {t_epoch:.4f} d")
    ax_flux.set_xlabel(t_label)
    ax_flux.set_ylabel("Normalised flux")
    ax_flux.set_title("Transit light curve")
    ax_flux.grid(True, alpha=0.15, color="#555555")
    ax_flux.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#444444",
                   labelcolor="#eeeeee", loc="lower center")
    if flux_ylim:
        ax_flux.set_ylim(*flux_ylim)

    # ── RM anomaly ────────────────────────────────────────────────────────────
    ax_rv.plot(t_plot, result.delta_rv, color="#ff6655", lw=1.4, zorder=3)
    ax_rv.axhline(0, color="#555555", lw=0.7, ls="--")
    ax_rv.axvline(ep_x, color="#ffcc00", lw=1.2, ls="--", zorder=5)
    ax_rv.scatter([ep_x], [rv_ep], color="#ffcc00", s=40, zorder=6)
    ax_rv.set_xlabel(t_label)
    ax_rv.set_ylabel("ΔRV  (km/s)")
    ax_rv.set_title("RM anomaly")
    ax_rv.grid(True, alpha=0.15, color="#555555")
    if rv_ylim:
        ax_rv.set_ylim(*rv_ylim)

    # ── Contact lines on time-series panels ───────────────────────────────────
    if contact_lines:
        styles = {
            "T1": ("#44ff88", "-"),  "T2": ("#44ff88", "--"),
            "T3": ("#44ff88", "--"), "T4": ("#44ff88", "-"),
        }
        for lbl, (col, ls) in styles.items():
            tv = result.contact_times.get(lbl, float("nan"))
            if not math.isnan(tv):
                xv = (tv - t0_ref) * ct_sc
                for ax_ts in (ax_flux, ax_rv):
                    ax_ts.axvline(xv, color=col, lw=0.7, ls=ls, alpha=0.5)
                    ax_ts.text(xv, ax_ts.get_ylim()[0], f" {lbl}",
                               color=col, fontsize=6, va="bottom", alpha=0.7)

    # ── Super-title ───────────────────────────────────────────────────────────
    phase   = ((t_epoch - transit_model.orbit.t0)
               / transit_model.orbit.period + 0.5) % 1. - 0.5
    ep_h    = (t_epoch - transit_model.orbit.t0) * 24.
    lam_deg = (math.degrees(star._velocity.obliquity_rad)
               if star._velocity is not None else 0.0)
    status  = "IN TRANSIT" if in_transit else "out of transit"

    fig.suptitle(
        f"{star.name}  —  Transit overview  │  λ = {lam_deg:+.1f}°\n"
        f"t = {t_epoch:.5f} d  │  Δt = {ep_h:+.3f} h  │  "
        f"φ = {phase:+.4f}  │  {status}",
        color="#eeeeee", fontsize=10.5, fontweight="bold", y=0.975,
    )

    return fig
