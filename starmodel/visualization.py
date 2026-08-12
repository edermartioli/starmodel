"""
visualization.py — Optional matplotlib-based visualization for starmodel.

Functions
---------
plot_surface_map(grid, quantity, ...)
    Mollweide or rectangular map of a scalar quantity over the surface.

plot_disk(grid, quantity, ...)
    Synthetic disk image (star as seen by observer).

plot_spectrum(wavelengths, flux, ...)
    Line plot of a disk-integrated spectrum.

plot_velocity_field(grid, ...)
    Color map of the radial velocity field on the disk.
"""

from __future__ import annotations

import math
import numpy as np


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        return plt, cm
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with:  pip install matplotlib"
        ) from exc


# ======================================================================== #
#  Surface map (θ-φ projection)                                            #
# ======================================================================== #

def plot_surface_map(
    grid,
    quantity: str,
    projection: str = "mollweide",
    cmap: str = "inferno",
    title: str | None = None,
    colorbar_label: str | None = None,
    ax=None,
    **scatter_kw,
):
    """
    Plot a scalar quantity over the stellar surface in a sky-projection.

    Parameters
    ----------
    grid : SurfaceGrid
    quantity : str
        Name of the stored quantity.
    projection : {"mollweide", "rect"}
        Map projection.
    cmap : str
        Matplotlib colormap name.
    title : str, optional
    colorbar_label : str, optional
    ax : matplotlib Axes, optional
        If given, draw onto this axes (must match *projection*).

    Returns
    -------
    fig, ax
    """
    plt, cm = _require_matplotlib()

    values = grid.get_array(quantity).astype(float)
    theta = grid.get_property_array("theta")
    phi = grid.get_property_array("phi")

    # Convert to longitude ∈ [-π, π] for Mollweide
    lon = phi.copy()
    lon[lon > math.pi] -= 2 * math.pi
    lat = math.pi / 2 - theta          # colatitude → latitude

    if ax is None:
        if projection == "mollweide":
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111, projection="mollweide")
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    sc = ax.scatter(
        lon, lat, c=values, cmap=cmap, s=10, linewidths=0, **scatter_kw
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.05, shrink=0.8)
    if colorbar_label:
        cb.set_label(colorbar_label)

    ax.set_title(title or quantity)
    if projection != "mollweide":
        ax.set_xlabel("Longitude (rad)")
        ax.set_ylabel("Latitude (rad)")

    return fig, ax


# ======================================================================== #
#  Disk image (observer view)                                               #
# ======================================================================== #

def plot_disk(
    grid,
    quantity: str,
    line_of_sight: tuple[float, float, float] = (0.0, 0.0, 1.0),
    resolution: int = 300,
    cmap: str = "inferno",
    title: str | None = None,
    colorbar_label: str | None = None,
    ax=None,
    **imshow_kw,
):
    """
    Render a synthetic disk image of the star as seen by the observer.

    Projects visible elements onto the plane of the sky and fills a
    pixel grid using nearest-neighbour assignment.

    Parameters
    ----------
    grid : SurfaceGrid
    quantity : str
        Stored scalar quantity to colormap.
    line_of_sight : (x, y, z) unit vector
    resolution : int
        Image size in pixels (resolution × resolution).
    cmap, title, colorbar_label, ax, **imshow_kw
        Passed to matplotlib.

    Returns
    -------
    fig, ax
    """
    plt, _ = _require_matplotlib()

    # Build an orthonormal frame: los ≡ z-axis, pick arbitrary x-y axes.
    lx, ly, lz = line_of_sight
    # x-axis perpendicular to los in the x-z plane
    if abs(lx) < 0.9:
        px, py, pz = 1.0, 0.0, 0.0
    else:
        px, py, pz = 0.0, 1.0, 0.0

    # Right-handed orthonormal basis
    def cross(a, b):
        return (
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0],
        )

    def normalize(v):
        n = math.sqrt(sum(x**2 for x in v))
        return tuple(x / n for x in v)

    ey = normalize(cross(line_of_sight, (px, py, pz)))
    ex = normalize(cross(ey, line_of_sight))

    vis = [e for e in grid if e.is_visible(line_of_sight)]
    if not vis:
        print("No visible elements for the given line of sight.")
        return None, None

    vals = np.array([e.get(quantity) for e in vis], dtype=float)
    xs = np.array([sum(e.cartesian[k]*ex[k] for k in range(3)) for e in vis])
    ys = np.array([sum(e.cartesian[k]*ey[k] for k in range(3)) for e in vis])

    r = grid.radius
    # Build image
    img = np.full((resolution, resolution), np.nan)
    ix = ((xs / r + 1.0) * 0.5 * (resolution - 1)).astype(int)
    iy = ((ys / r + 1.0) * 0.5 * (resolution - 1)).astype(int)
    ix = np.clip(ix, 0, resolution - 1)
    iy = np.clip(iy, 0, resolution - 1)
    for i, (xi, yi) in enumerate(zip(ix, iy)):
        img[yi, xi] = vals[i]

    # Mask outside stellar disk
    yg, xg = np.mgrid[0:resolution, 0:resolution]
    cx = (xg / (resolution - 1) * 2 - 1)
    cy = (yg / (resolution - 1) * 2 - 1)
    img[cx**2 + cy**2 > 1.01] = np.nan

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    import matplotlib.cm as mpl_cm
    cmap_obj = mpl_cm.get_cmap(cmap).copy()
    cmap_obj.set_bad("black")

    im = ax.imshow(
        img,
        origin="lower",
        extent=(-r, r, -r, r),
        cmap=cmap_obj,
        **imshow_kw,
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    if colorbar_label:
        cb.set_label(colorbar_label)
    ax.set_title(title or f"{quantity} (disk view)")
    ax.set_xlabel("x / R★")
    ax.set_ylabel("y / R★")
    ax.set_aspect("equal")

    # Draw stellar limb
    theta_c = np.linspace(0, 2 * math.pi, 300)
    ax.plot(r * np.cos(theta_c), r * np.sin(theta_c), "w-", lw=0.8, alpha=0.5)

    return fig, ax


# ======================================================================== #
#  Spectrum plot                                                             #
# ======================================================================== #

def plot_transit(
    result,
    fig=None,
    contact_lines: bool = True,
):
    """
    Four-panel transit diagnostic figure:
      1. Light curve
      2. Rossiter-McLaughlin anomalous RV
      3. Sky-plane trajectory
      4. Transmission spectrum (if available)

    Parameters
    ----------
    result : TransitResult
    fig : matplotlib Figure, optional
    contact_lines : bool
        Draw vertical dashed lines at T1–T4.

    Returns
    -------
    fig
    """
    plt, _ = _require_matplotlib()

    has_spec = (
        result.transmission_spectrum is not None
        and result.wavelengths is not None
    )
    n_panels = 4 if has_spec else 3

    if fig is None:
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    else:
        axes = fig.axes

    t = result.times
    t0 = float(result.times[len(t) // 2])
    dt = (t - t0) * 24.0   # hours from mid-transit

    # ── Panel 1: Light curve ─────────────────────────────────────────────
    ax = axes[0]
    ax.plot(dt, result.flux, color="steelblue", lw=1.5)
    ax.set_xlabel("Time from mid-transit (h)")
    ax.set_ylabel("Normalised flux")
    ax.set_title("Light curve")
    ax.grid(True, alpha=0.3)
    if contact_lines:
        _draw_contacts(ax, result.contact_times, t0)

    # ── Panel 2: RM effect ───────────────────────────────────────────────
    ax = axes[1]
    ax.plot(dt, result.delta_rv, color="tomato", lw=1.5)
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("Time from mid-transit (h)")
    ax.set_ylabel("ΔRV (km/s)")
    ax.set_title("Rossiter-McLaughlin effect")
    ax.grid(True, alpha=0.3)
    if contact_lines:
        _draw_contacts(ax, result.contact_times, t0)

    # ── Panel 3: Sky-plane trajectory ────────────────────────────────────
    ax = axes[2]
    in_front = result.planet_z > 0
    ax.plot(result.planet_x[in_front], result.planet_y[in_front],
            color="steelblue", lw=2, label="Transit path")
    ax.plot(result.planet_x[~in_front], result.planet_y[~in_front],
            color="steelblue", lw=1, ls=":", alpha=0.4)

    # Mark mid-transit position
    mid = len(t) // 2
    ax.scatter([result.planet_x[mid]], [result.planet_y[mid]],
               color="navy", zorder=5, s=40)

    # Stellar disk circle
    theta_c = np.linspace(0, 2 * math.pi, 300)
    ax.plot(np.cos(theta_c), np.sin(theta_c), "k-", lw=1.5)
    ax.fill_between(np.cos(theta_c), np.sin(theta_c),
                    alpha=0.08, color="gold")
    ax.set_aspect("equal")
    ax.set_xlabel("x / R★")
    ax.set_ylabel("y / R★")
    ax.set_title("Sky-plane trajectory")
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Transmission spectrum ───────────────────────────────────
    if has_spec:
        ax = axes[3]
        in_tr = result.flux < 1.0 - 1e-5
        if in_tr.any():
            mean_depth = result.transmission_spectrum[in_tr].mean(axis=0)
        else:
            mean_depth = result.transmission_spectrum.mean(axis=0)
        ax.plot(result.wavelengths, mean_depth * 100, color="purple", lw=1.2)
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Transit depth (%)")
        ax.set_title("Transmission spectrum")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _draw_contacts(ax, contact_times, t0_days):
    """Draw T1–T4 vertical lines; x-axis is hours from t0."""
    colors = {"T1": "green", "T2": "limegreen", "T3": "limegreen", "T4": "green"}
    for label, color in colors.items():
        t = contact_times.get(label, float("nan"))
        if not math.isnan(t):
            dt_h = (t - t0_days) * 24.0
            ax.axvline(dt_h, color=color, lw=0.8, ls="--", alpha=0.7, label=label)


def plot_transit_disk_snapshot(
    star,
    transit_model,
    t: float,
    brightness_key: str = "brightness",
    cmap_star: str = "hot",
    ax=None,
):
    """
    Render the stellar disk at a single transit time with the planet overlaid.

    Parameters
    ----------
    star : Star
    transit_model : TransitModel
    t : float
        Time in days.

    Returns
    -------
    fig, ax
    """
    import matplotlib
    plt, _ = _require_matplotlib()
    from matplotlib.patches import Circle

    if ax is None:
        fig, ax_use = plt.subplots(figsize=(6, 6))
    else:
        ax_use = ax
        fig = ax.figure

    plot_disk(
        star.grid, brightness_key,
        line_of_sight=star.line_of_sight,
        cmap=cmap_star,
        ax=ax_use,
        title=f"Transit snapshot  t = {t:.4f} d",
    )

    # Overlay planet disk
    px, py, pz = transit_model.planet_position(t)
    rp = transit_model.orbit.planet_radius
    if pz > 0:
        planet_circle = Circle(
            (px * star.radius, py * star.radius), rp * star.radius,
            color="black", alpha=0.88, zorder=10
        )
        ax_use.add_patch(planet_circle)
        atm = Circle(
            (px * star.radius, py * star.radius), rp * star.radius * 1.04,
            color="navy", fill=False, lw=1.0, alpha=0.5, zorder=11
        )
        ax_use.add_patch(atm)

    return fig, ax_use


def plot_spectrum(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    title: str = "Disk-integrated spectrum",
    xlabel: str = "Wavelength (Å)",
    ylabel: str = "Normalised flux",
    ax=None,
    **plot_kw,
):
    """
    Simple line plot of a disk-integrated spectrum.

    Returns
    -------
    fig, ax
    """
    plt, _ = _require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ax.plot(wavelengths, flux, **plot_kw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig, ax
