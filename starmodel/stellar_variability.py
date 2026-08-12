"""
stellar_variability.py — Stellar rotation variability simulator.

Computes disk-integrated observables over one or more full stellar rotation
cycles, capturing the modulation caused by surface features (starspots,
faculae, plages, granulation) as they rotate across the visible disk.

Simulated observables
---------------------
* **Photometric flux** — the stellar activity light curve (brightness modulation
  as spots darken and faculae brighten the visible hemisphere).
* **SDSS g, r, i, z differential colour light curves** — chromatic variability
  from temperature-contrast features (cool spots redden the star; hot faculae
  make it bluer).
* **Radial velocity** — activity-induced RV jitter from the suppression of
  convective blueshift in spots and from the rotational Doppler imbalance when
  asymmetric features cross the disk.
* **CCF profile evolution** — the CCF bisector shape and centroid shift as
  active regions modulate the disk-integrated line profile.

Physical model
--------------
The stellar surface is represented by the :class:`~starmodel.Star` grid with
surface features (from :mod:`starmodel.surface_features`) already applied via
:meth:`~starmodel.Star.compute`.  At each rotation phase φ (0 to 2π per
cycle), all element normals are rotated around the angular-momentum pole axis:

    n_i(φ) = R(pole_AM, φ) × n_i

where R is the Rodrigues rotation matrix.  The rotation preserves the feature
modifications (brightness, T_eff, v_conv, line_depth_factor) which are fixed
in the stellar frame; only the projection onto the line of sight changes.

The rotational LOS velocity at phase φ is:
    v_rot,i(φ) = v_eq × ω(lat_i) × (pole_AM × n_i(φ)) · los

where ω(lat) = 1 − α sin²(lat) is the differential rotation law.
The total element velocity adds the convective contribution (from granulation):
    v_i(φ) = v_rot,i(φ) + v_conv,i × μ_i(φ)

All integrals are computed as flux-weighted sums over visible elements
(μ > 0), using brightness × area × μ as weights.

CCF computation
---------------
The CCF is modelled using a Gaussian line template of given FWHM.  Each
visible element contributes a Gaussian centred at its total LOS velocity,
weighted by its brightness × line-depth factor.  This captures:

* **Bisector asymmetry** from granulation (bright blue-shifted granule centres
  dominate the blue wing of the CCF).
* **Bump from starspots** as they rotate from limb to disk centre — a
  distinctive distortion in the CCF residual map.
* **Facular fill-in** (shallower lines in hot facular gas).

Usage
-----
    from starmodel import Star, GranulationField, StarSpot, Facula
    from starmodel.stellar_variability import RotationSimulator

    # Build and configure the star (with surface features)
    star = (Star(n_theta=60, n_phi=120, name="Active star")
            .set_brightness(law="quadratic", coefficients={"a": 0.45, "b": 0.30})
            .set_rotation(v_eq=2.0, inclination=90.)
            .set_temperature_map(lambda e: 5778.))
    star.add_feature(GranulationField(n_cells=800, seed=42))
    star.add_feature(StarSpot(lat_deg=30., lon_deg=0., radius_deg=12., T_contrast=-500.))
    star.add_feature(Facula(lat_deg=30., lon_deg=40., radius_deg=8., alpha=0.10))
    star.compute()

    # Simulate over 2 full rotation cycles
    sim = RotationSimulator(star, rotation_period_days=25., n_phases_per_cycle=300)
    result = sim.run(n_cycles=2, ccf_rv_range=30., ccf_n_rv=200)

    print(result.summary())
    sim.plot(result, save_path="rotation_variability.png")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .transit import _band_flux_planck, _SDSS_BANDS


# =========================================================================== #
#  Limb-darkening vectorised evaluator                                         #
# =========================================================================== #

def _make_ld_func(star):
    """
    Return a vectorised function ``ld(mu_array) -> np.ndarray`` that evaluates
    the stellar limb-darkening law for any array of μ values.

    This is needed because the rotation simulator must recompute LD(μ) at
    every rotation phase (as elements sweep across the disk, their μ changes).
    The Brightness object stores LD only at the phase-0 LOS.

    Parameters
    ----------
    star : Star
    """
    b = star._brightness
    if b is None:
        return lambda mu: np.ones_like(np.asarray(mu, dtype=float))

    I0  = float(b.I0)
    law = b.law
    c   = b.coefficients or {}

    if law == "uniform":
        return lambda mu: np.full_like(np.asarray(mu, dtype=float), I0)

    elif law == "linear":
        u = c.get("u", 0.6)
        def _ld(mu):
            mu = np.asarray(mu, dtype=float)
            return I0 * (1. - u * (1. - mu))
        return _ld

    elif law == "quadratic":
        a_ = c.get("a", 0.4)
        b_ = c.get("b", 0.26)
        def _ld(mu):
            mu = np.asarray(mu, dtype=float)
            x  = 1. - mu
            return I0 * (1. - a_ * x - b_ * x**2)
        return _ld

    elif law == "sqrt":
        c_ = c.get("c", 0.46)
        d_ = c.get("d", 0.07)
        def _ld(mu):
            mu = np.asarray(mu, dtype=float)
            return I0 * (1. - c_ * (1. - mu) - d_ * (1. - np.sqrt(np.clip(mu, 0., 1.))))
        return _ld

    elif law == "claret4":
        a1 = c.get("a1", 0.5);  a2 = c.get("a2", -0.2)
        a3 = c.get("a3", 0.3);  a4 = c.get("a4", -0.1)
        def _ld(mu):
            mu = np.asarray(mu, dtype=float)
            return I0 * (1. - a1*(1-mu**0.5) - a2*(1-mu)
                            - a3*(1-mu**1.5)  - a4*(1-mu**2))
        return _ld

    else:
        # Custom law — fall back to flat
        return lambda mu: np.full_like(np.asarray(mu, dtype=float), I0)


# =========================================================================== #
#  Rodrigues rotation helper                                                   #
# =========================================================================== #

def _rodrigues(vectors: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate an array of vectors around *axis* by *angle* radians.

    Uses Rodrigues' rotation formula (vectorised):
        R(p̂, φ) v = v cos φ + (p̂ × v) sin φ + p̂ (p̂ · v)(1 − cos φ)

    Parameters
    ----------
    vectors : (n, 3)
    axis    : (3,) unit vector
    angle   : float  radians

    Returns
    -------
    (n, 3)
    """
    axis = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    dot  = (vectors @ axis)[:, np.newaxis]          # (n, 1)
    cross = np.cross(axis[np.newaxis, :], vectors)   # (n, 3): p̂ × v_i
    return vectors * c + cross * s + axis[np.newaxis, :] * dot * (1. - c)


# =========================================================================== #
#  Result container                                                            #
# =========================================================================== #

@dataclass
class RotationResult:
    """
    All time-series observables from a stellar rotation simulation.

    Attributes
    ----------
    times : (n,) array   — days since t=0 (0 to n_cycles × P_rot).
    phases : (n,) array  — rotation phase in cycles (0 to n_cycles).
    flux : (n,)          — normalised photometric flux.  1.0 = mean flux over
                           the first complete rotation.
    flux_diff_ppm : (n,) — differential photometry in parts-per-million
                           relative to the unspotted photosphere flux.
    color_lcs : (3, n)   — differential colour light curves in millimagnitudes:
                           rows = [Δ(g−r), Δ(r−i), Δ(i−z)].
                           Zero is the mean colour over the first rotation.
    color_names : list   — ['g−r', 'r−i', 'i−z'].
    rv : (n,)            — activity-induced radial velocity in km/s.
                           Zero is the flux-weighted mean RV of the unspotted
                           photosphere (which includes the convective blueshift).
    rv_m_s : (n,)        — same as *rv* but in m/s.
    ccf_map : (n, n_rv)  — CCF profile at each rotation phase, normalised to the
                           mean CCF peak over the first rotation.
    ccf_residual : (n, n_rv) — ccf_map − mean CCF (shows the rotating signal).
    rv_grid : (n_rv,)    — RV axis for ccf_map columns (km/s).
    ccf_mean : (n_rv,)   — mean CCF profile (reference).
    star_name : str
    P_rot : float        — rotation period (days).
    n_cycles : float
    """

    times:         np.ndarray
    phases:        np.ndarray
    flux:          np.ndarray
    flux_diff_ppm: np.ndarray
    color_lcs:     np.ndarray        # (3, n)
    color_names:   list
    rv:            np.ndarray
    rv_m_s:        np.ndarray
    ccf_map:       np.ndarray        # (n, n_rv)
    ccf_residual:  np.ndarray
    rv_grid:       np.ndarray
    ccf_mean:      np.ndarray
    star_name:     str = "Star"
    P_rot:         float = 1.
    n_cycles:      float = 1.

    def summary(self) -> str:
        lines = [
            f"RotationResult — {self.star_name}",
            f"  Rotation period : {self.P_rot:.2f} days",
            f"  Cycles          : {self.n_cycles}",
            f"  Time steps      : {len(self.times)}",
            f"  Flux peak-to-peak : {(self.flux.max()-self.flux.min())*1e6:.0f} ppm",
            f"  RV peak-to-peak   : {(self.rv.max()-self.rv.min())*1e3:.2f} m/s",
        ]
        for i, name in enumerate(self.color_names):
            pk = (self.color_lcs[i].max() - self.color_lcs[i].min())
            lines.append(f"  Δ({name}) pk-pk       : {pk:.4f} mmag")
        return "\n".join(lines)


# =========================================================================== #
#  Simulator                                                                   #
# =========================================================================== #

class RotationSimulator:
    """
    Simulates stellar variability over full rotation cycles.

    Parameters
    ----------
    star : Star
        Fully configured and ``compute()``-d :class:`~starmodel.Star`.
        Surface features should already be added and baked into element
        quantities via :meth:`~starmodel.Star.compute`.
    rotation_period_days : float
        Stellar rotation period in days.
    n_phases_per_cycle : int
        Number of evenly-spaced rotation phases per cycle (default 300).
        Higher → smoother curves; 200–500 is usually sufficient.
    """

    def __init__(
        self,
        star,
        rotation_period_days: float,
        n_phases_per_cycle: int = 300,
    ) -> None:
        self.star   = star
        self.P_rot  = rotation_period_days
        self.n_pp   = n_phases_per_cycle
        self._precompute()

    # ------------------------------------------------------------------ #
    #  Pre-computation: extract fixed stellar quantities                   #
    # ------------------------------------------------------------------ #

    def _precompute(self) -> None:
        """
        Extract and cache all element quantities that are fixed in the
        stellar frame.  Called once at construction.

        Key design: the limb-darkening brightness stored on each element was
        computed at the phase-0 line-of-sight and MUST NOT be used directly
        during the rotation loop.  Instead we factorise:

            brightness_element = LD(μ_i(φ=0)) × bf_feature_i

        extracting the feature brightness factor ``bf_feature_i`` by dividing
        out the phase-0 limb-darkening.  At each rotation phase φ the correct
        integrated intensity is:

            I_i(φ) = LD(μ_i(φ)) × bf_feature_i

        re-evaluated with the current μ.  This ensures the light curve is
        purely driven by the rotation of surface features and not by
        artificial limb-darkening phase noise.
        """
        star    = self.star
        grid    = star.grid
        n       = len(grid)
        los     = np.array(star.line_of_sight)
        vel     = star._velocity
        _DEFAULT_T = 5778.

        # ── Geometry ──────────────────────────────────────────────────
        self._los       = los
        self._n_elem    = n
        self._normals   = np.array([e.normal for e in grid])   # (n, 3)
        self._areas     = np.array([e.area   for e in grid])   # (n,)

        # Angular-momentum pole axis
        if vel is not None:
            self._pole_AM = np.array(vel.rotation_axis, dtype=float)
            self._pole_AM /= np.linalg.norm(self._pole_AM)
        else:
            self._pole_AM = np.array([0., 0., 1.])

        self._v_eq  = float(vel.v_eq) if vel else 0.
        self._alpha = float(vel.alpha) if vel else 0.

        # Sin²(latitude) for each element (for differential rotation)
        self._sin_lat_sq = (self._normals @ self._pole_AM) ** 2   # (n,)

        # ── Build a vectorised limb-darkening evaluator ───────────────────
        # This re-evaluates LD(μ) for any μ array without touching elements.
        self._ld_func = _make_ld_func(star)

        # ── Feature-modified element quantities ───────────────────────────
        def _get(key, default):
            return np.array([
                e.get(key) if e.has(key) else default
                for e in grid
            ])

        brightness_stored = _get("brightness", 0.)   # LD(μ_0) × bf_feature
        self._T_eff       = _get("T_eff",          _DEFAULT_T)
        self._v_conv      = _get("v_conv",             0.)
        self._ldf         = _get("line_depth_factor",  1.)

        # Divide out phase-0 limb darkening to isolate the feature factor
        mu_0   = np.clip(self._normals @ los, 0., 1.)
        ld_0   = self._ld_func(mu_0)
        # Avoid division by zero at the exact limb (ld_0 → 0)
        ld_0_safe = np.where(ld_0 > 1e-6, ld_0, 1.)
        self._bf_feature = brightness_stored / ld_0_safe   # pure feature factor

        # ── SDSS band Planck fluxes per element ───────────────────────────
        band_names = ["g", "r", "i", "z"]
        self._planck_bands = np.zeros((4, n))
        for bi, bn in enumerate(band_names):
            for i, T in enumerate(self._T_eff):
                self._planck_bands[bi, i] = _band_flux_planck(float(T), bn)
        self._band_names = band_names

    # ------------------------------------------------------------------ #
    #  Main simulation loop                                                #
    # ------------------------------------------------------------------ #

    def run(
        self,
        n_cycles:          float = 2.,
        ccf_rv_range:      float = 30.,
        ccf_n_rv:          int   = 200,
        ccf_template_fwhm: float = 8.,
        t_start:           float = 0.,
    ) -> RotationResult:
        """
        Run the rotation simulation.

        Parameters
        ----------
        n_cycles : float
            Number of rotation cycles to simulate (default 2).
        ccf_rv_range : float
            Half-width of the CCF RV axis in km/s (default 30).
        ccf_n_rv : int
            Number of CCF RV grid points (default 200).
        ccf_template_fwhm : float
            FWHM of the Gaussian CCF template in km/s (default 8).
        t_start : float
            Start time in days (default 0).

        Returns
        -------
        RotationResult
        """
        n_total   = int(n_cycles * self.n_pp)
        phi_arr   = np.linspace(0., 2. * math.pi * n_cycles, n_total, endpoint=False)
        times     = t_start + np.linspace(0., self.P_rot * n_cycles, n_total, endpoint=False)
        phases    = phi_arr / (2. * math.pi)

        rv_grid   = np.linspace(-ccf_rv_range, ccf_rv_range, ccf_n_rv)
        sigma_ccf = ccf_template_fwhm / (2. * math.sqrt(2. * math.log(2.)))

        los        = self._los
        pole       = self._pole_AM
        normals0   = self._normals
        areas      = self._areas
        bf_feature = self._bf_feature      # feature brightness factor (no LD)
        ld_func    = self._ld_func         # LD(mu_array)
        v_conv     = self._v_conv
        ldf        = self._ldf
        sin_lat_sq = self._sin_lat_sq
        v_eq       = self._v_eq
        alpha      = self._alpha
        planck_b   = self._planck_bands    # (4, n)

        # Storage
        flux       = np.zeros(n_total)
        band_lc    = np.zeros((4, n_total))
        rv_arr     = np.zeros(n_total)
        ccf_map    = np.zeros((n_total, ccf_n_rv))

        # Differential rotation factor (fixed in stellar frame)
        omega = 1. - alpha * sin_lat_sq      # (n,)

        # ── Main loop ─────────────────────────────────────────────────────
        for ti, phi in enumerate(phi_arr):
            # ── Rotate normals by φ around pole_AM ────────────────────────
            n_rot = _rodrigues(normals0, pole, phi)    # (n, 3)

            # ── Visibility and limb angle ──────────────────────────────────
            mu_raw = n_rot @ los                        # (n,) — signed
            vis    = mu_raw > 0.
            mu     = np.where(vis, mu_raw, 0.)          # (n,) — clamped

            # ── Limb-darkening at current phase (the key fix) ──────────────
            ld_phi = ld_func(np.clip(mu, 0., 1.))       # (n,) LD(μ(φ))

            # Physical intensity: LD × feature_factor
            intensity = ld_phi * bf_feature             # (n,)

            # ── Rotational LOS velocity ────────────────────────────────────
            cross_pn = np.cross(pole[np.newaxis, :], n_rot)   # (n, 3)
            v_rot    = v_eq * omega * (cross_pn @ los)         # (n,)
            v_total  = v_rot + v_conv * mu                     # (n,)

            # ── Flux weights: intensity × area × μ ────────────────────────
            w = intensity * areas * mu       # (n,)
            w = np.where(vis, w, 0.)
            w_sum = float(w.sum())

            flux[ti] = w_sum

            # ── Band fluxes ────────────────────────────────────────────────
            for bi in range(4):
                band_lc[bi, ti] = float((planck_b[bi] * w).sum())

            # ── Activity RV ────────────────────────────────────────────────
            rv_arr[ti] = float((v_total * w).sum()) / w_sum if w_sum > 0 else 0.

            # ── CCF (Gaussian template, fully vectorised) ──────────────────
            if vis.any():
                v_vis  = v_total[vis]
                w_ldf  = (w * ldf)[vis]
                dv     = rv_grid[:, np.newaxis] - v_vis[np.newaxis, :]
                gauss  = np.exp(-0.5 * (dv / sigma_ccf) ** 2)
                ccf_map[ti] = (gauss * w_ldf[np.newaxis, :]).sum(axis=1)

        # ── Normalisation ──────────────────────────────────────────────────
        n_first   = self.n_pp
        flux_ref  = float(flux[:n_first].mean()) or 1.
        flux_norm = flux / flux_ref

        # Differential photometry in ppm relative to first-cycle mean
        flux_diff_ppm = (flux_norm - 1.) * 1e6

        # Band flux reference (mean over first cycle)
        band_ref = band_lc[:, :n_first].mean(axis=1)
        band_ref = np.where(band_ref > 0, band_ref, 1.)
        band_norm = band_lc / band_ref[:, np.newaxis]
        with np.errstate(divide="ignore", invalid="ignore"):
            band_mag = -2.5 * np.log10(np.clip(band_norm, 1e-10, None))
        color_lcs = np.array([
            band_mag[0] - band_mag[1],
            band_mag[1] - band_mag[2],
            band_mag[2] - band_mag[3],
        ]) * 1000.   # → millimagnitudes

        # RV: subtract first-cycle mean
        rv_mean     = float(rv_arr[:n_first].mean())
        rv_activity = rv_arr - rv_mean

        # CCF
        ccf_mean  = ccf_map[:n_first].mean(axis=0)
        ccf_peak  = float(ccf_mean.max()) or 1.
        ccf_norm  = ccf_map / ccf_peak
        ccf_mean_n = ccf_mean / ccf_peak
        ccf_resid  = ccf_norm - ccf_mean_n[np.newaxis, :]

        return RotationResult(
            times         = times,
            phases        = phases,
            flux          = flux_norm,
            flux_diff_ppm = flux_diff_ppm,
            color_lcs     = color_lcs,
            color_names   = ["g−r", "r−i", "i−z"],
            rv            = rv_activity,
            rv_m_s        = rv_activity * 1000.,
            ccf_map       = ccf_norm,
            ccf_residual  = ccf_resid,
            rv_grid       = rv_grid,
            ccf_mean      = ccf_mean_n,
            star_name     = self.star.name,
            P_rot         = self.P_rot,
            n_cycles      = n_cycles,
        )

    # ------------------------------------------------------------------ #
    #  Plotting                                                            #
    # ------------------------------------------------------------------ #

    def plot(
        self,
        result: RotationResult,
        save_path: str | None = None,
        figsize: tuple = (14, 16),
        cmap_ccf: str = "RdBu_r",
        time_unit: str = "days",
    ) -> "matplotlib.figure.Figure":
        """
        Four-panel rotation variability figure.

        Panels
        ------
        1. Differential photometric light curve (ppm)
        2. SDSS g−r, r−i, i−z colour light curves (mmag)
        3. Activity-induced RV (m/s)
        4. CCF residual map (phase × RV velocity)

        Parameters
        ----------
        result : RotationResult
        save_path : str, optional  — save figure to this path.
        figsize : (width, height)
        cmap_ccf : str             — diverging colormap for CCF residuals.
        time_unit : {"days", "phase"}
                                   — x-axis unit.
        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.cm as mpl_cm
        import matplotlib.gridspec as gridspec

        matplotlib.rcParams.update({
            "font.size": 10, "axes.titlesize": 10,
            "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        })

        C_bg   = "#0d0d0d"
        C_card = "#111111"
        C_bord = "#444444"
        C_text = "#bbbbbb"
        C_ttl  = "#eeeeee"

        if time_unit == "phase":
            x_arr   = result.phases
            x_label = f"Rotation phase  (P_rot = {result.P_rot:.1f} d)"
        else:
            x_arr   = result.times
            x_label = "Time  (days)"

        fig = plt.figure(figsize=figsize, facecolor=C_bg)
        gs  = gridspec.GridSpec(
            4, 1, figure=fig,
            hspace=0.40, top=0.94, bottom=0.06, left=0.10, right=0.96,
            height_ratios=[1., 1., 1., 1.4],
        )
        axes = [fig.add_subplot(gs[i]) for i in range(4)]

        for ax in axes:
            ax.set_facecolor(C_card)
            for sp in ax.spines.values(): sp.set_edgecolor(C_bord)
            ax.tick_params(colors=C_text)
            ax.xaxis.label.set_color(C_text)
            ax.yaxis.label.set_color(C_text)
            ax.title.set_color(C_ttl)
            ax.grid(True, alpha=0.15, color="#555555")

        # ── Panel 1: Photometric LC ────────────────────────────────────────
        ax = axes[0]
        ax.plot(x_arr, result.flux_diff_ppm, color="#5599ff", lw=1.3)
        ax.axhline(0., color="#555", lw=0.7, ls="--")
        ax.set_ylabel("ΔFlux  (ppm)")
        ax.set_title("Differential photometric light curve")
        ax.set_xlim(x_arr[0], x_arr[-1])
        self._draw_rotation_markers(ax, result, x_arr)

        # ── Panel 2: Colour LCs ────────────────────────────────────────────
        ax = axes[1]
        cols = ["#66aaff", "#ffaa33", "#cc55ff"]
        for i, (name, col) in enumerate(zip(result.color_names, cols)):
            ax.plot(x_arr, result.color_lcs[i], color=col, lw=1.3, label=name)
        ax.axhline(0., color="#555", lw=0.7, ls="--")
        ax.set_ylabel("Δcolour  (mmag)")
        ax.set_title("SDSS differential colour light curves")
        ax.legend(fontsize=8, facecolor="#1a1a1a", edgecolor=C_bord,
                  labelcolor=C_ttl, loc="upper right", ncol=3)
        ax.set_xlim(x_arr[0], x_arr[-1])
        self._draw_rotation_markers(ax, result, x_arr)

        # ── Panel 3: RV ────────────────────────────────────────────────────
        ax = axes[2]
        ax.plot(x_arr, result.rv_m_s, color="#ff6655", lw=1.3)
        ax.axhline(0., color="#555", lw=0.7, ls="--")
        ax.set_ylabel("ΔRV  (m/s)")
        ax.set_title("Activity-induced radial velocity")
        ax.set_xlim(x_arr[0], x_arr[-1])
        self._draw_rotation_markers(ax, result, x_arr)

        # ── Panel 4: CCF residual map ─────────────────────────────────────
        ax  = axes[3]
        res = result.ccf_residual
        in_cycle = result.phases < result.n_cycles
        sig  = float(np.std(res[result.phases < 1.])) if result.n_cycles >= 1 else 1e-4
        vlim = max(3. * sig, 1e-4)

        cmap_obj = mpl_cm.get_cmap(cmap_ccf).copy()
        cmap_obj.set_bad(C_bg)

        im = ax.imshow(
            res.T,
            origin="lower", aspect="auto",
            extent=[x_arr[0], x_arr[-1], result.rv_grid[0], result.rv_grid[-1]],
            cmap=cmap_obj, vmin=-vlim, vmax=vlim,
            interpolation="bilinear",
        )
        # Overplot mean CCF as a hairline guide
        rv_span  = result.rv_grid[-1] - result.rv_grid[0]
        ccf_n    = result.ccf_mean - result.ccf_mean.min()
        if ccf_n.max() > 0: ccf_n /= ccf_n.max()
        ccf_scaled = result.rv_grid[0] + ccf_n * rv_span
        for t_val in x_arr[::max(1, len(x_arr) // 10)]:
            ax.plot([t_val] * len(result.rv_grid), ccf_scaled,
                    color="#ffcc66", lw=0.4, alpha=0.15)
        ax.axhline(0., color="#888", lw=0.7, ls=":", alpha=0.5)

        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.015, aspect=25)
        cb.set_label("ΔCCF  (spot shadow)", color=C_text, fontsize=8)
        cb.ax.yaxis.set_tick_params(color=C_text, labelsize=7)
        cb.outline.set_edgecolor(C_bord)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=C_text)

        ax.set_xlabel(x_label)
        ax.set_ylabel("RV  (km/s)")
        ax.set_title("CCF residual map  (ΔCCF = CCF − mean CCF)")
        ax.tick_params(colors=C_text)
        self._draw_rotation_markers(ax, result, x_arr)

        # Global title
        v_eq = float(self.star._velocity.v_eq) if self.star._velocity else 0.
        lam  = (math.degrees(self.star._velocity.obliquity_rad)
                if self.star._velocity else 0.)
        fig.suptitle(
            f"{result.star_name}  —  Stellar rotation variability\n"
            f"P_rot = {result.P_rot:.1f} d  │  v sin i = {v_eq:.1f} km/s  │  "
            f"λ = {lam:+.1f}°  │  {result.n_cycles:.0f} cycles",
            color=C_ttl, fontsize=11, fontweight="bold", y=0.975,
        )

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=C_bg)

        return fig

    @staticmethod
    def _draw_rotation_markers(ax, result, x_arr):
        """Draw vertical dashed lines at each full rotation period."""
        for k in range(1, int(result.n_cycles) + 1):
            if result.P_rot * k <= x_arr[-1]:
                xv = (result.times[0] + k * result.P_rot
                      if hasattr(result, '_t_start')
                      else x_arr[0] + (k / result.n_cycles) * (x_arr[-1] - x_arr[0]))
                # Use phase axis
                xv = x_arr[int(round(k * result.n_cycles / result.n_cycles
                                     * (len(x_arr) - 1) / result.n_cycles))]
                ax.axvline(xv, color="#44ff88", lw=0.8, ls="--", alpha=0.45,
                           label=f"P_rot × {k}" if k == 1 else None)

    def plot_disk_snapshots(
        self,
        result: RotationResult,
        phases_deg: list[float] | None = None,
        save_path: str | None = None,
        figsize: tuple = (16, 8),
    ) -> "matplotlib.figure.Figure":
        """
        Show brightness disk maps at several rotation phases.

        Parameters
        ----------
        phases_deg : list of float, optional
            Rotation angles in degrees to snapshot (default [0, 90, 180, 270]).
        save_path : str, optional
        figsize : tuple

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.cm as mpl_cm
        import matplotlib.gridspec as gridspec
        from scipy.ndimage import gaussian_filter

        if phases_deg is None:
            phases_deg = [0., 90., 180., 270.]

        C_bg = "#0d0d0d"
        fig = plt.figure(figsize=figsize, facecolor=C_bg)
        n_snap = len(phases_deg)
        gs = gridspec.GridSpec(
            1, n_snap, figure=fig, wspace=0.12,
            top=0.88, bottom=0.06, left=0.04, right=0.96,
        )
        los      = self._los
        normals0 = self._normals
        bf_feat  = self._bf_feature
        ld_func  = self._ld_func
        areas    = self._areas
        pole     = self._pole_AM
        r        = self.star.radius
        res      = 200

        # Sky-plane projection helpers
        def _sky_frame(los):
            lx, ly, lz = los
            ref = np.array([1., 0., 0.]) if abs(lx) < 0.9 else np.array([0., 1., 0.])
            ey  = np.cross(los, ref); ey /= np.linalg.norm(ey)
            ex  = np.cross(ey, los); ex /= np.linalg.norm(ex)
            return ex, ey

        ex, ey = _sky_frame(los)

        for col_i, phi_deg in enumerate(phases_deg):
            phi     = math.radians(phi_deg)
            n_rot   = _rodrigues(normals0, pole, phi)
            mu      = n_rot @ los
            xs      = (n_rot @ ex) * r
            ys      = (n_rot @ ey) * r

            # Physical intensity: LD(μ) × feature_factor (same as simulator)
            mu_clip = np.clip(mu, 0., 1.)
            ld_phi  = ld_func(mu_clip)
            intensity = ld_phi * bf_feat   # (n,)

            # Rasterise
            from scipy.interpolate import griddata
            vis  = mu > 0.
            vals = (intensity * mu_clip)[vis]
            xi   = np.linspace(-r, r, res)
            yi   = np.linspace(-r, r, res)
            xg, yg = np.meshgrid(xi, yi)
            disk   = xg**2 + yg**2 <= (r * 0.999)**2

            img = np.full((res, res), np.nan)
            if vis.any():
                interp = griddata(
                    np.column_stack([xs[vis], ys[vis]]), vals,
                    (xg[disk], yg[disk]), method="linear"
                )
                img[disk] = interp
                nans_in = disk & np.isnan(img)
                if nans_in.any():
                    fill = griddata(
                        np.column_stack([xs[vis], ys[vis]]), vals,
                        (xg[nans_in], yg[nans_in]), method="nearest"
                    )
                    img[nans_in] = fill
            img = gaussian_filter(np.nan_to_num(img, nan=0.), res / 120.)
            outside = xg**2 + yg**2 > 1.005 * r**2
            img[outside] = np.nan

            ax = fig.add_subplot(gs[0, col_i])
            ax.set_facecolor("#0d0d0d")
            cm = mpl_cm.get_cmap("afmhot").copy()
            cm.set_bad("#0d0d0d")
            ax.imshow(img, origin="lower", extent=(-r, r, -r, r),
                      cmap=cm, aspect="equal", interpolation="bilinear",
                      vmin=0., vmax=float(np.nanmax(img)) if np.any(~np.isnan(img)) else 1.)
            th = np.linspace(0, 2 * math.pi, 400)
            ax.plot(r * np.cos(th), r * np.sin(th), color="#aaaaaa", lw=0.8, alpha=0.7)
            ax.set_xlim(-r * 1.15, r * 1.15)
            ax.set_ylim(-r * 1.15, r * 1.15)
            ax.set_title(f"φ = {phi_deg:.0f}°", color="#eeeeee", fontsize=10)
            ax.set_xlabel("x / R★", color="#bbbbbb"); ax.tick_params(colors="#bbbbbb")
            if col_i == 0: ax.set_ylabel("y / R★", color="#bbbbbb")

        fig.suptitle(f"{self.star.name}  —  Brightness maps at different rotation phases",
                     color="#eeeeee", fontsize=11, fontweight="bold")
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=C_bg)
        return fig
