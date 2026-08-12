"""
surface_features.py — Stellar surface feature models for starmodel.

Implements physically motivated models for:

  * GranulationField  — convective granulation pattern with upflow/downflow
                        velocity field and brightness/temperature contrast.
  * StarSpot          — cool, dark magnetic region with umbra + penumbra.
  * Facula            — bright, limb-dependent magnetic facular region.
  * Plage             — chromospheric bright region (flat contrast model).
  * SurfaceFeatureSet — container that applies all features to a star grid.

Physical propagation
--------------------
After ``Star.compute()`` each surface element stores per-element modifications:

  * ``"brightness"``        — multiplied by the feature brightness factor.
  * ``"T_eff"``             — shifted by the feature temperature delta (K).
  * ``"v_conv"``            — radial convective velocity (km/s, + = redshift).
                              Projected onto the LOS in Velocity.at().
  * ``"line_depth_factor"`` — scale factor for spectral line depths.
                              1.0 = photosphere; >1 for cool spots (deeper
                              molecular lines); <1 for hot faculae.

These propagate automatically to the CCF:
  - Spots remove flux from their region and replace it with a cooler,
    deeper-lined spectrum → produces a bump in the CCF profile.
  - Faculae add bright, hotter flux → slightly fills in spectral lines.
  - Granulation introduces a spread of convective velocities correlated
    with brightness → creates the characteristic C-shaped CCF bisector
    (bright blue-wing excess from upflowing granule centres).

Usage
-----
    from starmodel import Star
    from starmodel.surface_features import GranulationField, StarSpot, Facula

    star = (Star(n_theta=60, n_phi=120)
            .set_brightness(law="quadratic", coefficients={"a": 0.45, "b": 0.30})
            .set_rotation(v_eq=2.0, inclination=90.)
            .set_spectrum(np.linspace(6300, 6900, 400))
            .add_spectral_line(6563., depth=0.65, width=1.5)
            .set_temperature_map(lambda e: 5778.))

    star.add_feature(GranulationField(n_cells=800, seed=42))
    star.add_feature(StarSpot(lat_deg=15., lon_deg=30., radius_deg=10.,
                              T_contrast=-500.))
    star.add_feature(Facula(lat_deg=15., lon_deg=55., radius_deg=8.,
                            alpha=0.10))
    star.compute()
"""

from __future__ import annotations

import math
from typing import List

import numpy as np


# =========================================================================== #
#  Internal geometry helpers                                                   #
# =========================================================================== #

def _stellar_frame(velocity_model, line_of_sight=(0., 0., 1.)):
    """
    Return orthonormal stellar frame (pole_N, eq_x, eq_y).

    pole_N = RHR North pole = −rotation_axis  (rotation looks CCW from here)
    eq_x, eq_y = equatorial basis vectors spanning the stellar equatorial plane.

    Parameters
    ----------
    velocity_model : Velocity or None
    line_of_sight : (3,) tuple — observer direction (unit vector).
    """
    if velocity_model is None:
        pole_N = np.array([0., 0., 1.])
    else:
        pole_am = np.array(velocity_model.rotation_axis, dtype=float)
        n = np.linalg.norm(pole_am)
        pole_N = -pole_am / n if n > 0 else np.array([0., 0., 1.])

    los = np.array(line_of_sight, dtype=float)
    los = los / np.linalg.norm(los)

    # Project LOS onto equatorial plane → eq_x points toward observer
    # This makes lon = 0° always the sub-observer longitude (disk centre).
    los_eq = los - np.dot(los, pole_N) * pole_N
    los_eq_norm = np.linalg.norm(los_eq)
    if los_eq_norm > 1e-6:
        eq_x = los_eq / los_eq_norm
    else:
        # Pole-on: LOS parallel to rotation axis, choose arbitrary eq_x
        ref  = np.array([1., 0., 0.]) if abs(pole_N[0]) < 0.9 else np.array([0., 1., 0.])
        eq_x = np.cross(pole_N, ref); eq_x /= np.linalg.norm(eq_x)

    eq_y = np.cross(pole_N, eq_x); eq_y /= np.linalg.norm(eq_y)
    return pole_N, eq_x, eq_y


def _latlon_to_vec(lat_deg: float, lon_deg: float,
                   pole_N, eq_x, eq_y) -> np.ndarray:
    """Convert stellar (latitude, longitude) in degrees to a world unit vector."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    return (math.cos(lat) * math.cos(lon) * eq_x
          + math.cos(lat) * math.sin(lon) * eq_y
          + math.sin(lat) * pole_N)


def _gaussian_profile(angular_seps: np.ndarray, radius_deg: float,
                      edge_softness: float = 0.15) -> np.ndarray:
    """
    Smooth profile that is ~1 inside ``radius_deg`` and falls to 0 outside.

    Uses a Gaussian roll-off so that feature edges are not abrupt.

    Parameters
    ----------
    angular_seps : (n,) array of angular separations in radians.
    radius_deg : float — feature radius in degrees.
    edge_softness : float — fractional width of the Gaussian edge (0.15 = 15 %).
    """
    r_rad   = math.radians(radius_deg)
    sigma   = r_rad * edge_softness
    # Inside the core: profile ≈ 1; outside: Gaussian decay
    x = (angular_seps - r_rad) / sigma
    return np.where(angular_seps <= r_rad, 1.0, np.exp(-0.5 * x**2))


# =========================================================================== #
#  Base class                                                                  #
# =========================================================================== #

class SurfaceFeature:
    """
    Abstract base for stellar surface features.

    Subclasses implement ``apply()`` which returns per-element modification
    arrays.  All arrays have shape ``(n_elements,)``.
    """

    def apply(
        self,
        normals:        np.ndarray,   # (n, 3) outward unit normals
        mus:            np.ndarray,   # (n,)   cos(angle to LOS)
        T_eff_arr:      np.ndarray,   # (n,)   current T_eff (K)
        velocity_model,               # Velocity or None
        line_of_sight:  tuple = (0., 0., 1.),
    ) -> dict[str, np.ndarray]:
        """
        Compute per-element feature modifications.

        Returns
        -------
        dict with keys:
            ``brightness_factor``  : multiplicative (default 1.0).
            ``T_eff_delta``        : additive in K  (default 0.0).
            ``v_conv``             : radial velocity in km/s (default 0.0).
                                    + = redshift (downflow); − = blueshift (upflow).
            ``line_depth_factor``  : multiplicative on line depths (default 1.0).
        """
        raise NotImplementedError

    @staticmethod
    def _empty(n: int) -> dict[str, np.ndarray]:
        return {
            "brightness_factor": np.ones(n),
            "T_eff_delta":       np.zeros(n),
            "v_conv":            np.zeros(n),
            "line_depth_factor": np.ones(n),
        }


# =========================================================================== #
#  Granulation                                                                 #
# =========================================================================== #

class GranulationField(SurfaceFeature):
    """
    Statistical convective granulation pattern.

    The stellar surface is divided into Voronoi cells using randomly
    distributed seed points on the sphere.  Each cell is independently
    assigned as a **granule** (bright, rising) or an **intergranular lane**
    (dark, sinking).

    The key observable consequence is the **convective blueshift**: bright
    granule centres are blueshifted (rising gas) and contribute more flux,
    while dark lanes are redshifted (sinking gas) and contribute less.
    The net flux-weighted mean RV is therefore negative (blueshifted) by
    ~−300 m/s for a solar-like star.

    The CCF bisector acquires a characteristic **C-shape** (or reverse-C):
    the blue wing of the CCF is enhanced by the bright, blue-shifted granules,
    causing the bisector to lean toward the blue at mid-depths.

    Parameters
    ----------
    n_cells : int
        Number of granulation cells (default 800).  Higher → finer pattern.
        Solar granulation has ~10⁶ cells at any instant; 500–2000 cells
        captures the statistical velocity distribution well on a coarse grid.
    T_granule : float
        Temperature excess of granule centres in K (default +90 K).
    T_lane : float
        Temperature deficit of intergranular lanes in K (default −250 K).
    v_granule : float
        Upflow velocity of granule centres in km/s (default −0.40 km/s;
        negative = blueshift).
    v_lane : float
        Downflow velocity of intergranular lanes in km/s (default +0.80 km/s;
        positive = redshift).
    granule_fraction : float
        Fraction of cells that are granules (default 0.70).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_cells:          int   = 800,
        T_granule:        float = 90.,
        T_lane:           float = -250.,
        v_granule:        float = -0.40,
        v_lane:           float =  0.80,
        granule_fraction: float = 0.70,
        seed:             int   = 42,
    ) -> None:
        self.n_cells          = n_cells
        self.T_granule        = T_granule
        self.T_lane           = T_lane
        self.v_granule        = v_granule
        self.v_lane           = v_lane
        self.granule_fraction = granule_fraction
        self.seed             = seed
        self._build_cells()

    def _build_cells(self) -> None:
        rng = np.random.default_rng(self.seed)
        # Uniform random points on unit sphere
        phi       = rng.uniform(0., 2 * math.pi, self.n_cells)
        cos_theta = rng.uniform(-1., 1., self.n_cells)
        sin_theta = np.sqrt(1. - cos_theta**2)
        self._seeds = np.column_stack([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta,
        ])   # (n_cells, 3)
        # Assign each seed as granule (True) or lane (False)
        self._is_granule = rng.random(self.n_cells) < self.granule_fraction

    def apply(self, normals, mus, T_eff_arr, velocity_model, line_of_sight=(0.,0.,1.)):
        n = len(normals)
        out = self._empty(n)

        # Vectorised nearest-seed assignment via dot product (max → nearest)
        dots    = normals @ self._seeds.T          # (n, n_cells)
        nearest = np.argmax(dots, axis=1)          # (n,)
        is_gran = self._is_granule[nearest]        # (n,) bool

        # Temperature and velocity deltas
        dT = np.where(is_gran, self.T_granule, self.T_lane)
        vc = np.where(is_gran, self.v_granule, self.v_lane)

        # Brightness factor from temperature perturbation (Stefan-Boltzmann approx.)
        T_base  = np.where(T_eff_arr > 0, T_eff_arr, 5778.)
        T_new   = np.clip(T_base + dT, 1., None)
        bf      = (T_new / T_base) ** 4

        out["brightness_factor"] = bf
        out["T_eff_delta"]       = dT
        out["v_conv"]            = vc
        # Granules have slightly weaker (hotter) atomic lines; lanes have
        # stronger lines (cooler) — effect is small at solar-like ΔT
        out["line_depth_factor"] = np.where(is_gran, 0.97, 1.04)

        return out

    def __repr__(self) -> str:
        return (f"GranulationField(n_cells={self.n_cells}, "
                f"T_gran={self.T_granule:+.0f}K, T_lane={self.T_lane:+.0f}K, "
                f"v_gran={self.v_granule:+.3f}km/s, v_lane={self.v_lane:+.3f}km/s)")


# =========================================================================== #
#  Starspot                                                                    #
# =========================================================================== #

class StarSpot(SurfaceFeature):
    """
    Cool, dark magnetic starspot with umbra + penumbra structure.

    The spot is a circular region on the stellar surface centred at a given
    stellar latitude and longitude.  Inside the spot:

    * **Umbra** (inner ``umbra_fraction × radius``): maximum temperature contrast,
      strongest line-depth modification.
    * **Penumbra** (outer annulus): linearly interpolated contrast from umbra
      to the photosphere at the outer edge.

    Brightness and velocity
    -----------------------
    The spot brightness is reduced as:

        I_spot / I_phot ≈ (T_spot / T_phot)⁴

    This approximation holds when the bolometric correction is small.
    The spot rotates with the same angular velocity as the photosphere
    (rigid rotation), so it does not add a separate velocity contribution.
    The velocity field shows the spot as a deficit of the normal rotational
    velocity pattern when it transits.

    CCF signature
    -------------
    When a starspot transits the stellar disk the planet sequentially:
    (a) blocks photospheric flux → normal RM effect,
    (b) blocks spot flux → reduced contribution of cooler spectrum to the CCF.
    This produces the characteristic **spot-crossing bump** in the transit
    light curve and a corresponding anomaly in the CCF residual map.

    Parameters
    ----------
    lat_deg, lon_deg : float
        Centre of the spot in stellar coordinates (degrees).
        Latitude: −90° (south pole) to +90° (north pole).
        Longitude: 0°–360° measured from a reference meridian.
    radius_deg : float
        Angular radius of the entire spot (umbra + penumbra) in degrees.
    T_contrast : float
        Temperature contrast ΔT = T_spot − T_phot (K).
        Must be negative for a dark spot (typical: −300 to −2000 K).
    umbra_fraction : float
        Fraction of the spot radius that is umbra (default 0.4).
        Set to 0 for a uniform-contrast spot (no penumbra).
    umbra_T_factor : float
        Additional temperature contrast factor for the umbra relative to
        the average spot contrast (default 1.6, so the umbra is 60% cooler
        than the penumbra centre).
    line_depth_factor : float
        Scale factor for spectral line depths inside the spot.
        Default 1.5 — cool spots have deeper molecular (e.g. TiO) lines.
        Applies to the full spot; the umbra uses ``line_depth_factor × 1.2``.
    edge_softness : float
        Fractional width of the Gaussian edge roll-off (default 0.15).
    """

    def __init__(
        self,
        lat_deg:           float,
        lon_deg:           float,
        radius_deg:        float,
        T_contrast:        float = -500.,
        umbra_fraction:    float = 0.40,
        umbra_T_factor:    float = 1.60,
        line_depth_factor: float = 1.50,
        edge_softness:     float = 0.15,
    ) -> None:
        if T_contrast >= 0:
            raise ValueError("StarSpot T_contrast must be negative (cooler than photosphere).")
        self.lat_deg           = lat_deg
        self.lon_deg           = lon_deg
        self.radius_deg        = radius_deg
        self.T_contrast        = T_contrast
        self.umbra_fraction    = umbra_fraction
        self.umbra_T_factor    = umbra_T_factor
        self.line_depth_factor = line_depth_factor
        self.edge_softness     = edge_softness
        self._center_vec: np.ndarray | None = None

    def _ensure_center(self, velocity_model, line_of_sight):
        key = id(velocity_model), tuple(line_of_sight)
        if not hasattr(self, "_center_cache") or self._center_cache_key != key:
            pole_N, eq_x, eq_y = _stellar_frame(velocity_model, line_of_sight)
            self._center_vec = _latlon_to_vec(self.lat_deg, self.lon_deg, pole_N, eq_x, eq_y)
            self._center_cache_key = key

    def apply(self, normals, mus, T_eff_arr, velocity_model, line_of_sight=(0.,0.,1.)):
        n = len(normals)
        out = self._empty(n)
        self._ensure_center(velocity_model, line_of_sight)

        # Angular separations from spot centre (radians)
        dot_c = np.clip(normals @ self._center_vec, -1., 1.)
        seps  = np.arccos(dot_c)   # (n,)

        r_rad       = math.radians(self.radius_deg)
        r_umbra_rad = r_rad * self.umbra_fraction

        # Penumbra profile: 1 inside spot, softened Gaussian edge
        pen_profile = _gaussian_profile(seps, self.radius_deg, self.edge_softness)
        in_spot     = pen_profile > 1e-4

        # Temperature delta: linear from umbra centre to spot edge
        # Umbra: T_contrast × umbra_T_factor
        # Edge:  T_contrast × (1 - linear fade)
        T_base    = np.where(T_eff_arr > 0, T_eff_arr, 5778.)
        dT_umbra  = self.T_contrast * self.umbra_T_factor
        dT_pen    = self.T_contrast   # penumbra boundary value

        in_umbra  = seps <= r_umbra_rad
        in_penumb = (~in_umbra) & (seps <= r_rad * (1. + self.edge_softness * 3))

        # Interpolate linearly in the penumbra
        pen_alpha = np.zeros(n)
        if in_penumb.any():
            pen_alpha[in_penumb] = np.clip(
                (seps[in_penumb] - r_umbra_rad) / (r_rad - r_umbra_rad + 1e-30),
                0., 1.
            )

        dT = np.where(in_umbra,
                      dT_umbra,
                      np.where(in_penumb,
                               dT_umbra + (dT_pen - dT_umbra) * pen_alpha,
                               0.))
        # Apply edge softening
        dT = dT * pen_profile

        T_new  = np.clip(T_base + dT, 1., None)
        bf     = (T_new / T_base) ** 4

        # Line-depth factor
        ldf = (np.where(in_umbra,
                        self.line_depth_factor * 1.2,
                        np.where(in_penumb,
                                 1. + (self.line_depth_factor - 1.) * (1. - pen_alpha),
                                 1.))
               * pen_profile + (1. - pen_profile))

        out["brightness_factor"] = np.where(in_spot, bf, 1.)
        out["T_eff_delta"]       = dT
        out["v_conv"]            = np.zeros(n)   # spots co-rotate with photosphere
        out["line_depth_factor"] = ldf

        return out

    def __repr__(self) -> str:
        return (f"StarSpot(lat={self.lat_deg:+.1f}°, lon={self.lon_deg:.1f}°, "
                f"r={self.radius_deg:.1f}°, ΔT={self.T_contrast:+.0f}K)")


# =========================================================================== #
#  Facula                                                                      #
# =========================================================================== #

class Facula(SurfaceFeature):
    """
    Bright, limb-dependent magnetic facular region.

    Faculae are concentrations of small-scale magnetic flux that appear as
    bright structures near the solar/stellar limb.  Their contrast relative
    to the photosphere follows:

        C_fac(μ) = alpha × (1/max(μ, μ_min) − 1) × spatial_profile

    so faculae are invisible at disk centre (μ = 1) and brightest at the
    limb (μ → 0).  This reproduces the observed limb-brightening of facular
    regions in the optical.

    CCF signature
    -------------
    Faculae add brighter, slightly hotter flux near the limb.  Their spectral
    lines are marginally shallower than the photosphere (higher T_eff).  During
    transit the planet may cross facular regions, producing a slight dimming
    in the transit light curve (facular darkening when a bright region is
    blocked) and a small velocity anomaly in the CCF.

    Parameters
    ----------
    lat_deg, lon_deg : float
        Centre of the facular region in stellar coordinates.
    radius_deg : float
        Angular radius of the facula.
    alpha : float
        Facular contrast parameter (dimensionless, default 0.10).
        For the Sun, alpha ≈ 0.08–0.15 in the optical.
    T_contrast : float
        Temperature excess above the photosphere (K, default +100 K).
    mu_min : float
        Minimum μ to prevent divergence at the exact limb (default 0.05).
    line_depth_factor : float
        Scale factor for line depths inside the facula (default 0.95 —
        slightly shallower lines in hotter facular gas).
    edge_softness : float
        Fractional width of the Gaussian edge roll-off (default 0.20).
    """

    def __init__(
        self,
        lat_deg:           float,
        lon_deg:           float,
        radius_deg:        float,
        alpha:             float = 0.10,
        T_contrast:        float = 100.,
        mu_min:            float = 0.05,
        line_depth_factor: float = 0.95,
        edge_softness:     float = 0.20,
    ) -> None:
        self.lat_deg           = lat_deg
        self.lon_deg           = lon_deg
        self.radius_deg        = radius_deg
        self.alpha             = alpha
        self.T_contrast        = T_contrast
        self.mu_min            = mu_min
        self.line_depth_factor = line_depth_factor
        self.edge_softness     = edge_softness
        self._center_vec: np.ndarray | None = None

    def _ensure_center(self, velocity_model, line_of_sight):
        key = id(velocity_model), tuple(line_of_sight)
        if not hasattr(self, "_center_cache") or self._center_cache_key != key:
            pole_N, eq_x, eq_y = _stellar_frame(velocity_model, line_of_sight)
            self._center_vec = _latlon_to_vec(self.lat_deg, self.lon_deg, pole_N, eq_x, eq_y)
            self._center_cache_key = key

    def apply(self, normals, mus, T_eff_arr, velocity_model, line_of_sight=(0.,0.,1.)):
        n = len(normals)
        out = self._empty(n)
        self._ensure_center(velocity_model, line_of_sight)

        dot_c    = np.clip(normals @ self._center_vec, -1., 1.)
        seps     = np.arccos(dot_c)
        profile  = _gaussian_profile(seps, self.radius_deg, self.edge_softness)
        in_fac   = profile > 1e-4

        # Limb-dependent contrast
        mu_safe  = np.maximum(np.abs(mus), self.mu_min)
        contrast = self.alpha * (1. / mu_safe - 1.) * profile

        # Clamp to avoid numerical blow-up very near limb
        contrast = np.clip(contrast, 0., 5.0)

        out["brightness_factor"] = 1. + contrast
        out["T_eff_delta"]       = self.T_contrast * profile
        out["v_conv"]            = np.zeros(n)
        out["line_depth_factor"] = np.where(in_fac,
                                            1. + (self.line_depth_factor - 1.) * profile,
                                            1.)
        return out

    def __repr__(self) -> str:
        return (f"Facula(lat={self.lat_deg:+.1f}°, lon={self.lon_deg:.1f}°, "
                f"r={self.radius_deg:.1f}°, α={self.alpha}, ΔT={self.T_contrast:+.0f}K)")


# =========================================================================== #
#  Plage                                                                       #
# =========================================================================== #

class Plage(SurfaceFeature):
    """
    Chromospheric plage / active region (photospheric model).

    Plages are bright chromospheric regions associated with areas of enhanced
    magnetic flux.  In the photospheric model used here they appear as warm,
    uniformly brighter regions (unlike faculae, their contrast does not depend
    strongly on limb position).

    They typically surround starspots in an activity complex and produce a
    moderate brightening and temperature excess.

    Parameters
    ----------
    lat_deg, lon_deg : float
        Centre in stellar coordinates.
    radius_deg : float
        Angular radius.
    intensity_factor : float
        Brightness contrast relative to photosphere (default 1.08 → 8% brighter).
    T_contrast : float
        Temperature excess (K, default +150 K).
    line_depth_factor : float
        Line-depth scale factor (default 0.92 — plages have shallower lines).
    edge_softness : float
        Fractional width of the Gaussian edge (default 0.20).
    """

    def __init__(
        self,
        lat_deg:           float,
        lon_deg:           float,
        radius_deg:        float,
        intensity_factor:  float = 1.08,
        T_contrast:        float = 150.,
        line_depth_factor: float = 0.92,
        edge_softness:     float = 0.20,
    ) -> None:
        self.lat_deg           = lat_deg
        self.lon_deg           = lon_deg
        self.radius_deg        = radius_deg
        self.intensity_factor  = intensity_factor
        self.T_contrast        = T_contrast
        self.line_depth_factor = line_depth_factor
        self.edge_softness     = edge_softness
        self._center_vec: np.ndarray | None = None

    def _ensure_center(self, velocity_model, line_of_sight):
        key = id(velocity_model), tuple(line_of_sight)
        if not hasattr(self, "_center_cache") or self._center_cache_key != key:
            pole_N, eq_x, eq_y = _stellar_frame(velocity_model, line_of_sight)
            self._center_vec = _latlon_to_vec(self.lat_deg, self.lon_deg, pole_N, eq_x, eq_y)
            self._center_cache_key = key

    def apply(self, normals, mus, T_eff_arr, velocity_model, line_of_sight=(0.,0.,1.)):
        n = len(normals)
        out = self._empty(n)
        self._ensure_center(velocity_model, line_of_sight)

        dot_c   = np.clip(normals @ self._center_vec, -1., 1.)
        seps    = np.arccos(dot_c)
        profile = _gaussian_profile(seps, self.radius_deg, self.edge_softness)

        contrast = (self.intensity_factor - 1.) * profile
        out["brightness_factor"] = 1. + contrast
        out["T_eff_delta"]       = self.T_contrast * profile
        out["v_conv"]            = np.zeros(n)
        out["line_depth_factor"] = 1. + (self.line_depth_factor - 1.) * profile

        return out

    def __repr__(self) -> str:
        return (f"Plage(lat={self.lat_deg:+.1f}°, lon={self.lon_deg:.1f}°, "
                f"r={self.radius_deg:.1f}°, f={self.intensity_factor:.2f}, "
                f"ΔT={self.T_contrast:+.0f}K)")


# =========================================================================== #
#  SurfaceFeatureSet                                                           #
# =========================================================================== #

class SurfaceFeatureSet:
    """
    Container that applies a list of surface features to a star grid.

    Features are applied in order.  Each feature's modifications are
    *accumulated* (multiplied for factors, added for deltas).

    Parameters
    ----------
    features : list of SurfaceFeature
    """

    def __init__(self, features: List[SurfaceFeature]) -> None:
        self._features = list(features)

    def apply_to_grid(self, grid, velocity_model, line_of_sight,
                      brightness_key: str = "brightness",
                      T_eff_key:      str = "T_eff",
                      default_T:      float = 5778.) -> None:
        """
        Apply all features to every element of ``grid``.

        Modifies in-place:
          * ``brightness``        → multiplied by combined brightness_factor
          * ``T_eff``             → shifted by combined T_eff_delta
          * ``v_conv``            → set to combined convective velocity (km/s)
          * ``line_depth_factor`` → multiplied; used by Spectrum.at()

        Parameters
        ----------
        grid : SurfaceGrid
        velocity_model : Velocity or None
        line_of_sight : (3,) tuple
        brightness_key : str
        T_eff_key : str
        default_T : float
            Fallback T_eff when elements have none set.
        """
        if not self._features:
            return

        n       = len(grid)
        normals = np.array([e.normal for e in grid])  # (n, 3)
        mus     = np.array([e.mu(line_of_sight) for e in grid])
        T_eff   = np.array([
            e.get(T_eff_key) if e.has(T_eff_key) else default_T
            for e in grid
        ])
        brightness = np.array([
            e.get(brightness_key) if e.has(brightness_key) else 0.
            for e in grid
        ])

        # Accumulated modification arrays
        bf_total  = np.ones(n)
        dT_total  = np.zeros(n)
        vc_total  = np.zeros(n)
        ldf_total = np.ones(n)

        for feat in self._features:
            mods      = feat.apply(normals, mus, T_eff + dT_total, velocity_model, line_of_sight)
            bf_total  *= mods["brightness_factor"]
            dT_total  += mods["T_eff_delta"]
            vc_total  += mods["v_conv"]
            ldf_total *= mods["line_depth_factor"]

        # Write modifications back to elements
        for i, elem in enumerate(grid):
            # Brightness
            if elem.has(brightness_key):
                elem.set(brightness_key,
                         elem.get(brightness_key) * float(bf_total[i]))

            # T_eff: shift existing value (or set from default)
            T_base = elem.get(T_eff_key) if elem.has(T_eff_key) else default_T
            elem.set(T_eff_key, max(float(T_base) + float(dT_total[i]), 100.))

            # Convective velocity (will be projected onto LOS in Velocity.at())
            elem.set("v_conv", float(vc_total[i]))

            # Line-depth factor (read by Spectrum.at())
            elem.set("line_depth_factor", float(ldf_total[i]))

    def summary(self) -> str:
        if not self._features:
            return "SurfaceFeatureSet (empty)"
        lines = [f"SurfaceFeatureSet ({len(self._features)} features):"]
        for f in self._features:
            lines.append(f"  {f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"SurfaceFeatureSet([{', '.join(repr(f) for f in self._features)}])"
