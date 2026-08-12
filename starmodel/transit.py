"""
transit.py — Planetary transit model for starmodel.

Models a spherical, opaque planet crossing the stellar disk.  For each
time step the module computes:

  * Which surface elements are occulted by the planet.
  * The transit light curve (normalised flux vs time).
  * The Rossiter-McLaughlin (RM) effect (anomalous radial velocity).
  * The transmission spectrum (wavelength-dependent transit depth).
  * The contact times T1–T4 and key geometric parameters.

Physical model
--------------
The orbit is a Keplerian ellipse described by the standard parameters
(a, e, ω, i, Ω, P, t0).  The sky-plane position of the planet centre
is computed from the true anomaly at each requested time.  Occultation
of a surface element is decided by comparing the 2-D separation (on the
plane of the sky) between the element's projected centre and the planet
centre with the planet's radius.

Units
-----
All lengths are in units of the *stellar radius* R★ unless stated.
Velocities are in km/s, times in days, angles in degrees.

References
----------
Winn (2010), "Transits and Occultations", arXiv:1001.2010
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .star import Star
from .element import SurfaceElement


# =========================================================================== #
#  SDSS passband definitions                                                   #
# =========================================================================== #
#
# Approximate SDSS g, r, i, z transmission curves as (wavelength_Å, T) pairs.
# Constructed from the Doi et al. (2010) / Gunn et al. (1998) filter profiles
# sampled at ~50 Å resolution and normalised to unit peak.
# These are "throughput × atmosphere × CCD" total system responses.

def _make_sdss_bands() -> dict[str, np.ndarray]:
    """Return dict of {band_name: (wavelengths_Å, transmission)} arrays."""
    # Wavelength nodes and trapezoid weights for each filter.
    # Shape: each row = (wl_Å, T_norm) sampled at key wavelengths.
    _g = np.array([
        [3600, 0.000], [3800, 0.003], [4000, 0.050], [4100, 0.180],
        [4300, 0.600], [4500, 0.900], [4700, 0.980], [4900, 1.000],
        [5000, 0.980], [5100, 0.820], [5200, 0.500], [5300, 0.200],
        [5400, 0.050], [5500, 0.005], [5600, 0.000],
    ])
    _r = np.array([
        [5400, 0.000], [5500, 0.008], [5600, 0.070], [5700, 0.350],
        [5800, 0.700], [5900, 0.900], [6000, 0.970], [6100, 1.000],
        [6200, 1.000], [6300, 0.980], [6400, 0.920], [6500, 0.800],
        [6600, 0.600], [6700, 0.350], [6800, 0.120], [6900, 0.030],
        [7000, 0.005], [7100, 0.000],
    ])
    _i = np.array([
        [6700, 0.000], [6800, 0.005], [6900, 0.040], [7000, 0.200],
        [7100, 0.550], [7200, 0.820], [7300, 0.940], [7400, 1.000],
        [7500, 1.000], [7600, 0.950], [7700, 0.880], [7800, 0.820],
        [7900, 0.750], [8000, 0.680], [8100, 0.580], [8200, 0.380],
        [8300, 0.150], [8400, 0.040], [8500, 0.005], [8600, 0.000],
    ])
    _z = np.array([
        [8300, 0.000], [8400, 0.005], [8500, 0.040], [8600, 0.150],
        [8700, 0.380], [8800, 0.650], [8900, 0.820], [9000, 0.900],
        [9100, 0.940], [9200, 0.950], [9300, 0.920], [9500, 0.880],
        [9700, 0.800], [9900, 0.680], [10100, 0.500], [10300, 0.300],
        [10500, 0.120], [10700, 0.030], [10900, 0.005], [11000, 0.000],
    ])
    return {
        "g": (_g[:, 0], _g[:, 1]),
        "r": (_r[:, 0], _r[:, 1]),
        "i": (_i[:, 0], _i[:, 1]),
        "z": (_z[:, 0], _z[:, 1]),
    }

_SDSS_BANDS = _make_sdss_bands()


def _band_flux_planck(T_K: float, band: str) -> float:
    """
    Compute the band-integrated Planck flux for a surface element.

    Returns ∫ B_λ(T) · S(λ) dλ where S is the SDSS passband throughput.
    The limb-darkening brightness weight is applied separately in the caller.

    Parameters
    ----------
    T_K : float
        Effective temperature in Kelvin.
    band : str
        One of 'g', 'r', 'i', 'z'.

    Returns
    -------
    float
        Band-integrated Planck flux in arbitrary linear units.
    """
    wl_nodes, T_nodes = _SDSS_BANDS[band]
    wl = np.linspace(wl_nodes[0], wl_nodes[-1], 400)
    S  = np.interp(wl, wl_nodes, T_nodes)
    lam_m = wl * 1e-10
    h, c_ms, k_B = 6.626e-34, 2.998e8, 1.381e-23
    exponent = np.clip(h * c_ms / (lam_m * k_B * T_K), 0., 700.)
    B = lam_m ** -5 / (np.exp(exponent) - 1.0)
    integrand = B * S
    try:
        result = float(np.trapezoid(integrand, wl))
    except AttributeError:
        result = float(np.trapz(integrand, wl))
    return max(result, 1e-300)


def _sky_frame_from_los(line_of_sight):
    """Return orthonormal (ex, ey) sky-plane basis for the given LOS."""
    lx, ly, lz = line_of_sight

    def _cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

    def _norm(v):
        n = math.sqrt(sum(c**2 for c in v))
        return tuple(c/n for c in v)

    ref = (1., 0., 0.) if abs(lx) < 0.9 else (0., 1., 0.)
    ey = _norm(_cross(line_of_sight, ref))
    ex = _norm(_cross(ey, line_of_sight))
    return ex, ey


# =========================================================================== #
#  Orbital parameters dataclass                                                #
# =========================================================================== #

@dataclass
class OrbitalParameters:
    """
    Keplerian orbital parameters of a transiting planet.

    All angles in degrees; lengths in stellar radii; times in days.

    Parameters
    ----------
    period : float
        Orbital period P (days).
    t0 : float
        Time of mid-transit (days).
    semi_major_axis : float
        Orbital semi-major axis a / R★.
    inclination : float
        Orbital inclination i (degrees).  90° = edge-on transit.
    eccentricity : float
        Orbital eccentricity e.  0 = circular.
    omega : float
        Argument of periastron ω (degrees).  Ignored when e = 0.
    Omega : float
        Longitude of ascending node Ω (degrees).  Defines the
        orientation of the transit chord on the sky plane.
    planet_radius : float
        Planet radius Rp / R★.
    impact_parameter : float | None
        If given, overrides *inclination* so that the mid-transit
        impact parameter b = a·cos(i) / R★ matches this value.
    obliquity : float
        Spin-orbit obliquity λ (degrees).  Stored here for
        documentation and summary output.  The physical effect on
        the velocity field is applied by setting the same value in
        ``Star.set_rotation(obliquity=λ)``.
    """

    period: float = 3.0
    t0: float = 0.0
    semi_major_axis: float = 10.0
    inclination: float = 90.0
    eccentricity: float = 0.0
    omega: float = 90.0
    Omega: float = 0.0
    planet_radius: float = 0.1
    impact_parameter: float | None = None
    obliquity: float = 0.0          # spin-orbit obliquity λ (degrees)

    def __post_init__(self) -> None:
        if self.impact_parameter is not None:
            # Derive inclination from b = a * cos(i)
            cos_i = self.impact_parameter / self.semi_major_axis
            cos_i = max(-1.0, min(1.0, cos_i))
            self.inclination = math.degrees(math.acos(cos_i))

    # ------------------------------------------------------------------ #

    @property
    def inc_rad(self) -> float:
        return math.radians(self.inclination)

    @property
    def omega_rad(self) -> float:
        return math.radians(self.omega)

    @property
    def Omega_rad(self) -> float:
        return math.radians(self.Omega)

    @property
    def b(self) -> float:
        """Mid-transit impact parameter b = a·cos(i)·(1-e²)/(1+e·sin(ω))."""
        factor = (1 - self.eccentricity ** 2) / (
            1 + self.eccentricity * math.sin(self.omega_rad)
        )
        return self.semi_major_axis * math.cos(self.inc_rad) * factor

    @property
    def transit_duration_approx(self) -> float:
        """
        Approximate transit duration T14 (days), ignoring eccentricity
        corrections (Winn 2010, eq. 14).
        """
        a = self.semi_major_axis
        rp = self.planet_radius
        b = self.b
        if a <= 0:
            return 0.0
        arg = ((1 + rp) ** 2 - b ** 2)
        if arg <= 0:
            return 0.0
        return (self.period / math.pi) * math.asin(math.sqrt(arg) / a)

    def summary(self) -> str:
        lines = [
            f"  Period         : {self.period:.4f} days",
            f"  t0 (mid-transit): {self.t0:.4f} days",
            f"  a / R★         : {self.semi_major_axis:.2f}",
            f"  Inclination    : {self.inclination:.3f}°",
            f"  Eccentricity   : {self.eccentricity:.4f}",
            f"  ω              : {self.omega:.2f}°",
            f"  Ω              : {self.Omega:.2f}°",
            f"  Rp / R★        : {self.planet_radius:.4f}",
            f"  Impact param b : {self.b:.4f}",
            f"  Obliquity λ    : {self.obliquity:.2f}°",
            f"  T14 (approx)   : {self.transit_duration_approx*24:.3f} h",
        ]
        return "\n".join(lines)


# =========================================================================== #
#  Planet sky position                                                         #
# =========================================================================== #

def _true_anomaly(M: float, e: float, tol: float = 1e-10) -> float:
    """
    Solve Kepler's equation M = E - e·sin(E) for E, return true anomaly f.

    Parameters
    ----------
    M : float
        Mean anomaly (radians).
    e : float
        Eccentricity.
    tol : float
        Convergence tolerance.
    """
    # Eccentric anomaly via Newton-Raphson
    E = M if e < 0.8 else math.pi
    for _ in range(100):
        dE = (M - E + e * math.sin(E)) / (1.0 - e * math.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    # True anomaly
    f = 2.0 * math.atan2(
        math.sqrt(1 + e) * math.sin(E / 2),
        math.sqrt(1 - e) * math.cos(E / 2),
    )
    return f


def planet_sky_position(
    t: float,
    orb: OrbitalParameters,
) -> tuple[float, float, float]:
    """
    Return the planet centre position in the sky plane at time *t*.

    The coordinate system has the star at the origin, x-axis pointing
    toward the observer's right (East on sky), y-axis pointing North,
    and z-axis pointing toward the observer.

    Returns
    -------
    (x_sky, y_sky, z) in units of R★.
    z > 0 means the planet is in front of the star (transiting side).
    """
    # Mean motion
    n = 2 * math.pi / orb.period
    # Mean anomaly (referenced to t0 = time of inferior conjunction)
    # At inferior conjunction (transit), true anomaly f_conj = π/2 - ω
    f_conj = math.pi / 2.0 - orb.omega_rad
    # Eccentric anomaly at conjunction
    E_conj = 2.0 * math.atan2(
        math.sqrt(1 - orb.eccentricity) * math.sin(f_conj / 2),
        math.sqrt(1 + orb.eccentricity) * math.cos(f_conj / 2),
    )
    M_conj = E_conj - orb.eccentricity * math.sin(E_conj)
    M = n * (t - orb.t0) + M_conj

    f = _true_anomaly(M % (2 * math.pi), orb.eccentricity)

    # Distance from focus
    r = orb.semi_major_axis * (1 - orb.eccentricity ** 2) / (
        1 + orb.eccentricity * math.cos(f)
    )

    # Position in orbital plane (x_orb toward periastron, z_orb = normal)
    x_orb = r * math.cos(f)
    y_orb = r * math.sin(f)

    # Rotate to sky frame:
    # 1. rotate by ω around z-orbital
    # 2. rotate by i around x
    # 3. rotate by Ω around z-sky
    cos_o = math.cos(orb.omega_rad)
    sin_o = math.sin(orb.omega_rad)
    cos_i = math.cos(orb.inc_rad)
    sin_i = math.sin(orb.inc_rad)
    cos_O = math.cos(orb.Omega_rad)
    sin_O = math.sin(orb.Omega_rad)

    # After ω rotation
    xw = x_orb * cos_o - y_orb * sin_o
    yw = x_orb * sin_o + y_orb * cos_o

    # After i rotation (tilt into 3D)
    xi = xw
    yi = yw * cos_i
    zi = yw * sin_i   # z positive toward observer (planet in front at inferior conjunction)

    # After Ω rotation (sky-plane orientation)
    # Sign convention: x_sky increases to the right as seen by the observer.
    # For a prograde orbit the planet moves from East to West on the sky,
    # i.e. from negative x to positive x with increasing time.
    x_sky = -(xi * cos_O - yi * sin_O)   # negated so planet moves +x with time
    y_sky =   xi * sin_O + yi * cos_O
    z_sky = zi

    return x_sky, y_sky, z_sky


# =========================================================================== #
#  Occultation mask                                                            #
# =========================================================================== #

def occultation_mask(
    star: Star,
    planet_x: float,
    planet_y: float,
    planet_radius: float,
) -> np.ndarray:
    """
    Boolean array (len = n_elements) — True where an element is occulted.

    An element is considered occulted if:
      1. It is on the visible hemisphere (μ > 0).
      2. Its projected sky-plane centre falls within the planet disk.

    Parameters
    ----------
    star : Star
    planet_x, planet_y : float
        Planet centre in sky coordinates (R★ units).
    planet_radius : float
        Planet radius in R★ units.

    Returns
    -------
    np.ndarray[bool], shape (n_elements,)
    """
    los = star.line_of_sight

    # Build orthonormal sky-plane frame (same as visualization.plot_disk)
    lx, ly, lz = los

    def cross(a, b):
        return (
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0],
        )

    def normalize(v):
        n = math.sqrt(sum(c**2 for c in v))
        return tuple(c / n for c in v)

    if abs(lx) < 0.9:
        ref = (1.0, 0.0, 0.0)
    else:
        ref = (0.0, 1.0, 0.0)

    ey = normalize(cross(los, ref))
    ex = normalize(cross(ey, los))

    mask = np.zeros(len(star.grid), dtype=bool)
    rp2 = planet_radius ** 2

    for idx, elem in enumerate(star.grid):
        mu = elem.mu(los)
        if mu <= 0.0:
            continue
        cx, cy, cz = elem.cartesian
        # Project onto sky plane
        ex_proj = cx * ex[0] + cy * ex[1] + cz * ex[2]
        ey_proj = cx * ey[0] + cy * ey[1] + cz * ey[2]
        dx = ex_proj - planet_x
        dy = ey_proj - planet_y
        if dx * dx + dy * dy <= rp2:
            mask[idx] = True

    return mask


# =========================================================================== #
#  TransitModel                                                                #
# =========================================================================== #

@dataclass
class TransitResult:
    """Container for results returned by :meth:`TransitModel.compute`."""

    times: np.ndarray
    """Time array (days)."""

    planet_x: np.ndarray
    """Planet sky-x position vs time (R★)."""

    planet_y: np.ndarray
    """Planet sky-y position vs time (R★)."""

    planet_z: np.ndarray
    """Planet sky-z (>0 = in front of star) vs time (R★)."""

    flux: np.ndarray
    """Normalised light curve."""

    delta_rv: np.ndarray
    """Rossiter-McLaughlin anomalous RV (km/s)."""

    occulted_fraction: np.ndarray
    """Fraction of the visible stellar disk area occulted at each time."""

    transmission_spectrum: np.ndarray | None = None
    """
    2-D array (n_times × n_wavelengths) of transit depth per wavelength.
    Only computed when ``compute_spectrum=True`` and a Spectrum model exists.
    """

    contact_times: dict = field(default_factory=dict)
    """T1, T2, T3, T4 contact times (days); NaN if not found."""

    wavelengths: np.ndarray | None = None
    """Wavelength grid (Å) if a spectrum was computed."""

    ccf_map: np.ndarray | None = None
    """
    2-D CCF map, shape (n_times, n_rv).
    Each row is the disk-integrated CCF at that time step,
    normalised by the out-of-transit CCF peak.
    Only present when ``compute_ccf=True`` in :meth:`TransitModel.compute`.
    The CCF bump (planet shadow) moves through the profile during transit,
    tracing the Rossiter-McLaughlin effect in CCF space.
    """

    rv_grid: np.ndarray | None = None
    """RV grid in km/s corresponding to the columns of *ccf_map*."""

    ccf_oot: np.ndarray | None = None
    """Out-of-transit reference CCF (1-D), normalised to unit peak."""

    ld_map: np.ndarray | None = None
    """
    Limb-darkening contrast map, shape (n_times, n_y).
    Each row holds the brightness-weighted mean intensity of the stellar
    strip occulted by the planet at that time step, sampled as a function
    of the sky-plane y-coordinate (perpendicular to the transit chord).
    This shows how the limb-darkening profile is sampled along the chord
    and links the local brightness directly to the photometric depth.
    Always computed (no flag required) as it is cheap.
    """

    ld_y_grid: np.ndarray | None = None
    """Sky-plane y-coordinate grid (in R★) for the columns of *ld_map*."""

    color_lcs: np.ndarray | None = None
    """
    Differential colour light curves, shape (3, n_times).
    Rows correspond to SDSS g−r, r−i, i−z respectively.
    Each value is the instantaneous colour index (band1 − band2) in
    magnitudes relative to the out-of-transit baseline (Δmag = 0 OOT).
    Positive Δ(g−r) means the occulted region is bluer than the disk average.
    Always computed when the star has a T_eff map or a global T_eff is set.
    """

    color_names: list | None = None
    """Labels for the rows of *color_lcs*: ['g−r', 'r−i', 'i−z']."""

    def summary(self) -> str:
        in_transit = np.sum(self.flux < 1.0 - 1e-6)
        depth = 1.0 - float(np.nanmin(self.flux))
        rv_amp = float(np.nanmax(np.abs(self.delta_rv)))
        lines = [
            f"  Time steps     : {len(self.times)}",
            f"  In-transit pts : {in_transit}",
            f"  Transit depth  : {depth*100:.4f} %  ({depth*1e6:.0f} ppm)",
            f"  RM amplitude   : {rv_amp:.4f} km/s",
        ]
        ct = self.contact_times
        if ct:
            for k in ("T1", "T2", "T3", "T4"):
                v = ct.get(k, float("nan"))
                lines.append(f"  {k}             : {v:.6f} days")
        return "\n".join(lines)


class TransitModel:
    """
    Models a spherical planet transiting the stellar disk.

    Parameters
    ----------
    star : Star
        A configured and computed :class:`~starmodel.Star` instance.
    orbit : OrbitalParameters
        Orbital parameters of the planet.

    Examples
    --------
    >>> from starmodel import Star
    >>> from starmodel.transit import TransitModel, OrbitalParameters
    >>>
    >>> star = Star(n_theta=36, n_phi=72)
    >>> star.set_brightness(law="quadratic", coefficients={"a": 0.45, "b": 0.30})
    >>> star.set_rotation(v_eq=5.0, inclination=90.0)
    >>> star.compute()
    >>>
    >>> orbit = OrbitalParameters(
    ...     period=3.5,
    ...     t0=0.0,
    ...     semi_major_axis=12.0,
    ...     inclination=89.5,
    ...     planet_radius=0.1,
    ... )
    >>> model = TransitModel(star, orbit)
    >>> result = model.compute(n_times=500)
    >>> print(result.summary())
    """

    def __init__(self, star: Star, orbit: OrbitalParameters) -> None:
        self.star = star
        self.orbit = orbit

    # ------------------------------------------------------------------ #
    #  Main computation                                                    #
    # ------------------------------------------------------------------ #

    def compute(
        self,
        times: np.ndarray | None = None,
        n_times: int = 500,
        time_window: float | None = None,
        compute_spectrum: bool = False,
        compute_ccf: bool = False,
        ccf_rv_range: float = 50.0,
        ccf_n_rv: int = 200,
        ccf_template_fwhm: float = 10.0,
        brightness_key: str = "brightness",
        velocity_key: str = "velocity",
    ) -> TransitResult:
        """
        Compute the transit observables over a time array.

        Parameters
        ----------
        times : array_like, optional
            Explicit time array (days).  If None, a symmetric window
            around t0 of width *time_window* (or 1.5 × T14) is used.
        n_times : int
            Number of time steps when *times* is auto-generated.
        time_window : float, optional
            Half-width of the auto time array (days).  Default: 1.5 × T14.
        compute_spectrum : bool
            Also compute the wavelength-dependent transit depth.
        compute_ccf : bool
            Compute the full CCF map (n_times × n_rv).  Requires a
            Spectrum model with at least one spectral line.
            The map captures the planet shadow moving through the CCF
            during transit, directly tracing the RM effect in CCF space.
        ccf_rv_range : float
            Half-width of the RV grid for the CCF in km/s (default 50).
        ccf_n_rv : int
            Number of RV points in the CCF grid (default 200).
        ccf_template_fwhm : float
            FWHM of the Gaussian CCF template in km/s (default 10).
        brightness_key : str
            Name of the brightness quantity on each surface element.
        velocity_key : str
            Name of the velocity quantity on each surface element.

        Returns
        -------
        TransitResult
        """
        orb = self.orbit
        star = self.star

        # ── Time array ──────────────────────────────────────────────────
        if times is None:
            half = time_window or 1.5 * max(orb.transit_duration_approx, 1e-3)
            times = np.linspace(orb.t0 - half, orb.t0 + half, n_times)
        else:
            times = np.asarray(times, dtype=float)

        # ── Pre-compute out-of-transit quantities ────────────────────────
        has_brightness = all(e.has(brightness_key) for e in star.grid)
        if not has_brightness:
            raise RuntimeError(
                f"Surface elements do not have '{brightness_key}'. "
                "Call star.compute() first."
            )

        has_velocity = all(e.has(velocity_key) for e in star.grid)

        los = star.line_of_sight
        vis_elements = star.grid.visible_elements(los)

        # Flux weights: brightness × area × μ
        weights = np.array([
            e.get(brightness_key) * e.area * e.mu(los)
            for e in vis_elements
        ])
        vis_indices = [i for i, e in enumerate(star.grid) if e.is_visible(los)]
        total_flux_oot = float(weights.sum())

        # Velocity array for RM
        if has_velocity:
            vel_arr = np.array([e.get(velocity_key) for e in vis_elements])
        else:
            vel_arr = np.zeros(len(vis_elements))

        # Spectrum arrays for transmission spectrum and CCF
        has_spectrum_data = (
            star._spectrum is not None
            and all(e.has("spectrum") for e in star.grid)
        )

        if (compute_spectrum or compute_ccf) and has_spectrum_data:
            wl       = star._spectrum.wavelengths
            spec_arr = np.array([e.get("spectrum") for e in vis_elements])
            # Out-of-transit disk spectrum (weighted sum)
            disk_flux_oot = (spec_arr * weights[:, None]).sum(axis=0)
            if total_flux_oot > 0:
                disk_flux_oot /= total_flux_oot
        else:
            wl = None
            spec_arr = None
            disk_flux_oot = None

        trans_spec = (
            np.zeros((len(times), len(wl)))
            if compute_spectrum and wl is not None else None
        )
        total_spec_oot = (
            (spec_arr * weights[:, None]).sum(axis=0)
            if compute_spectrum and spec_arr is not None else None
        )

        # CCF grid and out-of-transit reference
        do_ccf = compute_ccf and has_spectrum_data and star._spectrum is not None
        if do_ccf:
            rv_grid = np.linspace(-ccf_rv_range, ccf_rv_range, ccf_n_rv)
            ccf_oot = star._spectrum.compute_ccf(
                disk_flux_oot, rv_grid, ccf_template_fwhm
            )
            ccf_peak = float(np.max(np.abs(ccf_oot))) or 1.0
            ccf_oot  = ccf_oot / ccf_peak
            ccf_map  = np.zeros((len(times), ccf_n_rv))
        else:
            rv_grid = None
            ccf_oot = None
            ccf_map = None

        # ── SDSS colour light-curve pre-computation ──────────────────────
        # Per-element Planck × passband flux (no brightness — applied via weights).
        # weights[ei] = brightness(ei) × area(ei) × μ(ei), so:
        #   F_band_oot = Σ_i  planck_band(T_i) × weights_i
        #   F_band(t)  = Σ_{i not occulted}  planck_band(T_i) × weights_i
        _DEFAULT_T = 5778.
        band_names = ["g", "r", "i", "z"]

        # planck_arr shape (4, n_vis): pure Planck × passband per element per band
        planck_arr = np.zeros((4, len(vis_elements)))
        for bi, bn in enumerate(band_names):
            for ei, elem in enumerate(vis_elements):
                T = elem.get("T_eff") if elem.has("T_eff") else _DEFAULT_T
                planck_arr[bi, ei] = _band_flux_planck(T, bn)

        # OOT band fluxes = Σ planck × (brightness × area × μ)
        band_flux_oot = np.array([
            float((planck_arr[bi] * weights).sum()) for bi in range(4)
        ])
        band_flux_oot = np.where(band_flux_oot > 0, band_flux_oot, 1.)

        # Storage: per-band normalised LC (1.0 OOT)
        band_lc = np.ones((4, len(times)))

        # ── Limb-darkening contrast map pre-computation ──────────────────
        # Build sky-plane y-coordinate and brightness arrays for visible elements.
        # We bin by y-sky to build a 1-D brightness profile as a function of
        # sky-y, then sample it at each time step from the occulted elements.
        ex_sky, ey_sky = _sky_frame_from_los(los)
        ey_arr = np.array([
            sum(e.cartesian[k] * ey_sky[k] for k in range(3))
            for e in vis_elements
        ])  # sky-y coordinate of each visible element (in R★)

        ld_n_y   = 80   # resolution of the LD profile axis
        ld_y_grid = np.linspace(-star.radius, star.radius, ld_n_y)
        ld_map   = np.full((len(times), ld_n_y), np.nan)

        # Brightness array of visible elements
        bright_arr = np.array([
            e.get(brightness_key) for e in vis_elements
        ])

        # ── Loop over time ───────────────────────────────────────────────
        flux         = np.ones(len(times))
        delta_rv     = np.zeros(len(times))
        occ_fraction = np.zeros(len(times))
        px_arr = np.zeros(len(times))
        py_arr = np.zeros(len(times))
        pz_arr = np.zeros(len(times))

        for ti, t in enumerate(times):
            px, py, pz = planet_sky_position(t, orb)
            px_arr[ti] = px
            py_arr[ti] = py
            pz_arr[ti] = pz

            sep = math.sqrt(px ** 2 + py ** 2)
            if pz <= 0 or sep > (1.0 + orb.planet_radius + 0.1):
                flux[ti] = 1.0
                # OOT: CCF equals reference
                if do_ccf:
                    ccf_map[ti] = ccf_oot
                # LD map: fill with NaN for OOT (no occulted strip)
                continue

            # Occultation mask for visible elements
            occ_mask_full = occultation_mask(star, px, py, orb.planet_radius)
            occ_vis = occ_mask_full[vis_indices]

            occ_weights  = weights * occ_vis
            occ_flux     = occ_weights.sum()
            occ_fraction[ti] = occ_flux / total_flux_oot if total_flux_oot else 0.0
            flux[ti] = (total_flux_oot - occ_flux) / total_flux_oot

            # SDSS band light curves
            for bi in range(4):
                occ_band = float((planck_arr[bi] * weights * occ_vis).sum())
                band_lc[bi, ti] = (band_flux_oot[bi] - occ_band) / band_flux_oot[bi]

            # RM: flux-weighted mean velocity of un-occulted elements
            if total_flux_oot > 0 and has_velocity:
                unocc_w = weights * ~occ_vis
                w_sum   = unocc_w.sum()
                rv_in   = float((vel_arr * unocc_w).sum() / w_sum) if w_sum > 0 else 0.0
                rv_oot  = float((vel_arr * weights).sum() / total_flux_oot)
                delta_rv[ti] = rv_in - rv_oot

            # Transmission spectrum
            if compute_spectrum and spec_arr is not None and trans_spec is not None:
                occ_spec_flux = (spec_arr * occ_vis[:, None] * weights[:, None]).sum(axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    trans_spec[ti] = np.where(
                        total_spec_oot > 0,
                        occ_spec_flux / total_spec_oot,
                        0.0,
                    )

            # CCF at this time step: disk spectrum of un-occulted hemisphere
            if do_ccf and spec_arr is not None:
                unocc_w   = weights * ~occ_vis
                w_sum_unocc = unocc_w.sum()
                if w_sum_unocc > 0:
                    disk_flux_t = (spec_arr * unocc_w[:, None]).sum(axis=0) / w_sum_unocc
                else:
                    disk_flux_t = disk_flux_oot
                ccf_t = star._spectrum.compute_ccf(
                    disk_flux_t, rv_grid, ccf_template_fwhm
                )
                ccf_map[ti] = ccf_t / ccf_peak

            # LD contrast map: mean occulted brightness binned by sky-y
            if occ_vis.any():
                occ_bright = bright_arr[occ_vis]
                occ_y      = ey_arr[occ_vis]
                # Bin occulted elements into the y-grid
                bin_idx = np.searchsorted(ld_y_grid, occ_y) - 1
                bin_idx = np.clip(bin_idx, 0, ld_n_y - 1)
                for bi in range(ld_n_y):
                    mask_bin = bin_idx == bi
                    if mask_bin.any():
                        ld_map[ti, bi] = float(occ_bright[mask_bin].mean())

        # ── Differential colour light curves ─────────────────────────────
        # Convert per-band LC to magnitudes relative to OOT baseline,
        # then compute adjacent-band differences: g-r, r-i, i-z.
        # Δmag = -2.5 * log10(F_t / F_oot)  — positive when flux drops more.
        _safe = np.clip(band_lc, 1e-10, None)
        band_mag = -2.5 * np.log10(_safe)    # shape (4, n_times); all ≥ 0 in transit
        color_lcs = np.array([
            band_mag[0] - band_mag[1],   # g - r
            band_mag[1] - band_mag[2],   # r - i
            band_mag[2] - band_mag[3],   # i - z
        ])   # shape (3, n_times)

        # ── Contact times ────────────────────────────────────────────────
        contacts = self._find_contacts(times, flux, orb.planet_radius)

        return TransitResult(
            times=times,
            planet_x=px_arr,
            planet_y=py_arr,
            planet_z=pz_arr,
            flux=flux,
            delta_rv=delta_rv,
            occulted_fraction=occ_fraction,
            transmission_spectrum=trans_spec,
            contact_times=contacts,
            wavelengths=wl,
            ccf_map=ccf_map,
            rv_grid=rv_grid,
            ccf_oot=ccf_oot,
            ld_map=ld_map,
            ld_y_grid=ld_y_grid,
            color_lcs=color_lcs,
            color_names=["g−r", "r−i", "i−z"],
        )


    # ------------------------------------------------------------------ #
    #  Contact time finder                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_contacts(
        times: np.ndarray,
        flux: np.ndarray,
        rp: float,
    ) -> dict[str, float]:
        """
        Estimate T1–T4 from the light curve by threshold crossings.

        T1: first external contact  (flux starts dropping)
        T2: first internal contact  (full disk ingress complete)
        T3: last  internal contact  (egress begins)
        T4: last  external contact  (flux restored)
        """
        eps1 = 1e-4    # flux drop threshold for T1 / T4
        eps2 = rp ** 2 * 0.5   # deeper drop for T2 / T3

        contacts: dict[str, float] = {}
        drop = 1.0 - flux

        for label, thresh, direction in [
            ("T1", eps1, "first_above"),
            ("T2", eps2, "first_above"),
            ("T3", eps2, "last_above"),
            ("T4", eps1, "last_above"),
        ]:
            idx = np.where(drop > thresh)[0]
            if len(idx) == 0:
                contacts[label] = float("nan")
            elif direction == "first_above":
                contacts[label] = float(times[idx[0]])
            else:
                contacts[label] = float(times[idx[-1]])

        return contacts

    # ------------------------------------------------------------------ #
    #  Single-time helpers                                                 #
    # ------------------------------------------------------------------ #

    def planet_position(self, t: float) -> tuple[float, float, float]:
        """Return (x_sky, y_sky, z_sky) of the planet at time *t*."""
        return planet_sky_position(t, self.orbit)

    def is_transiting(self, t: float) -> bool:
        """Return True if the planet overlaps the stellar disk at time *t*."""
        px, py, pz = self.planet_position(t)
        if pz <= 0:
            return False
        sep = math.sqrt(px ** 2 + py ** 2)
        return sep < (1.0 + self.orbit.planet_radius)

    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        lam = math.degrees(self.star._velocity.obliquity_rad) if self.star._velocity else 0.0
        return (
            f"TransitModel(star='{self.star.name}', "
            f"Rp/R★={self.orbit.planet_radius:.3f}, "
            f"P={self.orbit.period:.2f}d, "
            f"b={self.orbit.b:.3f}, "
            f"λ={lam:.1f}°)"
        )
