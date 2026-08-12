"""
quantities.py — Physical quantity models for stellar surface elements.

Provides:
  * Brightness  — scalar intensity / limb-darkening models
  * Spectrum    — wavelength-dependent flux per element
  * Velocity    — line-of-sight (radial) velocity from rotation / pulsation
"""

from __future__ import annotations
import math
from typing import Callable

import numpy as np

from .element import SurfaceElement
from .grid import SurfaceGrid


# ======================================================================== #
#  Brightness                                                               #
# ======================================================================== #

class Brightness:
    """
    Assigns a scalar intensity to each surface element using a
    limb-darkening law or an arbitrary user function.

    Supported built-in laws
    -----------------------
    ``"uniform"``
        I(μ) = I₀  (no limb darkening)
    ``"linear"``
        I(μ) = I₀ · (1 - u · (1 - μ))
    ``"quadratic"``
        I(μ) = I₀ · (1 - a·(1-μ) - b·(1-μ)²)
    ``"sqrt"``
        I(μ) = I₀ · (1 - c·(1-μ) - d·(1-√μ))
    ``"claret4"``
        Four-term (Claret 2000) law.
    ``"custom"``
        Supply your own callable ``f(element) -> float``.

    Parameters
    ----------
    law : str
        One of the names above.
    I0 : float
        Central intensity (μ = 1).
    coefficients : dict
        Law-specific coefficients (see examples below).
    custom_func : callable, optional
        Only used when ``law="custom"``.
    """

    _LAWS = {"uniform", "linear", "quadratic", "sqrt", "claret4", "custom"}

    def __init__(
        self,
        law: str = "linear",
        I0: float = 1.0,
        coefficients: dict | None = None,
        custom_func: Callable[[SurfaceElement], float] | None = None,
    ) -> None:
        if law not in self._LAWS:
            raise ValueError(f"Unknown law '{law}'.  Choose from {self._LAWS}")
        self.law = law
        self.I0 = I0
        self.coefficients: dict = coefficients or {}
        self._custom_func = custom_func

    # ------------------------------------------------------------------ #

    def at(
        self,
        element: SurfaceElement,
        line_of_sight: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> float:
        """Return the intensity for a single element."""
        mu = max(element.mu(line_of_sight), 0.0)

        if self.law == "uniform":
            return self.I0

        elif self.law == "linear":
            u = self.coefficients.get("u", 0.6)
            return self.I0 * (1.0 - u * (1.0 - mu))

        elif self.law == "quadratic":
            a = self.coefficients.get("a", 0.4)
            b = self.coefficients.get("b", 0.26)
            return self.I0 * (1.0 - a * (1.0 - mu) - b * (1.0 - mu) ** 2)

        elif self.law == "sqrt":
            c = self.coefficients.get("c", 0.46)
            d = self.coefficients.get("d", 0.07)
            return self.I0 * (1.0 - c * (1.0 - mu) - d * (1.0 - math.sqrt(mu)))

        elif self.law == "claret4":
            a1 = self.coefficients.get("a1", 0.5)
            a2 = self.coefficients.get("a2", -0.2)
            a3 = self.coefficients.get("a3", 0.3)
            a4 = self.coefficients.get("a4", -0.1)
            return self.I0 * (
                1.0
                - a1 * (1 - mu ** 0.5)
                - a2 * (1 - mu)
                - a3 * (1 - mu ** 1.5)
                - a4 * (1 - mu ** 2)
            )

        else:  # custom
            if self._custom_func is None:
                raise ValueError("law='custom' requires a custom_func.")
            return self._custom_func(element)

    def apply(
        self,
        grid: SurfaceGrid,
        name: str = "brightness",
        line_of_sight: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        """Compute and store brightness on every element of *grid*."""
        for elem in grid:
            elem.set(name, self.at(elem, line_of_sight))

    def __repr__(self) -> str:
        return f"Brightness(law='{self.law}', I0={self.I0}, coeffs={self.coefficients})"


# ======================================================================== #
#  Spectrum                                                                 #
# ======================================================================== #

class Spectrum:
    """
    A wavelength-dependent spectral model for each surface element.

    The spectrum at a given element is built from a *continuum* function
    plus optional *spectral lines* (Gaussian absorption or emission).

    Parameters
    ----------
    wavelengths : array_like
        Wavelength grid in Ångströms (or any consistent unit).
    continuum : callable, optional
        ``f(element, wavelengths) -> array``  returning the continuum flux
        at each wavelength for the given element.  Defaults to a
        Planck-like blackbody if ``T_eff_key`` is set on the element, or
        a flat spectrum otherwise.
    T_eff_key : str, optional
        Name of the effective temperature quantity stored on elements
        (used by the default continuum).
    """

    def __init__(
        self,
        wavelengths: np.ndarray,
        continuum: Callable[[SurfaceElement, np.ndarray], np.ndarray] | None = None,
        T_eff_key: str = "T_eff",
    ) -> None:
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        self._continuum = continuum
        self.T_eff_key = T_eff_key
        self._lines: list[dict] = []

    # ------------------------------------------------------------------ #
    #  Line management                                                     #
    # ------------------------------------------------------------------ #

    def add_line(
        self,
        center: float,
        depth: float = 0.5,
        width: float = 2.0,
        kind: str = "absorption",
    ) -> "Spectrum":
        """
        Add a Gaussian spectral line.

        Parameters
        ----------
        center : float
            Line centre wavelength (same units as *wavelengths*).
        depth : float
            Peak depth (absorption) or height (emission) relative to continuum.
        width : float
            Gaussian sigma in wavelength units.
        kind : {"absorption", "emission"}

        Returns
        -------
        self  (for method chaining)
        """
        self._lines.append(
            {"center": center, "depth": depth, "width": width, "kind": kind}
        )
        return self

    def load_template(
        self,
        path: str,
        wl_col: int = 0,
        flux_col: int = 1,
        wl_unit: str = "angstrom",
        delimiter: str | None = None,
        skip_rows: int = 0,
    ) -> "Spectrum":
        """
        Load a high-resolution stellar template spectrum from a data file
        to be used as the CCF cross-correlation mask.

        The template replaces the internal Gaussian line-mask when computing
        CCFs.  It should be a rest-frame intrinsic stellar spectrum
        (normalised or un-normalised), provided in the star's rest frame.

        Supported formats
        -----------------
        * **Two-column ASCII / CSV** — whitespace or delimiter-separated
          columns of (wavelength, flux).  Comment lines starting with ``#``
          are ignored automatically.
        * **FITS** (requires ``astropy``) — the first FITS table or image
          extension containing wavelength and flux columns is read.

        Parameters
        ----------
        path : str
            Path to the template file.
        wl_col : int
            Column index for wavelength (default 0).
        flux_col : int
            Column index for flux (default 1).
        wl_unit : {"angstrom", "nm", "micron"}
            Unit of the wavelength column.  The template is always stored
            internally in Ångströms.
        delimiter : str, optional
            Column separator for ASCII files.  None = any whitespace.
        skip_rows : int
            Number of header rows to skip (in addition to ``#`` comments).

        Returns
        -------
        self  (for method chaining)

        Notes
        -----
        The template wavelength grid does **not** need to match the model
        wavelength grid — the CCF engine interpolates the template at each
        Doppler-shifted position.  A resolution of ~0.01 Å (R ~ 100 000)
        is sufficient to resolve individual lines; coarser grids work but
        produce broader effective masks.
        """
        import os

        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Template spectrum file not found: {path!r}")

        # ── FITS ──────────────────────────────────────────────────────────────
        if path.lower().endswith((".fits", ".fit", ".fits.gz")):
            try:
                from astropy.io import fits as _fits
                from astropy.table import Table as _Table
                with _fits.open(path) as hdul:
                    # Try binary table first
                    for hdu in hdul[1:]:
                        if hdu.data is not None:
                            try:
                                t = _Table(hdu.data)
                                cols = list(t.colnames)
                                wl_arr   = np.asarray(t[cols[wl_col]],   dtype=float)
                                flux_arr = np.asarray(t[cols[flux_col]], dtype=float)
                                break
                            except Exception:
                                continue
                    else:
                        # Fall back to image extension
                        wl_arr   = np.asarray(hdul[wl_col].data,   dtype=float).ravel()
                        flux_arr = np.asarray(hdul[flux_col].data, dtype=float).ravel()
            except ImportError as e:
                raise ImportError(
                    "astropy is required to load FITS template spectra. "
                    "Install it with: pip install astropy"
                ) from e

        # ── ASCII / CSV ───────────────────────────────────────────────────────
        else:
            data = np.loadtxt(
                path,
                comments="#",
                delimiter=delimiter,
                skiprows=skip_rows,
            )
            if data.ndim == 1:
                raise ValueError(
                    f"Template file {path!r} appears to have only one column. "
                    "Expected at least two columns: wavelength and flux."
                )
            wl_arr   = data[:, wl_col].astype(float)
            flux_arr = data[:, flux_col].astype(float)

        # ── Unit conversion ───────────────────────────────────────────────────
        _factors = {"angstrom": 1.0, "nm": 10.0, "micron": 1e4}
        if wl_unit not in _factors:
            raise ValueError(
                f"Unknown wl_unit '{wl_unit}'. "
                f"Choose from: {list(_factors)}"
            )
        wl_arr = wl_arr * _factors[wl_unit]

        # Sort by wavelength
        order = np.argsort(wl_arr)
        self._template_wl   = wl_arr[order]
        self._template_flux = flux_arr[order]

        # Normalise template to unit maximum so CCF depth is independent
        # of the absolute flux scale of the input file.
        mx = self._template_flux.max()
        if mx > 0:
            self._template_flux = self._template_flux / mx

        return self

    @property
    def has_template(self) -> bool:
        """True if an external template spectrum has been loaded."""
        return (hasattr(self, "_template_wl")
                and self._template_wl is not None
                and len(self._template_wl) > 0)

    def _default_continuum(self, element: SurfaceElement) -> np.ndarray:
        """Simple blackbody approximation or flat spectrum."""
        if element.has(self.T_eff_key):
            T = element.get(self.T_eff_key)
            return _planck_relative(self.wavelengths, T)
        return np.ones_like(self.wavelengths)

    def at(self, element: SurfaceElement, doppler_shift: float = 0.0) -> np.ndarray:
        """
        Return the spectrum (flux vs wavelength) at a single element.

        Respects the ``"line_depth_factor"`` quantity stored on the element
        by surface features: depths of all registered lines are scaled by this
        factor before computing the spectrum.

        * ``line_depth_factor > 1`` → deeper lines (cool starspot, molecular bands)
        * ``line_depth_factor < 1`` → shallower lines (hot faculae, plages)
        * ``line_depth_factor = 1`` → photospheric (default)

        Parameters
        ----------
        element : SurfaceElement
        doppler_shift : float
            Radial velocity in km/s used to Doppler-shift spectral lines.
        """
        if self._continuum is not None:
            flux = self._continuum(element, self.wavelengths).copy()
        else:
            flux = self._default_continuum(element).copy()

        c_kms = 2.998e5

        # Line-depth scale factor from surface features (1.0 if not set)
        ldf = float(element.get("line_depth_factor")) if element.has("line_depth_factor") else 1.0
        ldf = max(ldf, 0.)   # guard against negative values

        for line in self._lines:
            shift          = line["center"] * doppler_shift / c_kms
            shifted_center = line["center"] + shift
            profile = np.exp(
                -0.5 * ((self.wavelengths - shifted_center) / line["width"]) ** 2
            )
            depth_scaled = line["depth"] * ldf
            if line["kind"] == "absorption":
                flux *= 1.0 - depth_scaled * profile
            else:
                flux += depth_scaled * flux * profile

        return flux

    def apply(
        self,
        grid: SurfaceGrid,
        name: str = "spectrum",
        velocity_key: str | None = None,
    ) -> None:
        """
        Compute and store spectra on all elements.

        Parameters
        ----------
        grid : SurfaceGrid
        name : str
            Key under which the spectrum array is stored on each element.
        velocity_key : str, optional
            If set, retrieve the Doppler velocity (km/s) from each element
            and apply it when computing spectral lines.
        """
        for elem in grid:
            v = elem.get(velocity_key) if (velocity_key and elem.has(velocity_key)) else 0.0
            elem.set(name, self.at(elem, doppler_shift=v))

    def compute_ccf(
        self,
        disk_flux: np.ndarray,
        rv_grid: np.ndarray,
        template_fwhm: float = 10.0,
    ) -> np.ndarray:
        """
        Compute the Cross-Correlation Function (CCF) of a disk-integrated
        spectrum against a line-mask template.

        Two modes are supported:

        **Gaussian-line mask** (default)
            A sum of Gaussian profiles, one per registered spectral line,
            evaluated on the model wavelength grid.  The FWHM in velocity
            space is set by *template_fwhm*.

        **External high-resolution template** (activated by calling
        :meth:`load_template` before this method)
            The loaded stellar template spectrum is used instead.  It is
            interpolated onto the Doppler-shifted model wavelength grid at
            each RV step, which correctly handles templates at any
            resolution.

        In both cases the CCF is:

            CCF(v) = Σ_λ  F(λ) · T_v(λ)  /  ||T||²

        where T_v is the template shifted to velocity v.

        Parameters
        ----------
        disk_flux : np.ndarray, shape (n_wavelengths,)
            Disk-integrated flux at the current epoch.
        rv_grid : np.ndarray, shape (n_rv,)
            RV shifts in km/s.
        template_fwhm : float
            FWHM of the Gaussian mask in km/s (ignored when an external
            template has been loaded via :meth:`load_template`).

        Returns
        -------
        np.ndarray, shape (n_rv,)
        """
        c_kms = 2.998e5
        wl    = self.wavelengths
        disk_flux = np.asarray(disk_flux, dtype=float)

        # ── External template mode ─────────────────────────────────────────
        if self.has_template:
            twl  = self._template_wl
            tflx = self._template_flux

            # Normalise the template at zero shift
            t0 = np.interp(wl, twl, tflx, left=0., right=0.)
            # Invert: CCF peaks where spectrum dips match template dips.
            # We correlate (1 - F) against (1 - T) for absorption-line stars.
            f_inv = 1.0 - disk_flux / (disk_flux.max() or 1.0)
            t0_inv = 1.0 - t0
            norm = np.dot(t0_inv, t0_inv)
            if norm <= 0:
                return np.ones(len(rv_grid))

            ccf = np.empty(len(rv_grid))
            for j, rv in enumerate(rv_grid):
                # Shift template: redshift wl by rv → evaluate template at shorter λ
                wl_shifted = wl / (1.0 + rv / c_kms)
                t_shifted = np.interp(wl_shifted, twl, tflx, left=0., right=0.)
                t_shifted_inv = 1.0 - t_shifted
                ccf[j] = np.dot(f_inv, t_shifted_inv) / norm
            return ccf

        # ── Gaussian-line mask mode ────────────────────────────────────────
        sigma_kms = template_fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))

        template = np.zeros_like(wl)
        for line in self._lines:
            lc = line["center"]
            sigma_wl = sigma_kms * lc / c_kms
            template += line["depth"] * np.exp(-0.5 * ((wl - lc) / sigma_wl) ** 2)

        if template.sum() == 0:
            return np.ones(len(rv_grid))

        template_norm = np.dot(template, template)

        ccf = np.empty(len(rv_grid))
        for j, rv in enumerate(rv_grid):
            template_shifted = np.zeros_like(wl)
            for line in self._lines:
                lc   = line["center"]
                lc_s = lc * (1.0 + rv / c_kms)
                sigma_wl = sigma_kms * lc / c_kms
                template_shifted += line["depth"] * np.exp(
                    -0.5 * ((wl - lc_s) / sigma_wl) ** 2
                )
            ccf[j] = np.dot(disk_flux, template_shifted) / max(template_norm, 1e-30)

        return ccf

    def disk_integrated(
        self,
        grid: SurfaceGrid,
        name: str = "spectrum",
        line_of_sight: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> np.ndarray:
        """
        Return the area-and-μ-weighted disk-integrated spectrum.

        Elements must already have had their spectra stored via :meth:`apply`.
        """
        vis = grid.visible_elements(line_of_sight)
        if not vis:
            return np.zeros_like(self.wavelengths)

        total = np.zeros_like(self.wavelengths)
        w_sum = 0.0
        for e in vis:
            mu = e.mu(line_of_sight)
            w = e.area * mu
            total += e.get(name) * w
            w_sum += w

        return total / w_sum if w_sum else total

    def __repr__(self) -> str:
        wl = self.wavelengths
        return (
            f"Spectrum(λ=[{wl[0]:.0f}…{wl[-1]:.0f}] Å, "
            f"n_λ={len(wl)}, n_lines={len(self._lines)})"
        )


# ======================================================================== #
#  Velocity                                                                 #
# ======================================================================== #

class Velocity:
    """
    Computes the line-of-sight (radial) velocity for each surface element.

    Three contributions can be superimposed:

    * **Rotation**  — solid-body or differential (surface-shear) rotation,
                      with full spin-orbit obliquity support.
    * **Pulsation** — radial pulsation with a user-defined amplitude map.
    * **Custom**    — arbitrary function of the element.

    Parameters
    ----------
    v_eq : float
        Equatorial rotational velocity (km/s).
    inclination : float
        Inclination of the stellar rotation axis to the line-of-sight
        (degrees).  0° = pole-on, 90° = equator-on.
    obliquity : float
        Spin-orbit obliquity λ (degrees).  The angle between the stellar
        rotation axis projected on the sky and the normal to the orbital
        plane (i.e. the transit chord direction on the sky).

        * λ = 0°  → aligned: stellar equator runs parallel to transit chord.
        * λ = 90° → perpendicular: pole points along transit chord.
        * λ = 180° → retrograde alignment.

        Obliquity rotates the stellar spin axis in the sky plane (around
        the line-of-sight), producing the characteristic RM anomaly shape
        that distinguishes aligned from misaligned systems.
    differential_rotation : float
        Surface differential-rotation parameter α such that
        Ω(θ) = Ω_eq · (1 − α · sin²(latitude)).  Sun: α ≈ 0.2.
    """

    def __init__(
        self,
        v_eq: float = 0.0,
        inclination: float = 90.0,
        obliquity: float = 0.0,
        differential_rotation: float = 0.0,
    ) -> None:
        self.v_eq = v_eq
        self.inclination_rad = math.radians(inclination)
        self.obliquity_rad   = math.radians(obliquity)
        self.alpha = differential_rotation

        self._pulsation_func: Callable[[SurfaceElement], float] | None = None
        self._custom_func: Callable[[SurfaceElement], float] | None = None

    @property
    def rotation_axis(self) -> tuple[float, float, float]:
        """
        Unit vector of the stellar rotation axis in the world frame.

        Derived from inclination i and sky-plane obliquity λ:

            pole = (sin i · sin λ,  −sin i · cos λ,  cos i)

        This satisfies  v_los = v_eq · (pole × n)_z, which expands to:

            v_los = v_eq · sin i · sin θ_elem · cos(φ_elem − λ)

        The lat/lon grid in the viewer uses this pole so that parallels and
        meridians are always aligned with the actual velocity pattern.
        """
        i = self.inclination_rad
        lam = self.obliquity_rad
        return (
            math.sin(i) * math.sin(lam),
            -math.sin(i) * math.cos(lam),
            math.cos(i),
        )

    # ------------------------------------------------------------------ #
    #  Optional contributions                                              #
    # ------------------------------------------------------------------ #

    def add_pulsation(self, func: Callable[[SurfaceElement], float]) -> "Velocity":
        """
        Add a radial-pulsation velocity contribution.

        Parameters
        ----------
        func : callable
            ``f(element) -> float`` giving the pulsation velocity (km/s)
            at each element.  Positive = expansion.

        Returns
        -------
        self
        """
        self._pulsation_func = func
        return self

    def add_custom(self, func: Callable[[SurfaceElement], float]) -> "Velocity":
        """Add an arbitrary velocity field contribution (km/s)."""
        self._custom_func = func
        return self

    # ------------------------------------------------------------------ #
    #  Computation                                                         #
    # ------------------------------------------------------------------ #

    def _rotation_velocity(
        self,
        element: SurfaceElement,
        line_of_sight: tuple[float, float, float],
    ) -> float:
        """
        Line-of-sight component of rotational velocity, including obliquity.

        The stellar rotation pole is:
            pole = (sin i · sin λ,  −sin i · cos λ,  cos i)

        The surface velocity projected onto the LOS is:
            v_los = v_eq · (pole × n)_z
                  = v_eq · sin i · sin θ_elem · cos(φ_elem − λ)

        With differential rotation the local angular velocity scales by
            Ω(lat) / Ω_eq = 1 − α · sin²(lat_from_pole)
        where lat_from_pole is the latitude measured from the obliquity-
        rotated pole (not from the grid's geometric north pole).
        """
        if self.v_eq == 0.0:
            return 0.0

        i   = self.inclination_rad
        lam = self.obliquity_rad

        # Latitude measured from the actual (obliquity-rotated) pole,
        # used for the differential-rotation law.
        if self.alpha != 0.0:
            pole = np.array(self.rotation_axis)
            n    = np.array(element.normal)
            sin_lat = float(np.dot(pole, n))          # sin(latitude from pole)
            sin_lat = max(-1.0, min(1.0, sin_lat))
            omega_factor = 1.0 - self.alpha * sin_lat ** 2
        else:
            omega_factor = 1.0

        # Generalised LOS projection: v_eq·sin(i)·sin(θ)·cos(φ − λ)
        v_surface = self.v_eq * omega_factor * math.sin(i)
        return v_surface * math.sin(element.theta) * math.cos(element.phi - lam)

    def at(
        self,
        element: SurfaceElement,
        line_of_sight: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> float:
        """Return the total line-of-sight velocity (km/s) at one element.

        Includes rotation, pulsation, custom contributions, and the
        convective velocity set by surface features (``v_conv`` stored
        on the element).  The convective velocity is a radial (outward)
        velocity projected onto the LOS via μ = cos(θ_los):

            v_conv_los = v_conv × μ

        Positive ``v_conv`` = downflow (redshift); negative = upflow (blueshift).
        """
        v = self._rotation_velocity(element, line_of_sight)

        if self._pulsation_func is not None:
            v_puls = self._pulsation_func(element)
            mu = element.mu(line_of_sight)
            v += v_puls * mu

        if self._custom_func is not None:
            v += self._custom_func(element)

        # Convective velocity from surface features (granulation, etc.)
        if element.has("v_conv"):
            mu = element.mu(line_of_sight)
            v += element.get("v_conv") * mu

        return v

    def apply(
        self,
        grid: SurfaceGrid,
        name: str = "velocity",
        line_of_sight: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        """Compute and store line-of-sight velocity on every element."""
        for elem in grid:
            elem.set(name, self.at(elem, line_of_sight))

    def __repr__(self) -> str:
        return (
            f"Velocity(v_eq={self.v_eq} km/s, "
            f"inc={math.degrees(self.inclination_rad):.1f}°, "
            f"λ={math.degrees(self.obliquity_rad):.1f}°, "
            f"α={self.alpha})"
        )


# ======================================================================== #
#  Helpers                                                                  #
# ======================================================================== #

def _planck_relative(wavelengths_angstrom: np.ndarray, T: float) -> np.ndarray:
    """
    Planck function B_λ(T) normalised to its peak value.

    Parameters
    ----------
    wavelengths_angstrom : array
        Wavelengths in Ångströms.
    T : float
        Temperature in Kelvin.
    """
    lam_m = wavelengths_angstrom * 1e-10          # Å → m
    h = 6.626e-34
    c = 2.998e8
    k = 1.381e-23

    exponent = (h * c) / (lam_m * k * T)
    # Clip to avoid overflow for very short λ or low T
    exponent = np.clip(exponent, 0, 700)
    B = lam_m ** -5 / (np.exp(exponent) - 1.0)
    mx = B.max()
    return B / mx if mx > 0 else B
