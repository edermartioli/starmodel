"""
Star — Top-level model combining a SurfaceGrid with physical quantities.

This is the main entry point for most users.
"""

from __future__ import annotations
import math
from typing import Callable

import numpy as np

from .grid import SurfaceGrid
from .quantities import Brightness, Spectrum, Velocity
from .element import SurfaceElement


class Star:
    """
    A spherical stellar model with a discretized surface grid.

    Parameters
    ----------
    radius : float
        Stellar radius in arbitrary units (e.g., solar radii).
    n_theta : int
        Number of colatitude bands.
    n_phi : int
        Number of longitude cells per band (uniform grid).
    grid_type : {"uniform", "equalarea"}
        How the sphere is tiled.
    name : str
        Optional label for this star.

    Examples
    --------
    >>> from starmodel import Star
    >>> from starmodel.quantities import Brightness, Velocity
    >>>
    >>> star = Star(radius=1.0, n_theta=36, n_phi=72)
    >>> star.set_brightness(law="linear", coefficients={"u": 0.6})
    >>> star.set_rotation(v_eq=100.0, inclination=60.0)
    >>> star.compute()
    >>>
    >>> print(star.disk_flux())
    >>> print(star.grid.stats("velocity"))
    """

    def __init__(
        self,
        radius: float = 1.0,
        n_theta: int = 36,
        n_phi: int = 72,
        grid_type: str = "uniform",
        name: str = "Star",
    ) -> None:
        self.name = name
        self.radius = radius
        self.grid = SurfaceGrid(
            n_theta=n_theta,
            n_phi=n_phi,
            radius=radius,
            grid_type=grid_type,
        )

        self._brightness: Brightness | None = None
        self._spectrum: Spectrum | None = None
        self._velocity: Velocity | None = None
        self._features: list = []           # SurfaceFeature instances

        # Default line-of-sight: observer along +z axis
        self.line_of_sight: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def add_feature(self, feature) -> "Star":
        """
        Add a surface feature (starspot, facula, plage, granulation field, …).

        Features are applied during ``compute()`` in the order they are added.
        Modifications are accumulated: brightness factors multiply together,
        temperature deltas add, convective velocities add.

        Parameters
        ----------
        feature : SurfaceFeature
            Any object from :mod:`starmodel.surface_features`.

        Returns self.

        Example
        -------
        >>> from starmodel.surface_features import StarSpot, GranulationField
        >>> star.add_feature(GranulationField(n_cells=800, seed=0))
        >>> star.add_feature(StarSpot(lat_deg=20., lon_deg=45., radius_deg=10.,
        ...                           T_contrast=-600.))
        >>> star.compute()
        """
        self._features.append(feature)
        return self

    def clear_features(self) -> "Star":
        """Remove all surface features. Returns self."""
        self._features.clear()
        return self

    def list_features(self) -> list:
        """Return the current list of surface features."""
        return list(self._features)

    # ------------------------------------------------------------------ #
    #  Configuration helpers                                               #
    # ------------------------------------------------------------------ #

    def set_brightness(
        self,
        law: str = "linear",
        I0: float = 1.0,
        coefficients: dict | None = None,
        custom_func: Callable[[SurfaceElement], float] | None = None,
    ) -> "Star":
        """Configure the brightness / limb-darkening model. Returns self."""
        self._brightness = Brightness(
            law=law, I0=I0, coefficients=coefficients, custom_func=custom_func
        )
        return self

    def set_rotation(
        self,
        v_eq: float = 0.0,
        inclination: float = 90.0,
        obliquity: float = 0.0,
        differential_rotation: float = 0.0,
    ) -> "Star":
        """
        Configure stellar rotation, including spin-orbit obliquity.

        Parameters
        ----------
        v_eq : float
            Equatorial rotational velocity (km/s).
        inclination : float
            Inclination of the stellar rotation axis to the line-of-sight
            (degrees).  0° = pole-on, 90° = equator-on.
        obliquity : float
            Spin-orbit obliquity λ (degrees).  Sky-plane angle between the
            projected stellar spin axis and the normal to the orbital plane.
            λ = 0° → aligned, λ = 90° → perpendicular, λ = 180° → retrograde.
        differential_rotation : float
            Solar-like differential rotation coefficient α (0 = rigid body).

        Returns self.
        """
        if self._velocity is None:
            self._velocity = Velocity(v_eq, inclination, obliquity, differential_rotation)
        else:
            self._velocity.v_eq           = v_eq
            self._velocity.inclination_rad = math.radians(inclination)
            self._velocity.obliquity_rad   = math.radians(obliquity)
            self._velocity.alpha          = differential_rotation
        return self

    def set_pulsation(self, func: Callable[[SurfaceElement], float]) -> "Star":
        """
        Add a radial-pulsation velocity component.

        ``func(element) -> float`` returns the pulsation velocity in km/s.
        """
        if self._velocity is None:
            self._velocity = Velocity()
        self._velocity.add_pulsation(func)
        return self

    def set_spectrum(
        self,
        wavelengths: np.ndarray,
        continuum: Callable | None = None,
        T_eff_key: str = "T_eff",
    ) -> "Star":
        """Configure the spectral model. Returns self."""
        self._spectrum = Spectrum(wavelengths, continuum=continuum, T_eff_key=T_eff_key)
        return self

    def add_spectral_line(
        self,
        center: float,
        depth: float = 0.5,
        width: float = 2.0,
        kind: str = "absorption",
    ) -> "Star":
        """Add a Gaussian spectral line. Returns self."""
        if self._spectrum is None:
            raise RuntimeError("Call set_spectrum() before adding spectral lines.")
        self._spectrum.add_line(center, depth, width, kind)
        return self

    def set_temperature_map(
        self, func: Callable[[SurfaceElement], float], key: str = "T_eff"
    ) -> "Star":
        """Store a per-element effective temperature from a function."""
        self.grid.assign(key, func)
        return self

    def set_observer(
        self, direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    ) -> "Star":
        """Set the observer's line-of-sight direction (unit vector)."""
        n = math.sqrt(sum(x ** 2 for x in direction))
        self.line_of_sight = tuple(x / n for x in direction)
        return self

    # ------------------------------------------------------------------ #
    #  Computation                                                         #
    # ------------------------------------------------------------------ #

    def compute(self) -> "Star":
        """
        Evaluate all configured models and store results on the grid.

        Pipeline order
        --------------
        1. Brightness (limb darkening) → ``"brightness"``
        2. Velocity (rotation + obliquity) → ``"velocity"``
        3. Surface features (granulation / spots / faculae / plages):

           * Modify ``"brightness"`` in-place (spot darkening, facular brightening)
           * Update ``"T_eff"`` with temperature delta
           * Set ``"v_conv"`` (radial convective velocity, km/s)
           * Set ``"line_depth_factor"`` for spectral line scaling

        4. Velocity re-applied to include convective component:
           total v_los = v_rotation + v_conv × μ
        5. Spectrum → ``"spectrum"`` (uses updated T_eff, line_depth_factor,
           and total velocity for Doppler shifting)

        Returns self.
        """
        los = self.line_of_sight

        if self._brightness is not None:
            self._brightness.apply(self.grid, name="brightness", line_of_sight=los)

        if self._velocity is not None:
            self._velocity.apply(self.grid, name="velocity", line_of_sight=los)

        # Apply surface features (modifies brightness, T_eff, v_conv, line_depth_factor)
        if self._features:
            from .surface_features import SurfaceFeatureSet
            fset = SurfaceFeatureSet(self._features)
            fset.apply_to_grid(
                self.grid,
                velocity_model  = self._velocity,
                line_of_sight   = los,
                brightness_key  = "brightness",
                T_eff_key       = "T_eff",
            )
            # Re-compute velocity to include convective contribution (v_conv × μ)
            # The Velocity.at() method reads "v_conv" from each element.
            if self._velocity is not None:
                self._velocity.apply(self.grid, name="velocity", line_of_sight=los)

        if self._spectrum is not None:
            v_key = "velocity" if self._velocity is not None else None
            self._spectrum.apply(self.grid, name="spectrum", velocity_key=v_key)

        return self

    # ------------------------------------------------------------------ #
    #  Observables                                                         #
    # ------------------------------------------------------------------ #

    def disk_flux(self, quantity: str = "brightness") -> float:
        """
        Disk-integrated, area-μ-weighted flux for a scalar quantity.

        Parameters
        ----------
        quantity : str
            Stored scalar key (default ``"brightness"``).
        """
        return self.grid.disk_integrate(
            quantity, line_of_sight=self.line_of_sight, weight="area_mu"
        )

    def load_template(
        self,
        path: str,
        wl_col: int = 0,
        flux_col: int = 1,
        wl_unit: str = "angstrom",
        delimiter: str | None = None,
        skip_rows: int = 0,
    ) -> "Star":
        """
        Load a high-resolution stellar template spectrum for CCF computation.

        The template is passed directly to the underlying
        :class:`~starmodel.quantities.Spectrum` model.  When a template is
        loaded, all subsequent CCF calculations use it instead of the
        Gaussian line-mask built from :meth:`add_spectral_line` calls.

        Parameters
        ----------
        path : str
            Path to a two-column ASCII/CSV or FITS file with columns
            (wavelength, flux) in the stellar rest frame.
        wl_col, flux_col : int
            Column indices (default 0, 1).
        wl_unit : {"angstrom", "nm", "micron"}
            Wavelength unit; internally stored in Ångströms.
        delimiter : str, optional
            Column separator (None = whitespace).
        skip_rows : int
            Additional header rows to skip.

        Returns self.
        """
        if self._spectrum is None:
            raise RuntimeError(
                "Call set_spectrum() before load_template()."
            )
        self._spectrum.load_template(
            path, wl_col=wl_col, flux_col=flux_col,
            wl_unit=wl_unit, delimiter=delimiter, skip_rows=skip_rows,
        )
        return self

    def compute_ccf(
        self,
        rv_grid: np.ndarray | None = None,
        template_fwhm: float = 10.0,
        rv_range: float = 50.0,
        n_rv: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the out-of-transit disk-integrated CCF.

        Parameters
        ----------
        rv_grid : array_like, optional
            RV shifts in km/s.  If None, a symmetric grid spanning
            ±*rv_range* km/s with *n_rv* points is used.
        template_fwhm : float
            FWHM of the Gaussian template in km/s (default 10 km/s).
        rv_range : float
            Half-width of the auto RV grid in km/s (default 50).
        n_rv : int
            Number of RV points in the auto grid (default 200).

        Returns
        -------
        rv_grid : np.ndarray  (km/s)
        ccf     : np.ndarray  (normalised CCF values)
        """
        if self._spectrum is None:
            raise RuntimeError("No Spectrum model configured. Call set_spectrum() first.")
        if rv_grid is None:
            rv_grid = np.linspace(-rv_range, rv_range, n_rv)
        else:
            rv_grid = np.asarray(rv_grid, dtype=float)

        flux = self.disk_spectrum()
        ccf  = self._spectrum.compute_ccf(flux, rv_grid, template_fwhm)
        return rv_grid, ccf

    def disk_spectrum(self) -> np.ndarray:
        """
        Disk-integrated spectrum (requires a configured Spectrum model).
        """
        if self._spectrum is None:
            raise RuntimeError("No Spectrum model configured.")
        return self._spectrum.disk_integrated(
            self.grid, name="spectrum", line_of_sight=self.line_of_sight
        )

    def mean_radial_velocity(self) -> float:
        """
        Flux-weighted mean radial velocity (km/s) — i.e., the centroid
        of the cross-correlation profile.
        """
        return self.grid.disk_integrate(
            "velocity", line_of_sight=self.line_of_sight, weight="area_mu"
        )

    # ------------------------------------------------------------------ #
    #  Grid convenience                                                    #
    # ------------------------------------------------------------------ #

    def assign(self, name: str, func: Callable[[SurfaceElement], object]) -> "Star":
        """Assign a custom quantity to every element. Returns self."""
        self.grid.assign(name, func)
        return self

    def assign_array(self, name: str, values) -> "Star":
        """Assign a pre-computed array of values to elements. Returns self."""
        self.grid.assign_array(name, values)
        return self

    def get_array(self, name: str) -> np.ndarray:
        """Retrieve a stored quantity as a NumPy array."""
        return self.grid.get_array(name)

    # ------------------------------------------------------------------ #
    #  Dunder helpers                                                      #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        parts = []
        if self._brightness:
            parts.append(f"brightness={self._brightness.law}")
        if self._velocity:
            lam = math.degrees(self._velocity.obliquity_rad)
            parts.append(f"v_eq={self._velocity.v_eq}km/s, λ={lam:.1f}°")
        if self._spectrum:
            parts.append("spectrum=True")
        config = ", ".join(parts) if parts else "unconfigured"
        return (
            f"Star('{self.name}', R={self.radius}, "
            f"n_elements={len(self.grid)}, {config})"
        )
