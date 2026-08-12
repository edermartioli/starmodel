"""
system.py — Planetary system parameter loader for starmodel.

Reads all stellar and planetary parameters from a structured JSON or CSV
file and constructs a fully configured ``Star`` + ``OrbitalParameters``
pair ready to be passed to ``TransitModel``.

File formats
------------
JSON (recommended)
    Hierarchical format with ``[value, uncertainty]`` pairs for every
    parameter.  See the canonical example ``WASP-108.json`` shipped with
    the package.

CSV / flat
    Two mandatory columns: ``parameter`` and ``value``.  Optional columns
    ``uncertainty``, ``unit``, and ``component`` (``star`` or ``planet``).

Canonical JSON schema
---------------------
{
    "system_name": "WASP-108",
    "components": ["star A", "planet b"],

    "star A": {
        "name"                      : "WASP-108",
        "object_type"               : "star",
        "teff"                      : [5975.8, 80.0],      // K
        "logg"                      : [4.22,   0.05],      // cgs
        "mass"                      : [1.10,   0.05],      // M_sun
        "radius"                    : [1.344,  0.03],      // R_sun
        "metallicity"               : [0.0,    0.05],      // [Fe/H]
        "vsini"                     : [5.75,   0.20],      // km/s
        "stellar_inclination"       : [90.0,   5.0],       // deg  (axis to LOS)
        "differential_rotation"     : [0.0,    0.0],       // alpha param
        "limb_darkening_law"        : "quadratic",
        "limb_darkening_coeffs"     : [[0.45, 0.01], [0.30, 0.01]], // per coeff [val,err]
        "spinorbit_obliquity"       : [6.56,   1.5]        // deg
    },

    "planet b": {
        "name"                      : "WASP-108 b",
        "object_type"               : "planet",
        "transit"                   : true,
        "radius_rstar"              : [0.1115, 0.0010],    // Rp/R*
        "orbital_sma_rstar"         : [6.988,  0.050],     // a/R*
        "orbital_period_days"       : [2.6756, 0.0001],    // days
        "orbital_ecc"               : [0.0,    0.0],
        "orbital_omega"             : [90.0,   0.0],       // deg, arg. of periastron
        "orbital_Omega"             : [0.0,    0.0],       // deg, lon. of ascending node
        "orbital_inc"               : [88.58,  0.20],      // deg
        "impact_parameter"          : [null,   null],      // overrides inc if not null
        "spinorbit_obliquity"       : [6.56,   1.5],       // deg  (mirrored from star)
        "rv_semi_amplitude"         : [0.0891, 0.005],     // km/s
        "transit_time_bjd"          : [2458597.039, 0.001],// BJD
        "time_of_periastron_bjd"    : [2458597.039, 0.001] // BJD
    }
}

Parameter mapping
-----------------
Star parameters → starmodel fields
    teff                   → default T_eff for element spectra (K)
    logg                   → stored as metadata
    mass                   → stored as metadata (M_sun)
    radius                 → Star(radius=...)  [R_sun; kept as unit]
    vsini                  → v_eq = vsini / sin(stellar_inclination)
    stellar_inclination    → Star.set_rotation(inclination=...)
    differential_rotation  → Star.set_rotation(differential_rotation=...)
    limb_darkening_law     → Star.set_brightness(law=...)
    limb_darkening_coeffs  → Star.set_brightness(coefficients=...)
    spinorbit_obliquity    → Star.set_rotation(obliquity=...)

Planet parameters → OrbitalParameters fields
    radius_rstar           → planet_radius
    orbital_sma_rstar      → semi_major_axis
    orbital_period_days    → period
    orbital_ecc            → eccentricity
    orbital_omega          → omega
    orbital_Omega          → Omega
    orbital_inc            → inclination
    impact_parameter       → impact_parameter (overrides inc if provided)
    spinorbit_obliquity    → obliquity
    transit_time_bjd       → t0
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# =========================================================================== #
#  Parameter value + uncertainty container                                     #
# =========================================================================== #

@dataclass
class Param:
    """
    A single measured / adopted parameter with its uncertainty.

    Attributes
    ----------
    value : float or None
        Central value.
    error : float
        1-sigma uncertainty (0 if exact / fixed).
    unit : str
        Physical unit string (informational only).
    source : str
        Citation or origin tag.
    """
    value:  float | None = None
    error:  float        = 0.0
    unit:   str          = ""
    source: str          = ""

    def __repr__(self) -> str:
        if self.value is None:
            return f"Param(None ± {self.error} {self.unit})"
        return f"Param({self.value} ± {self.error} {self.unit})"

    def __float__(self) -> float:
        if self.value is None:
            raise ValueError("Parameter value is None.")
        return float(self.value)

    @property
    def is_set(self) -> bool:
        """True if the value is not None."""
        return self.value is not None

    @classmethod
    def from_pair(cls, pair, unit: str = "", source: str = "") -> "Param":
        """
        Construct from a ``[value, error]`` pair or a scalar.

        Accepts:
        * ``[value, error]``   — standard JSON pair
        * ``[null, null]``     — undefined parameter
        * scalar ``value``     — zero uncertainty assumed
        * ``None``             — undefined
        """
        if pair is None:
            return cls(value=None, error=0., unit=unit, source=source)
        if isinstance(pair, (int, float)):
            return cls(value=float(pair), error=0., unit=unit, source=source)
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            v, e = pair
            v = None if v is None else float(v)
            e = 0.   if e is None else float(e)
            return cls(value=v, error=e, unit=unit, source=source)
        raise ValueError(f"Cannot parse parameter from: {pair!r}")


# =========================================================================== #
#  Stellar parameter set                                                       #
# =========================================================================== #

@dataclass
class StarParams:
    """All stellar parameters loaded from a system file."""

    name:                   str   = "Star"
    teff:                   Param = field(default_factory=lambda: Param(5778., 0., "K"))
    logg:                   Param = field(default_factory=lambda: Param(4.44,  0., "cgs"))
    mass:                   Param = field(default_factory=lambda: Param(1.0,   0., "M_sun"))
    radius:                 Param = field(default_factory=lambda: Param(1.0,   0., "R_sun"))
    metallicity:            Param = field(default_factory=lambda: Param(0.0,   0., "[Fe/H]"))
    vsini:                  Param = field(default_factory=lambda: Param(0.0,   0., "km/s"))
    stellar_inclination:    Param = field(default_factory=lambda: Param(90.,   0., "deg"))
    differential_rotation:  Param = field(default_factory=lambda: Param(0.0,   0., ""))
    spinorbit_obliquity:    Param = field(default_factory=lambda: Param(0.0,   0., "deg"))
    limb_darkening_law:     str   = "quadratic"
    # ld_coeffs: list of Param objects, one per coefficient
    ld_coeffs:              list  = field(default_factory=list)

    @property
    def v_eq(self) -> float:
        """
        Equatorial rotational velocity (km/s).

        Derived from v sin i★:
            v_eq = vsini / sin(stellar_inclination)

        Falls back to vsini when i★ is 0° or not set.
        """
        if self.vsini.value is None:
            return 0.
        si = math.sin(math.radians(float(self.stellar_inclination.value or 90.)))
        if si < 1e-6:
            return float(self.vsini.value)
        return float(self.vsini.value) / si

    @property
    def v_eq_error(self) -> float:
        """Propagated uncertainty on v_eq (km/s)."""
        si = math.sin(math.radians(float(self.stellar_inclination.value or 90.)))
        si = max(si, 1e-6)
        # σ(v_eq) ≈ σ(vsini) / sin(i)  [dominant term; ignoring i uncertainty]
        return self.vsini.error / si

    @property
    def ld_coefficients_dict(self) -> dict:
        """
        Limb-darkening coefficients as a dict suitable for
        ``Star.set_brightness(coefficients=...)``.

        Mapping:
        * 1 coeff  → ``{"u": c0}``          (linear)
        * 2 coeffs → ``{"a": c0, "b": c1}`` (quadratic)
        * 4 coeffs → ``{"a1": c0, ...}``     (Claret 4-term)
        """
        vals = [p.value for p in self.ld_coeffs if p.value is not None]
        n = len(vals)
        if n == 0:
            return {}
        if n == 1:
            return {"u": vals[0]}
        if n == 2:
            return {"a": vals[0], "b": vals[1]}
        if n == 4:
            return {"a1": vals[0], "a2": vals[1], "a3": vals[2], "a4": vals[3]}
        return {f"c{i}": v for i, v in enumerate(vals)}

    def summary(self) -> str:
        lines = [
            f"  Name                : {self.name}",
            f"  T_eff               : {self.teff}",
            f"  log g               : {self.logg}",
            f"  Mass                : {self.mass}",
            f"  Radius              : {self.radius}",
            f"  [Fe/H]              : {self.metallicity}",
            f"  v sin i★            : {self.vsini}",
            f"  Stellar inclination : {self.stellar_inclination}",
            f"  v_eq (derived)      : {self.v_eq:.4f} ± {self.v_eq_error:.4f} km/s",
            f"  Diff. rotation α    : {self.differential_rotation}",
            f"  Obliquity λ         : {self.spinorbit_obliquity}",
            f"  LD law              : {self.limb_darkening_law}",
            f"  LD coeffs           : {self.ld_coefficients_dict}",
        ]
        return "\n".join(lines)


# =========================================================================== #
#  Planetary parameter set                                                     #
# =========================================================================== #

@dataclass
class PlanetParams:
    """All planetary / orbital parameters loaded from a system file."""

    name:                   str   = "Planet"
    radius_rstar:           Param = field(default_factory=lambda: Param(0.1,   0.))
    orbital_sma_rstar:      Param = field(default_factory=lambda: Param(10.,   0.))
    orbital_period_days:    Param = field(default_factory=lambda: Param(3.0,   0.,  "days"))
    orbital_ecc:            Param = field(default_factory=lambda: Param(0.0,   0.))
    orbital_omega:          Param = field(default_factory=lambda: Param(90.,   0.,  "deg"))
    orbital_Omega:          Param = field(default_factory=lambda: Param(0.0,   0.,  "deg"))
    orbital_inc:            Param = field(default_factory=lambda: Param(90.,   0.,  "deg"))
    impact_parameter:       Param = field(default_factory=lambda: Param(None,  0.))
    spinorbit_obliquity:    Param = field(default_factory=lambda: Param(0.0,   0.,  "deg"))
    rv_semi_amplitude:      Param = field(default_factory=lambda: Param(0.0,   0.,  "km/s"))
    transit_time_bjd:       Param = field(default_factory=lambda: Param(0.0,   0.,  "BJD"))
    time_of_periastron_bjd: Param = field(default_factory=lambda: Param(None,  0.,  "BJD"))

    def summary(self) -> str:
        lines = [
            f"  Name                : {self.name}",
            f"  Rp / R★             : {self.radius_rstar}",
            f"  a / R★              : {self.orbital_sma_rstar}",
            f"  Period              : {self.orbital_period_days}",
            f"  Eccentricity        : {self.orbital_ecc}",
            f"  ω (arg. periastron) : {self.orbital_omega}",
            f"  Ω (lon. asc. node)  : {self.orbital_Omega}",
            f"  Orbital inclination : {self.orbital_inc}",
            f"  Impact parameter b  : {self.impact_parameter}",
            f"  Obliquity λ         : {self.spinorbit_obliquity}",
            f"  K (RV semi-amp.)    : {self.rv_semi_amplitude}",
            f"  Transit T0 (BJD)    : {self.transit_time_bjd}",
        ]
        return "\n".join(lines)


# =========================================================================== #
#  PlanetarySystem — main loader and factory                                   #
# =========================================================================== #

class PlanetarySystem:
    """
    Load a planetary system from a JSON or CSV parameter file and build
    a ready-to-use ``Star`` + ``OrbitalParameters`` pair.

    Parameters
    ----------
    path : str
        Path to a ``.json`` or ``.csv`` parameter file.

    Attributes
    ----------
    name : str
        System name.
    star : StarParams
        All stellar parameters with uncertainties.
    planet : PlanetParams
        All planetary / orbital parameters with uncertainties.
    metadata : dict
        Raw loaded data (for inspection / re-export).

    Examples
    --------
    >>> from starmodel import PlanetarySystem
    >>> import numpy as np
    >>>
    >>> sys = PlanetarySystem("WASP-108.json")
    >>> print(sys.summary())
    >>>
    >>> # Build the starmodel Star object
    >>> wl = np.linspace(6300, 6900, 400)
    >>> star = sys.build_star(n_theta=40, n_phi=80, wavelengths=wl)
    >>> star.add_spectral_line(6563., depth=0.65, width=1.5)
    >>> star.compute()
    >>>
    >>> # Build the OrbitalParameters object
    >>> orbit = sys.build_orbit()
    >>>
    >>> from starmodel import TransitModel
    >>> model  = TransitModel(star, orbit)
    >>> result = model.compute(n_times=500, compute_ccf=True)
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(self, path: str) -> None:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"System parameter file not found: {path!r}")
        self.path = path
        self.metadata: dict[str, Any] = {}
        self.star   = StarParams()
        self.planet = PlanetParams()
        self.name   = "Unknown system"

        ext = os.path.splitext(path)[1].lower()
        if ext in (".json",):
            self._load_json(path)
        elif ext in (".csv", ".tsv", ".txt"):
            self._load_csv(path)
        else:
            # Try JSON first, then CSV
            try:
                self._load_json(path)
            except json.JSONDecodeError:
                self._load_csv(path)

    # ------------------------------------------------------------------ #
    #  JSON loader                                                         #
    # ------------------------------------------------------------------ #

    def _load_json(self, path: str) -> None:
        with open(path, "r") as fh:
            raw = json.load(fh)
        self.metadata = raw
        self.name = raw.get("system_name", "Unknown system")

        components = raw.get("components", [])

        # Find star and planet blocks
        star_block   = None
        planet_block = None
        for comp in components:
            block = raw.get(comp, {})
            otype = block.get("object_type", "").lower()
            if otype == "star" and star_block is None:
                star_block = block
            elif otype == "planet" and planet_block is None:
                planet_block = block

        # Fallback: try common keys
        if star_block is None:
            for key in ("star", "star A", "star_A", "stellar"):
                if key in raw:
                    star_block = raw[key]; break
        if planet_block is None:
            for key in ("planet", "planet b", "planet_b", "planetary"):
                if key in raw:
                    planet_block = raw[key]; break

        if star_block is not None:
            self._parse_star_json(star_block)
        if planet_block is not None:
            self._parse_planet_json(planet_block)

    def _parse_star_json(self, blk: dict) -> None:
        s = self.star
        s.name = blk.get("name", "Star")

        def p(key, unit="", default=None):
            v = blk.get(key, default)
            return Param.from_pair(v, unit=unit)

        s.teff                  = p("teff",                  unit="K",      default=[5778., 0.])
        s.logg                  = p("logg",                  unit="cgs",    default=[4.44,  0.])
        s.mass                  = p("mass",                  unit="M_sun",  default=[1.0,   0.])
        s.radius                = p("radius",                unit="R_sun",  default=[1.0,   0.])
        s.metallicity           = p("metallicity",           unit="[Fe/H]", default=[0.0,   0.])
        s.vsini                 = p("vsini",                 unit="km/s",   default=[0.0,   0.])
        s.stellar_inclination   = p("stellar_inclination",   unit="deg",    default=[90.,   0.])
        s.differential_rotation = p("differential_rotation",               default=[0.0,   0.])
        s.spinorbit_obliquity   = p("spinorbit_obliquity",   unit="deg",    default=[0.0,   0.])
        s.limb_darkening_law    = blk.get("limb_darkening_law", "quadratic")

        # Limb-darkening coefficients:
        # Accept [[val, err], [val, err], ...] or [[val, err], [val, err]]
        raw_ld = blk.get("limb_darkening_coeffs", [])
        if raw_ld and isinstance(raw_ld[0], (list, tuple)):
            # List of [val, err] pairs
            s.ld_coeffs = [Param.from_pair(c, unit="") for c in raw_ld]
        elif raw_ld:
            # Flat list of values — wrap with zero error
            s.ld_coeffs = [Param.from_pair([v, 0.], unit="") for v in raw_ld]
        else:
            s.ld_coeffs = []

    def _parse_planet_json(self, blk: dict) -> None:
        pl = self.planet
        pl.name = blk.get("name", "Planet")

        def p(key, unit="", default=None):
            v = blk.get(key, default)
            return Param.from_pair(v, unit=unit)

        pl.radius_rstar           = p("radius_rstar",           unit="Rp/R*",  default=[0.1,  0.])
        pl.orbital_sma_rstar      = p("orbital_sma_rstar",      unit="a/R*",   default=[10.,  0.])
        pl.orbital_period_days    = p("orbital_period_days",    unit="days",   default=[3.0,  0.])
        pl.orbital_ecc            = p("orbital_ecc",                           default=[0.0,  0.])
        pl.orbital_omega          = p("orbital_omega",          unit="deg",    default=[90.,  0.])
        pl.orbital_Omega          = p("orbital_Omega",          unit="deg",    default=[0.0,  0.])
        pl.orbital_inc            = p("orbital_inc",            unit="deg",    default=[90.,  0.])
        pl.impact_parameter       = p("impact_parameter",                      default=[None, 0.])
        pl.spinorbit_obliquity    = p("spinorbit_obliquity",    unit="deg",    default=[0.0,  0.])
        pl.rv_semi_amplitude      = p("rv_semi_amplitude",      unit="km/s",   default=[0.0,  0.])
        pl.transit_time_bjd       = p("transit_time_bjd",       unit="BJD",    default=[0.0,  0.])
        pl.time_of_periastron_bjd = p("time_of_periastron_bjd", unit="BJD",   default=[None, 0.])

        # If spinorbit_obliquity not in planet block, inherit from star
        if not pl.spinorbit_obliquity.is_set:
            pl.spinorbit_obliquity = self.star.spinorbit_obliquity

    # ------------------------------------------------------------------ #
    #  CSV loader                                                          #
    # ------------------------------------------------------------------ #

    def _load_csv(self, path: str) -> None:
        """
        Load a flat CSV/TSV with columns:
            component, parameter, value, uncertainty, unit, source

        ``component`` should be ``star`` or ``planet`` (case-insensitive).
        ``parameter`` must match one of the canonical JSON keys above.
        """
        import csv

        star_raw:   dict[str, Any] = {}
        planet_raw: dict[str, Any] = {}

        with open(path, newline="") as fh:
            dialect = csv.Sniffer().sniff(fh.read(2048))
            fh.seek(0)
            reader = csv.DictReader(fh, dialect=dialect)
            for row in reader:
                comp  = row.get("component", "star").strip().lower()
                param = row.get("parameter", "").strip()
                val   = row.get("value",       "").strip()
                err   = row.get("uncertainty", "0").strip() or "0"
                unit  = row.get("unit",        "").strip()
                src   = row.get("source",      "").strip()

                if not param:
                    continue

                def _parse(s):
                    try: return float(s) if s not in ("", "null", "None") else None
                    except ValueError: return s

                pair = [_parse(val), float(err) if err not in ("", "null") else 0.]

                target = star_raw if "star" in comp else planet_raw
                target[param] = pair
                # Carry string fields
                if param in ("name", "object_type", "limb_darkening_law"):
                    target[param] = _parse(val)

        if "system_name" not in star_raw:
            self.name = os.path.splitext(os.path.basename(path))[0]

        # Handle LD coefficients from CSV rows named ld_coeff_0, ld_coeff_1, ...
        ld_pairs = []
        for i in range(10):
            key = f"ld_coeff_{i}"
            if key in star_raw:
                ld_pairs.append(star_raw.pop(key))
        if ld_pairs:
            star_raw["limb_darkening_coeffs"] = ld_pairs

        self._parse_star_json(star_raw)
        self._parse_planet_json(planet_raw)

    # ------------------------------------------------------------------ #
    #  Factory: build Star                                                 #
    # ------------------------------------------------------------------ #

    def build_star(
        self,
        n_theta:     int   = 40,
        n_phi:       int   = 80,
        grid_type:   str   = "uniform",
        wavelengths: "np.ndarray | None" = None,
        uniform_teff: bool = True,
    ) -> "Star":
        """
        Construct and return a configured :class:`~starmodel.Star` from
        the loaded parameters.

        Parameters
        ----------
        n_theta, n_phi : int
            Grid resolution (colatitude bands × longitude cells).
        grid_type : {"uniform", "equalarea"}
            Surface grid type.
        wavelengths : array_like, optional
            Wavelength grid (Å) for the spectral model.  If None, a default
            grid 5500–7000 Å with 400 points is used.
        uniform_teff : bool
            If True, assign T_eff uniformly to every element (no spatial
            gradient).  Set to False and call
            ``star.set_temperature_map(func)`` manually to override.

        Returns
        -------
        Star
            Configured but not yet ``compute()``-d.  Call
            ``star.compute()`` after adding any spectral lines.
        """
        from .star import Star

        if wavelengths is None:
            wavelengths = np.linspace(5500., 7000., 400)
        else:
            wavelengths = np.asarray(wavelengths, dtype=float)

        sp = self.star
        r  = float(sp.radius.value or 1.0)

        star = Star(
            radius    = r,
            n_theta   = n_theta,
            n_phi     = n_phi,
            grid_type = grid_type,
            name      = sp.name,
        )

        # ── Brightness / limb darkening ──────────────────────────────────
        law    = sp.limb_darkening_law or "quadratic"
        coeffs = sp.ld_coefficients_dict
        if not coeffs:
            # SDSS-band defaults for a solar-like star
            coeffs = {"a": 0.45, "b": 0.30}
            law    = "quadratic"
        star.set_brightness(law=law, coefficients=coeffs)

        # ── Rotation ─────────────────────────────────────────────────────
        star.set_rotation(
            v_eq                = sp.v_eq,
            inclination         = float(sp.stellar_inclination.value or 90.),
            obliquity           = float(sp.spinorbit_obliquity.value or 0.),
            differential_rotation = float(sp.differential_rotation.value or 0.),
        )

        # ── Spectrum ─────────────────────────────────────────────────────
        T_eff = float(sp.teff.value or 5778.)
        star.set_spectrum(wavelengths, T_eff_key="T_eff")

        if uniform_teff:
            star.set_temperature_map(lambda e: T_eff, key="T_eff")

        # Store physical metadata as custom grid quantities
        star._system_params = self   # back-reference for diagnostics

        return star

    # ------------------------------------------------------------------ #
    #  Factory: build OrbitalParameters                                   #
    # ------------------------------------------------------------------ #

    def build_orbit(self) -> "OrbitalParameters":
        """
        Construct and return an :class:`~starmodel.OrbitalParameters`
        from the loaded planetary parameters.

        Returns
        -------
        OrbitalParameters
        """
        from .transit import OrbitalParameters

        pl  = self.planet
        t0  = float(pl.transit_time_bjd.value or 0.)

        # impact_parameter overrides inclination if provided
        b_val = pl.impact_parameter.value

        return OrbitalParameters(
            period          = float(pl.orbital_period_days.value   or 3.),
            t0              = t0,
            semi_major_axis = float(pl.orbital_sma_rstar.value     or 10.),
            inclination     = float(pl.orbital_inc.value           or 90.),
            eccentricity    = float(pl.orbital_ecc.value           or 0.),
            omega           = float(pl.orbital_omega.value         or 90.),
            Omega           = float(pl.orbital_Omega.value         or 0.),
            planet_radius   = float(pl.radius_rstar.value          or 0.1),
            impact_parameter= b_val,
            obliquity       = float(pl.spinorbit_obliquity.value   or 0.),
        )

    # ------------------------------------------------------------------ #
    #  Export                                                              #
    # ------------------------------------------------------------------ #

    def to_json(self, path: str) -> None:
        """
        Write the system parameters back to a canonical JSON file.

        The output preserves the ``[value, error]`` pair convention and
        can be used as a template for new systems or to record fitted values.

        Parameters
        ----------
        path : str
            Destination file path.
        """

        def _pair(param: Param):
            return [param.value, param.error]

        sp  = self.star
        pl  = self.planet

        doc = {
            "system_name": self.name,
            "components" : ["star A", "planet b"],
            "star A": {
                "name"                      : sp.name,
                "object_type"               : "star",
                "teff"                      : _pair(sp.teff),
                "logg"                      : _pair(sp.logg),
                "mass"                      : _pair(sp.mass),
                "radius"                    : _pair(sp.radius),
                "metallicity"               : _pair(sp.metallicity),
                "vsini"                     : _pair(sp.vsini),
                "stellar_inclination"       : _pair(sp.stellar_inclination),
                "differential_rotation"     : _pair(sp.differential_rotation),
                "spinorbit_obliquity"       : _pair(sp.spinorbit_obliquity),
                "limb_darkening_law"        : sp.limb_darkening_law,
                "limb_darkening_coeffs"     : [_pair(c) for c in sp.ld_coeffs],
            },
            "planet b": {
                "name"                      : pl.name,
                "object_type"               : "planet",
                "transit"                   : True,
                "radius_rstar"              : _pair(pl.radius_rstar),
                "orbital_sma_rstar"         : _pair(pl.orbital_sma_rstar),
                "orbital_period_days"       : _pair(pl.orbital_period_days),
                "orbital_ecc"               : _pair(pl.orbital_ecc),
                "orbital_omega"             : _pair(pl.orbital_omega),
                "orbital_Omega"             : _pair(pl.orbital_Omega),
                "orbital_inc"               : _pair(pl.orbital_inc),
                "impact_parameter"          : _pair(pl.impact_parameter),
                "spinorbit_obliquity"       : _pair(pl.spinorbit_obliquity),
                "rv_semi_amplitude"         : _pair(pl.rv_semi_amplitude),
                "transit_time_bjd"          : _pair(pl.transit_time_bjd),
                "time_of_periastron_bjd"    : _pair(pl.time_of_periastron_bjd),
            },
        }

        with open(path, "w") as fh:
            json.dump(doc, fh, indent=4)
        print(f"System parameters written → {path}")

    def to_csv(self, path: str) -> None:
        """
        Write the system parameters to a flat CSV file with columns:
        component, parameter, value, uncertainty, unit.

        This format is suitable for quick inspection in a spreadsheet.
        """
        import csv

        sp = self.star
        pl = self.planet

        rows = [
            ["component", "parameter", "value", "uncertainty", "unit"],
            # ── Star ──────────────────────────────────────────────────────
            ["star", "name",                   sp.name,                      "",                        ""],
            ["star", "teff",                   sp.teff.value,                sp.teff.error,             "K"],
            ["star", "logg",                   sp.logg.value,                sp.logg.error,             "cgs"],
            ["star", "mass",                   sp.mass.value,                sp.mass.error,             "M_sun"],
            ["star", "radius",                 sp.radius.value,              sp.radius.error,           "R_sun"],
            ["star", "metallicity",            sp.metallicity.value,         sp.metallicity.error,      "[Fe/H]"],
            ["star", "vsini",                  sp.vsini.value,               sp.vsini.error,            "km/s"],
            ["star", "stellar_inclination",    sp.stellar_inclination.value, sp.stellar_inclination.error, "deg"],
            ["star", "differential_rotation",  sp.differential_rotation.value, sp.differential_rotation.error, ""],
            ["star", "spinorbit_obliquity",    sp.spinorbit_obliquity.value, sp.spinorbit_obliquity.error, "deg"],
            ["star", "limb_darkening_law",     sp.limb_darkening_law,        "",                        ""],
        ]
        for i, c in enumerate(sp.ld_coeffs):
            rows.append(["star", f"ld_coeff_{i}", c.value, c.error, ""])

        rows += [
            # ── Planet ────────────────────────────────────────────────────
            ["planet", "name",                   pl.name,                            "",                     ""],
            ["planet", "radius_rstar",            pl.radius_rstar.value,              pl.radius_rstar.error,  "Rp/R*"],
            ["planet", "orbital_sma_rstar",       pl.orbital_sma_rstar.value,         pl.orbital_sma_rstar.error, "a/R*"],
            ["planet", "orbital_period_days",     pl.orbital_period_days.value,       pl.orbital_period_days.error, "days"],
            ["planet", "orbital_ecc",             pl.orbital_ecc.value,               pl.orbital_ecc.error,   ""],
            ["planet", "orbital_omega",           pl.orbital_omega.value,             pl.orbital_omega.error, "deg"],
            ["planet", "orbital_Omega",           pl.orbital_Omega.value,             pl.orbital_Omega.error, "deg"],
            ["planet", "orbital_inc",             pl.orbital_inc.value,               pl.orbital_inc.error,   "deg"],
            ["planet", "impact_parameter",        pl.impact_parameter.value,          pl.impact_parameter.error, ""],
            ["planet", "spinorbit_obliquity",     pl.spinorbit_obliquity.value,       pl.spinorbit_obliquity.error, "deg"],
            ["planet", "rv_semi_amplitude",       pl.rv_semi_amplitude.value,         pl.rv_semi_amplitude.error, "km/s"],
            ["planet", "transit_time_bjd",        pl.transit_time_bjd.value,          pl.transit_time_bjd.error, "BJD"],
            ["planet", "time_of_periastron_bjd",  pl.time_of_periastron_bjd.value,    pl.time_of_periastron_bjd.error, "BJD"],
        ]

        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
        print(f"System parameters written → {path}")

    # ------------------------------------------------------------------ #
    #  Display                                                             #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        lines = [
            f"PlanetarySystem: {self.name}",
            f"  File: {self.path}",
            "",
            "── Star ──────────────────────────────────────",
            self.star.summary(),
            "",
            "── Planet ────────────────────────────────────",
            self.planet.summary(),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"PlanetarySystem('{self.name}', "
                f"star='{self.star.name}', planet='{self.planet.name}')")
