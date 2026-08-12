"""
SurfaceElement — Represents a single discrete patch on the stellar surface.

Each element is defined by its angular position (colatitude θ, longitude φ),
its solid angle (area weight), and the physical quantities assigned to it.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SurfaceElement:
    """
    A single cell on the stellar surface grid.

    Parameters
    ----------
    theta : float
        Colatitude in radians [0, π].  θ=0 is the north pole.
    phi : float
        Longitude in radians [0, 2π).
    d_theta : float
        Angular width of the cell in the θ direction (radians).
    d_phi : float
        Angular width of the cell in the φ direction (radians).
    radius : float
        Stellar radius (arbitrary units, default 1.0).
    """

    theta: float          # colatitude  [0, π]
    phi: float            # longitude   [0, 2π)
    d_theta: float        # cell height (rad)
    d_phi: float          # cell width  (rad)
    radius: float = 1.0

    # Physical quantities stored as a plain dict so users can attach anything.
    _quantities: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------ #
    #  Geometry                                                            #
    # ------------------------------------------------------------------ #

    @property
    def solid_angle(self) -> float:
        """Solid angle subtended by this element (steradians)."""
        return math.sin(self.theta) * self.d_theta * self.d_phi

    @property
    def area(self) -> float:
        """Physical surface area of the element (radius² × solid_angle)."""
        return self.radius ** 2 * self.solid_angle

    @property
    def normal(self) -> tuple[float, float, float]:
        """Outward unit normal vector in Cartesian coordinates (x, y, z)."""
        sin_t = math.sin(self.theta)
        return (
            sin_t * math.cos(self.phi),
            sin_t * math.sin(self.phi),
            math.cos(self.theta),
        )

    @property
    def cartesian(self) -> tuple[float, float, float]:
        """Centre of the element in Cartesian coordinates."""
        nx, ny, nz = self.normal
        return self.radius * nx, self.radius * ny, self.radius * nz

    def mu(self, line_of_sight: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> float:
        """
        Cosine of the angle between the surface normal and a given
        line-of-sight direction.  μ = cos(θ_los).

        Elements with μ ≤ 0 are on the hidden hemisphere.

        Parameters
        ----------
        line_of_sight : (x, y, z) unit vector pointing toward the observer.
        """
        nx, ny, nz = self.normal
        lx, ly, lz = line_of_sight
        return nx * lx + ny * ly + nz * lz

    def is_visible(self, line_of_sight: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> bool:
        """Return True if this element faces the observer (μ > 0)."""
        return self.mu(line_of_sight) > 0.0

    # ------------------------------------------------------------------ #
    #  Quantity accessors                                                  #
    # ------------------------------------------------------------------ #

    def set(self, name: str, value: Any) -> None:
        """Attach a named physical quantity to this element."""
        self._quantities[name] = value

    def get(self, name: str) -> Any:
        """Retrieve a stored quantity by name."""
        return self._quantities[name]

    def has(self, name: str) -> bool:
        """Check whether a quantity has been set."""
        return name in self._quantities

    @property
    def quantities(self) -> dict[str, Any]:
        """Read-only view of all stored quantities."""
        return dict(self._quantities)

    # ------------------------------------------------------------------ #
    #  Dunder helpers                                                      #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        deg = math.degrees
        return (
            f"SurfaceElement(θ={deg(self.theta):.1f}°, "
            f"φ={deg(self.phi):.1f}°, "
            f"area={self.area:.4g}, "
            f"quantities={list(self._quantities.keys())})"
        )
