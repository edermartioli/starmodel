"""
SurfaceGrid — Discretizes a sphere into a 2-D array of SurfaceElements.

Two grid types are supported:

  * ``"uniform"``   — equal spacing in (θ, φ).  Simple but cells pile up
                      near the poles.  Good for most stellar work.

  * ``"equalarea"`` — each latitude ring is subdivided so that every cell
                      has approximately the same solid angle.  Eliminates
                      the polar crowding but the grid is irregular.
"""

from __future__ import annotations
import math
import itertools
from typing import Callable, Generator, Iterable

import numpy as np

from .element import SurfaceElement


class SurfaceGrid:
    """
    A grid of :class:`SurfaceElement` objects covering the full sphere.

    Parameters
    ----------
    n_theta : int
        Number of latitude (colatitude) bands.
    n_phi : int
        Number of longitude divisions (for the ``"uniform"`` grid this is
        the same everywhere; for ``"equalarea"`` it varies by latitude ring).
    radius : float
        Stellar radius.
    grid_type : {"uniform", "equalarea"}
        Discretization strategy.
    """

    def __init__(
        self,
        n_theta: int = 36,
        n_phi: int = 72,
        radius: float = 1.0,
        grid_type: str = "uniform",
    ) -> None:
        if n_theta < 1 or n_phi < 1:
            raise ValueError("n_theta and n_phi must be >= 1")
        if grid_type not in ("uniform", "equalarea"):
            raise ValueError('grid_type must be "uniform" or "equalarea"')

        self.n_theta = n_theta
        self.n_phi = n_phi
        self.radius = radius
        self.grid_type = grid_type

        self._elements: list[SurfaceElement] = []
        self._build()

    # ------------------------------------------------------------------ #
    #  Grid construction                                                   #
    # ------------------------------------------------------------------ #

    def _build(self) -> None:
        self._elements.clear()
        d_theta = math.pi / self.n_theta

        for i in range(self.n_theta):
            theta_c = (i + 0.5) * d_theta           # cell-centre colatitude

            if self.grid_type == "uniform":
                n_phi_ring = self.n_phi
            else:  # equalarea: scale n_phi by sin(θ)
                n_phi_ring = max(1, round(self.n_phi * math.sin(theta_c)))

            d_phi = 2.0 * math.pi / n_phi_ring

            for j in range(n_phi_ring):
                phi_c = (j + 0.5) * d_phi
                elem = SurfaceElement(
                    theta=theta_c,
                    phi=phi_c,
                    d_theta=d_theta,
                    d_phi=d_phi,
                    radius=self.radius,
                )
                self._elements.append(elem)

    # ------------------------------------------------------------------ #
    #  Element access                                                      #
    # ------------------------------------------------------------------ #

    @property
    def elements(self) -> list[SurfaceElement]:
        """Flat list of all surface elements."""
        return self._elements

    def __len__(self) -> int:
        return len(self._elements)

    def __iter__(self) -> Generator[SurfaceElement, None, None]:
        yield from self._elements

    def __getitem__(self, index: int) -> SurfaceElement:
        return self._elements[index]

    def visible_elements(
        self, line_of_sight: tuple[float, float, float] = (0.0, 0.0, 1.0)
    ) -> list[SurfaceElement]:
        """Return only elements facing the observer."""
        return [e for e in self._elements if e.is_visible(line_of_sight)]

    # ------------------------------------------------------------------ #
    #  Bulk quantity assignment                                            #
    # ------------------------------------------------------------------ #

    def assign(self, name: str, func: Callable[[SurfaceElement], object]) -> None:
        """
        Compute and store a quantity for every element.

        Parameters
        ----------
        name : str
            Key under which the value will be stored on each element.
        func : callable
            Function ``f(element) -> value`` evaluated per element.

        Example
        -------
        >>> grid.assign("brightness", lambda e: 1.0 - 0.3 * np.cos(e.theta))
        """
        for elem in self._elements:
            elem.set(name, func(elem))

    def assign_array(self, name: str, values: Iterable) -> None:
        """
        Store a pre-computed array of values (one per element, same order).

        Parameters
        ----------
        name : str
            Quantity key.
        values : iterable
            Values in the same order as :attr:`elements`.
        """
        vals = list(values)
        if len(vals) != len(self._elements):
            raise ValueError(
                f"Expected {len(self._elements)} values, got {len(vals)}"
            )
        for elem, v in zip(self._elements, vals):
            elem.set(name, v)

    # ------------------------------------------------------------------ #
    #  Bulk quantity retrieval                                             #
    # ------------------------------------------------------------------ #

    def get_array(self, name: str) -> np.ndarray:
        """
        Return all stored values for *name* as a NumPy array.

        Parameters
        ----------
        name : str
            Quantity key.
        """
        return np.array([e.get(name) for e in self._elements])

    def get_property_array(self, prop: str) -> np.ndarray:
        """
        Return a geometric property (``"theta"``, ``"phi"``, ``"mu"``,
        ``"area"``, ``"solid_angle"``) for every element.
        """
        mapping = {
            "theta": lambda e: e.theta,
            "phi": lambda e: e.phi,
            "area": lambda e: e.area,
            "solid_angle": lambda e: e.solid_angle,
            "mu": lambda e: e.mu(),
            "x": lambda e: e.cartesian[0],
            "y": lambda e: e.cartesian[1],
            "z": lambda e: e.cartesian[2],
        }
        if prop not in mapping:
            raise ValueError(
                f"Unknown property '{prop}'. Choose from: {list(mapping)}"
            )
        return np.array([mapping[prop](e) for e in self._elements])

    # ------------------------------------------------------------------ #
    #  Disk-integrated quantities                                          #
    # ------------------------------------------------------------------ #

    def disk_integrate(
        self,
        name: str,
        line_of_sight: tuple[float, float, float] = (0.0, 0.0, 1.0),
        weight: str = "area_mu",
    ) -> float:
        """
        Compute a disk-integrated (flux-weighted) scalar quantity.

        Parameters
        ----------
        name : str
            Stored scalar quantity to integrate.
        line_of_sight : (x, y, z)
            Observer direction.
        weight : {"area_mu", "area", "none"}
            Weighting scheme:
            * ``"area_mu"`` — weight by projected area (A · μ), the physical choice.
            * ``"area"``    — weight by element area only.
            * ``"none"``    — simple mean over visible elements.

        Returns
        -------
        float
            Integrated value.
        """
        vis = self.visible_elements(line_of_sight)
        if not vis:
            return 0.0

        total_weight = 0.0
        total_value = 0.0

        for e in vis:
            v = e.get(name)
            mu = e.mu(line_of_sight)
            a = e.area

            if weight == "area_mu":
                w = a * mu
            elif weight == "area":
                w = a
            else:
                w = 1.0

            total_value += v * w
            total_weight += w

        return total_value / total_weight if total_weight else 0.0

    # ------------------------------------------------------------------ #
    #  Statistics                                                          #
    # ------------------------------------------------------------------ #

    def stats(self, name: str) -> dict[str, float]:
        """Return basic statistics for a stored scalar quantity."""
        arr = self.get_array(name).astype(float)
        return {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "total": float(arr.sum()),
        }

    # ------------------------------------------------------------------ #
    #  Geometry summary                                                    #
    # ------------------------------------------------------------------ #

    @property
    def total_area(self) -> float:
        """Total surface area of all elements (should ≈ 4π r²)."""
        return sum(e.area for e in self._elements)

    def __repr__(self) -> str:
        return (
            f"SurfaceGrid(n_theta={self.n_theta}, n_phi={self.n_phi}, "
            f"type='{self.grid_type}', n_elements={len(self)}, "
            f"radius={self.radius})"
        )
