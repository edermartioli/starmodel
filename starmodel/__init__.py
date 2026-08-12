"""
starmodel — A Python framework for modeling physical quantities
on the surface of a spherical star using a discretized grid of surface elements.
"""

from .grid import SurfaceGrid
from .element import SurfaceElement
from .quantities import Brightness, Spectrum, Velocity
from .star import Star
from .transit import TransitModel, OrbitalParameters
from .transit_viewer import plot_transit_epoch, animate_transit
from .transit_overview import plot_transit_overview
from .system import PlanetarySystem, StarParams, PlanetParams, Param
from .surface_features import (
    SurfaceFeatureSet, GranulationField, StarSpot, Facula, Plage
)
from .stellar_variability import RotationSimulator, RotationResult

__all__ = [
    "Star", "SurfaceGrid", "SurfaceElement",
    "Brightness", "Spectrum", "Velocity",
    "TransitModel", "OrbitalParameters",
    "plot_transit_epoch", "animate_transit",
    "plot_transit_overview",
    "PlanetarySystem", "StarParams", "PlanetParams", "Param",
    "SurfaceFeatureSet", "GranulationField", "StarSpot", "Facula", "Plage",
    "RotationSimulator", "RotationResult",
]
__version__ = "0.1.0"

# ── Package data helpers ──────────────────────────────────────────────────────
import os as _os

def get_data_path(filename: str) -> str:
    """
    Return the absolute path to a bundled data file.

    Parameters
    ----------
    filename : str
        File name inside the ``starmodel/data/`` directory.
        Example: ``"WASP-108.json"``, ``"WASP-108.csv"``.

    Returns
    -------
    str  — absolute path that can be passed directly to :class:`PlanetarySystem`.

    Example
    -------
    >>> from starmodel import PlanetarySystem, get_data_path
    >>> sys = PlanetarySystem(get_data_path("WASP-108.json"))
    """
    return _os.path.join(_os.path.dirname(__file__), "data", filename)


def list_data_files() -> list:
    """Return the names of all bundled data files."""
    data_dir = _os.path.join(_os.path.dirname(__file__), "data")
    return sorted(f for f in _os.listdir(data_dir)
                  if not f.startswith("_"))
