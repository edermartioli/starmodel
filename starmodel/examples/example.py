"""
example.py — End-to-end demonstration of the starmodel framework.

Run this file directly:
    python example.py

Requires:  pip install numpy matplotlib
"""

import math
import numpy as np

# ── make sure the local package is importable when run from this directory
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from starmodel import Star
from starmodel.quantities import Brightness, Spectrum, Velocity
from starmodel.visualization import plot_surface_map, plot_disk, plot_spectrum


# =========================================================================== #
#  1.  Build the stellar model                                                 #
# =========================================================================== #

star = (
    Star(radius=1.0, n_theta=36, n_phi=72, name="Demo Star")
    # Quadratic limb darkening
    .set_brightness(law="quadratic", coefficients={"a": 0.45, "b": 0.30})
    # Moderate rotation, tilted 60° toward the observer
    .set_rotation(v_eq=80.0, inclination=60.0, differential_rotation=0.20)
    # Observer along +z axis (default)
    .set_observer((0.0, 0.0, 1.0))
)

# ── Temperature map: hot spot near the equator at φ ≈ 0
def temperature_map(element):
    """Gaussian hot spot superimposed on a uniform background."""
    T_bg = 5800.0        # background (solar-like)
    T_spot = 8000.0      # hot region
    dtheta = element.theta - math.pi / 2    # deviation from equator
    dphi = element.phi - 0.0               # deviation from φ=0 meridian
    # Wrap φ
    dphi = (dphi + math.pi) % (2 * math.pi) - math.pi
    sigma = 0.4          # angular width of spot (radians)
    weight = math.exp(-0.5 * (dtheta**2 + dphi**2) / sigma**2)
    return T_bg + (T_spot - T_bg) * weight

star.set_temperature_map(temperature_map)

# ── Spectrum:  Hα at 6563 Å, Na D at 5893 Å
wavelengths = np.linspace(5800, 6700, 500)
(
    star
    .set_spectrum(wavelengths, T_eff_key="T_eff")
    .add_spectral_line(center=6563.0, depth=0.7, width=1.5, kind="absorption")
    .add_spectral_line(center=5893.0, depth=0.4, width=1.2, kind="absorption")
)

# ── Radial pulsation: higher amplitude near the equatorial belt
def pulsation(element):
    lat = abs(math.pi / 2 - element.theta)  # |latitude|
    return 5.0 * math.exp(-lat**2 / 0.3)     # km/s, gaussian belt

star.set_pulsation(pulsation)

# ── Compute everything
star.compute()

print(star)
print(f"\nGrid summary:")
print(f"  Total elements : {len(star.grid)}")
print(f"  Total area     : {star.grid.total_area:.4f}  (expected 4π={4*math.pi:.4f})")

print(f"\nDisk-integrated flux  : {star.disk_flux():.4f}")
print(f"Mean radial velocity  : {star.mean_radial_velocity():.3f} km/s")
print(f"\nBrightness statistics :", star.grid.stats("brightness"))
print(f"Velocity  statistics  :", star.grid.stats("velocity"))


# =========================================================================== #
#  2.  Attach a custom spot (starspot / magnetic region)                       #
# =========================================================================== #

def starspot_brightness(element):
    """Darken a circular region centred at θ=110°, φ=45°."""
    theta0, phi0 = math.radians(110), math.radians(45)
    dtheta = element.theta - theta0
    dphi   = (element.phi - phi0 + math.pi) % (2*math.pi) - math.pi
    if math.sqrt(dtheta**2 + dphi**2) < 0.3:
        return element.get("brightness") * 0.4    # 60 % darker
    return element.get("brightness")

star.assign("brightness_spot", starspot_brightness)
print(f"\nSpot brightness stats : {star.grid.stats('brightness_spot')}")


# =========================================================================== #
#  3.  Visualize                                                               #
# =========================================================================== #

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for saving
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("starmodel — Demo Star", fontsize=14, fontweight="bold")

    # Surface map of brightness
    plot_surface_map(
        star.grid, "brightness",
        projection="rect",
        cmap="hot",
        title="Brightness (quadratic limb darkening)",
        colorbar_label="I / I₀",
        ax=axes[0],
    )

    # Disk view of velocity field
    plot_disk(
        star.grid, "velocity",
        line_of_sight=star.line_of_sight,
        cmap="RdBu_r",
        title="Radial velocity field (km/s)",
        colorbar_label="v_los (km/s)",
        ax=axes[1],
    )

    # Disk-integrated spectrum
    flux = star.disk_spectrum()
    # Normalise by continuum (smooth with large window)
    from numpy import convolve, ones
    window = 50
    continuum = convolve(flux, ones(window)/window, mode="same")
    norm_flux = flux / continuum
    plot_spectrum(
        wavelengths, norm_flux,
        title="Disk-integrated spectrum (Hα + NaD)",
        ylabel="Normalised flux",
        ax=axes[2],
        color="steelblue",
        lw=1.2,
    )
    axes[2].axvline(6563, color="red",   lw=0.8, ls="--", alpha=0.6, label="Hα")
    axes[2].axvline(5893, color="orange", lw=0.8, ls="--", alpha=0.6, label="Na D")
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "demo_output.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    plt.close()

except ImportError:
    print("\n(matplotlib not available — skipping plots)")


# =========================================================================== #
#  4.  Low-level access example                                                #
# =========================================================================== #

print("\n── Low-level element inspection ──")
elem = star.grid[0]
print(f"Element 0 : {elem}")
print(f"  normal  : {elem.normal}")
print(f"  μ (los) : {elem.mu(star.line_of_sight):.4f}")
print(f"  quantities stored: {list(elem.quantities.keys())}")
