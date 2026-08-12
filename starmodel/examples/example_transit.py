"""
example_transit.py — Demonstration of the planetary transit module.

Run:
    PYTHONPATH=/path/to/parent python example_transit.py

Requires:  pip install numpy matplotlib
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np

from starmodel import Star, TransitModel, OrbitalParameters
from starmodel.visualization import plot_transit, plot_transit_disk_snapshot


# =========================================================================== #
#  1.  Build the star                                                          #
# =========================================================================== #

star = (
    Star(radius=1.0, n_theta=40, n_phi=80, name="Host Star")
    .set_brightness(law="quadratic", coefficients={"a": 0.45, "b": 0.30})
    .set_rotation(v_eq=5.0, inclination=90.0)   # 5 km/s equatorial rotation
    .compute()
)

print(star)
print(f"Grid elements : {len(star.grid)}")
print(f"OOT disk flux : {star.disk_flux():.6f}")


# =========================================================================== #
#  2.  Define the orbit                                                        #
# =========================================================================== #

orbit = OrbitalParameters(
    period          = 3.5,        # days
    t0              = 0.0,        # mid-transit reference time (days)
    semi_major_axis = 12.0,       # a / R★
    inclination     = 89.0,       # degrees (nearly edge-on)
    eccentricity    = 0.0,        # circular
    omega           = 90.0,       # argument of periastron (irrelevant for e=0)
    Omega           = 0.0,        # transit chord is horizontal
    planet_radius   = 0.12,       # Rp / R★  → ~Jupiter around Sun-like star
)

print("\nOrbital parameters:")
print(orbit.summary())


# =========================================================================== #
#  3.  Compute the transit                                                     #
# =========================================================================== #

model = TransitModel(star, orbit)
print(f"\n{model}")

result = model.compute(n_times=600)

print("\nTransit result:")
print(result.summary())


# =========================================================================== #
#  4.  Eccentric orbit variant                                                 #
# =========================================================================== #

orbit_ecc = OrbitalParameters(
    period          = 3.5,
    t0              = 0.0,
    semi_major_axis = 12.0,
    inclination     = 89.5,
    eccentricity    = 0.35,
    omega           = 60.0,
    planet_radius   = 0.12,
)

model_ecc = TransitModel(star, orbit_ecc)
result_ecc = model_ecc.compute(n_times=600)

print("\nEccentric orbit variant:")
print(result_ecc.summary())


# =========================================================================== #
#  5.  Impact-parameter scan  (b = 0, 0.5, 0.8)                               #
# =========================================================================== #

print("\nImpact-parameter scan:")
print(f"{'b':>6}  {'depth%':>8}  {'T14 h':>8}  {'RM km/s':>10}")
for b_val in [0.0, 0.3, 0.6, 0.8]:
    orb_b = OrbitalParameters(
        period=3.5, t0=0.0, semi_major_axis=12.0,
        impact_parameter=b_val, planet_radius=0.12,
    )
    res_b = TransitModel(star, orb_b).compute(n_times=400)
    depth = (1.0 - float(np.nanmin(res_b.flux))) * 100
    rm_amp = float(np.nanmax(np.abs(res_b.delta_rv)))
    ct = res_b.contact_times
    t14 = (ct.get("T4", float("nan")) - ct.get("T1", float("nan"))) * 24
    print(f"{b_val:>6.2f}  {depth:>8.4f}  {t14:>8.3f}  {rm_amp:>10.5f}")


# =========================================================================== #
#  6.  Plots                                                                   #
# =========================================================================== #

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.dirname(__file__)

    # ── Main transit diagnostic (circular orbit) ──────────────────────────
    fig = plot_transit(result)
    fig.suptitle("Transit: circular orbit  (Rp/R★=0.12, b=0.35, v_eq=5 km/s)",
                 fontsize=11, fontweight="bold", y=1.01)
    path1 = os.path.join(out_dir, "transit_circular.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {path1}")

    # ── Eccentric orbit diagnostic ────────────────────────────────────────
    fig2 = plot_transit(result_ecc)
    fig2.suptitle("Transit: eccentric orbit  (e=0.35, ω=60°)",
                  fontsize=11, fontweight="bold", y=1.01)
    path2 = os.path.join(out_dir, "transit_eccentric.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved → {path2}")

    # ── Disk snapshots at T1, mid-transit, T4 ────────────────────────────
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle("Transit disk snapshots", fontsize=12, fontweight="bold")
    snap_times = [
        result.contact_times.get("T1", orbit.t0 - 0.05),
        orbit.t0,
        result.contact_times.get("T4", orbit.t0 + 0.05),
    ]
    labels = ["T1 (ingress)", "Mid-transit", "T4 (egress)"]
    for ax, snap_t, lbl in zip(axes3, snap_times, labels):
        if math.isnan(snap_t):
            snap_t = orbit.t0
        plot_transit_disk_snapshot(star, model, snap_t, ax=ax)
        ax.set_title(lbl, fontsize=10)
    path3 = os.path.join(out_dir, "transit_snapshots.png")
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved → {path3}")

    # ── Impact-parameter comparison: light curves ─────────────────────────
    fig4, ax4 = plt.subplots(figsize=(9, 5))
    for b_val in [0.0, 0.3, 0.6, 0.8]:
        orb_b = OrbitalParameters(
            period=3.5, t0=0.0, semi_major_axis=12.0,
            impact_parameter=b_val, planet_radius=0.12,
        )
        res_b = TransitModel(star, orb_b).compute(n_times=400)
        dt_h = (res_b.times - orbit.t0) * 24
        ax4.plot(dt_h, res_b.flux, label=f"b = {b_val:.1f}", lw=1.5)
    ax4.set_xlabel("Time from mid-transit (h)")
    ax4.set_ylabel("Normalised flux")
    ax4.set_title("Light curves for different impact parameters")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    path4 = os.path.join(out_dir, "transit_impact_parameter.png")
    fig4.savefig(path4, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"Saved → {path4}")

except ImportError:
    print("\n(matplotlib not available — skipping plots)")
