"""
example_variability.py
======================
Quick demonstration of surface features + stellar rotation variability.

Run from the project root:
    PYTHONPATH=. python starmodel/example_variability.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from starmodel import Star, GranulationField, StarSpot, Facula, Plage
from starmodel.stellar_variability import RotationSimulator

# ── 1. Build the star ─────────────────────────────────────────────────────────
star = (
    Star(n_theta=30, n_phi=60, name="Example active star")
    .set_brightness(law="quadratic", coefficients={"a": 0.45, "b": 0.30})
    .set_rotation(v_eq=3.0, inclination=75., obliquity=15.)
    .set_temperature_map(lambda e: 5778.)
)

# ── 2. Add surface features ───────────────────────────────────────────────────
star.add_feature(GranulationField(n_cells=500, v_granule=-0.35, v_lane=0.70, seed=1))
star.add_feature(StarSpot(lat_deg=20., lon_deg=0., radius_deg=12., T_contrast=-550.))
star.add_feature(Facula  (lat_deg=20., lon_deg=35., radius_deg=8., alpha=0.10))
star.add_feature(Plage   (lat_deg=20., lon_deg=18., radius_deg=14., intensity_factor=1.06))

star.compute()
print(star)
print("Features:", [repr(f) for f in star.list_features()])

# ── 3. Run the rotation simulator ────────────────────────────────────────────
sim    = RotationSimulator(star, rotation_period_days=20., n_phases_per_cycle=200)
result = sim.run(n_cycles=2., ccf_rv_range=25., ccf_n_rv=150)
print(result.summary())

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig = sim.plot(result, time_unit="days")
out = os.path.join(os.path.dirname(__file__), "example_variability.png")
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved → {out}")

# Disk snapshots at 4 phases
fig2 = sim.plot_disk_snapshots(result, phases_deg=[0., 90., 180., 270.])
out2 = os.path.join(os.path.dirname(__file__), "example_variability_disks.png")
fig2.savefig(out2, dpi=130, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.close(fig2)
print(f"Saved → {out2}")
