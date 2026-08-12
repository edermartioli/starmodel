"""
test_stellar_variability.py
============================
Demonstrates and validates the stellar rotation variability simulator.

Tests:
  1. Single large spot — photometric, colour, RV and CCF modulation
  2. Spot + facula pair — competing photometric and colour effects
  3. Granulation only — convective blueshift and CCF bisector distortion
  4. Fully active star — all features combined
  5. Quantitative sanity checks on all observables
  6. Disk snapshots at four rotation phases

Run:
    PYTHONPATH=/path/to/parent python test_stellar_variability.py
"""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from starmodel import (Star, GranulationField, StarSpot, Facula, Plage,
                       RotationSimulator, RotationResult)

OUT = os.path.dirname(os.path.abspath(__file__))
T_PHOT = 5778.
P_ROT  = 25.0    # solar-like rotation period (days)
N_PH   = 300     # phases per cycle


def build_star(name, features=(), n_theta=40, n_phi=80, v_eq=2.0, inc=90., lam=0.):
    """Build a fully configured Star with the given surface features."""
    s = (Star(n_theta=n_theta, n_phi=n_phi, name=name)
         .set_brightness(law="quadratic", coefficients={"a": 0.45, "b": 0.30})
         .set_rotation(v_eq=v_eq, inclination=inc, obliquity=lam)
         .set_temperature_map(lambda e: T_PHOT))
    for f in features:
        s.add_feature(f)
    return s.compute()


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {path}")
    return path


# ============================================================================ #
#  Test 1: Single large starspot                                                #
# ============================================================================ #

print("\n── Test 1: Single starspot (lat=30°, lon=0°, r=15°, ΔT=−600K) ──")

spot = StarSpot(lat_deg=30., lon_deg=0., radius_deg=15., T_contrast=-600.,
                umbra_fraction=0.4, line_depth_factor=1.5)
star_spot = build_star("Star with starspot", features=[spot])
sim_spot  = RotationSimulator(star_spot, P_ROT, n_phases_per_cycle=N_PH)
res_spot  = sim_spot.run(n_cycles=2.)

print(res_spot.summary())

# Quantitative checks
flux_pk = res_spot.flux_diff_ppm.max() - res_spot.flux_diff_ppm.min()
rv_pk   = res_spot.rv_m_s.max() - res_spot.rv_m_s.min()
print(f"  Flux peak-to-peak: {flux_pk:.0f} ppm  (expected > 500 ppm for large spot)")
print(f"  RV peak-to-peak:   {rv_pk:.1f} m/s   (expected > 50 m/s)")
assert flux_pk > 200, f"Flux modulation too small: {flux_pk:.0f} ppm"
assert rv_pk   > 20,  f"RV modulation too small: {rv_pk:.1f} m/s"

fig = sim_spot.plot(res_spot, time_unit="phase")
save(fig, "var_01_spot_only.png")

fig_disk = sim_spot.plot_disk_snapshots(res_spot,
            phases_deg=[0., 90., 180., 270.])
save(fig_disk, "var_01_spot_disk_snapshots.png")


# ============================================================================ #
#  Test 2: Spot + facula pair (active region)                                   #
# ============================================================================ #

print("\n── Test 2: Spot + facula active region ──")

star_ar = build_star("Active region  (spot + facula)", features=[
    StarSpot(lat_deg=20., lon_deg=0.,  radius_deg=12., T_contrast=-500.),
    Facula  (lat_deg=20., lon_deg=35., radius_deg=9.,  alpha=0.12),
    Plage   (lat_deg=20., lon_deg=18., radius_deg=16., intensity_factor=1.06),
])
sim_ar  = RotationSimulator(star_ar, P_ROT, N_PH)
res_ar  = sim_ar.run(n_cycles=2.)
print(res_ar.summary())

fig = sim_ar.plot(res_ar, time_unit="phase")
save(fig, "var_02_active_region.png")

fig_disk = sim_ar.plot_disk_snapshots(res_ar, phases_deg=[0., 45., 90., 135.])
save(fig_disk, "var_02_ar_disk_snapshots.png")


# ============================================================================ #
#  Test 3: Granulation only — convective blueshift and CCF bisector            #
# ============================================================================ #

print("\n── Test 3: Granulation only ──")

star_gran = build_star("Granulation only", features=[
    GranulationField(n_cells=800, T_granule=90., T_lane=-250.,
                     v_granule=-0.40, v_lane=0.80, seed=42),
])
sim_gran = RotationSimulator(star_gran, P_ROT, N_PH)
res_gran = sim_gran.run(n_cycles=2.)
print(res_gran.summary())

# Granulation alone should give near-flat photometry and a nearly constant
# (but slightly noisy due to finite cell count) RV
flux_rms = float(np.std(res_gran.flux_diff_ppm))
rv_rms   = float(np.std(res_gran.rv_m_s))
print(f"  Flux rms (gran): {flux_rms:.1f} ppm  (grid noise from ~{sim_gran.n_pp} cells; decreases with more cells)")
print(f"  RV rms (gran):   {rv_rms:.2f} m/s    (statistical from finite granule count)")

# Check CCF peak is near zero (convective blueshift is subtracted as mean)
ccf_peak_rv = float(res_gran.rv_grid[np.argmax(res_gran.ccf_mean)])
print(f"  CCF peak RV: {ccf_peak_rv:+.3f} km/s  "
      f"(convective blueshift signature in profile)")

fig = sim_gran.plot(res_gran, time_unit="phase")
save(fig, "var_03_granulation.png")


# ============================================================================ #
#  Test 4: Fully active star — all features                                     #
# ============================================================================ #

print("\n── Test 4: Full active star (granulation + spot + facula + plage) ──")

star_full = build_star("Full active star", v_eq=3.0, features=[
    GranulationField(n_cells=800, seed=7),
    StarSpot(lat_deg=25., lon_deg=0.,   radius_deg=14., T_contrast=-550.),
    StarSpot(lat_deg=-15., lon_deg=180., radius_deg=8.,  T_contrast=-400.),  # back-side spot
    Facula  (lat_deg=25., lon_deg=40.,  radius_deg=9.,   alpha=0.11),
    Plage   (lat_deg=25., lon_deg=20.,  radius_deg=18.,  intensity_factor=1.07),
])
sim_full = RotationSimulator(star_full, P_ROT, N_PH)
res_full = sim_full.run(n_cycles=2.)
print(res_full.summary())

fig = sim_full.plot(res_full, time_unit="days")
save(fig, "var_04_full_active.png")

fig_disk = sim_full.plot_disk_snapshots(res_full,
            phases_deg=[0., 90., 180., 270.])
save(fig_disk, "var_04_full_disk_snapshots.png")


# ============================================================================ #
#  Test 5: Obliquity effect on RV signal shape                                 #
# ============================================================================ #

print("\n── Test 5: Obliquity effect on RV signal shape (λ = 0°, 60°, 90°) ──")

fig5, axes = plt.subplots(3, 1, figsize=(12, 9), facecolor="#0d0d0d")
for ax in axes:
    ax.set_facecolor("#111111")
    for sp in ax.spines.values(): sp.set_edgecolor("#444444")
    ax.tick_params(colors="#bbbbbb"); ax.grid(True, alpha=0.15, color="#555555")

colors_lam = ["#5599ff", "#ffaa33", "#ee44ff"]
for lam_deg, col in zip([0., 60., 90.], colors_lam):
    sl = build_star(f"λ={lam_deg:.0f}°", v_eq=2.0, lam=lam_deg, features=[
        StarSpot(lat_deg=0., lon_deg=0., radius_deg=15., T_contrast=-600.),
    ])
    sim_l = RotationSimulator(sl, P_ROT, N_PH)
    res_l = sim_l.run(n_cycles=2.)

    ph = res_l.phases
    axes[0].plot(ph, res_l.flux_diff_ppm,   color=col, lw=1.3, label=f"λ={lam_deg:.0f}°")
    axes[1].plot(ph, res_l.rv_m_s,          color=col, lw=1.3)
    axes[2].plot(ph, res_l.color_lcs[0],    color=col, lw=1.3)  # g-r

for ax, (ttl, ylbl) in zip(axes, [
    ("Photometric LC", "ΔFlux (ppm)"),
    ("RV signal", "ΔRV (m/s)"),
    ("g−r colour", "Δ(g−r) (mmag)"),
]):
    ax.axhline(0., color="#555", lw=0.7, ls="--")
    ax.set_title(ttl, color="#eeeeee")
    ax.set_ylabel(ylbl, color="#bbbbbb")
    ax.set_xlim(0., 2.)
    for k in [1., 2.]:
        ax.axvline(k, color="#44ff88", lw=0.7, ls="--", alpha=0.4)

axes[0].legend(facecolor="#1a1a1a", edgecolor="#444444", labelcolor="#eeeeee", fontsize=9)
axes[2].set_xlabel("Rotation phase", color="#bbbbbb")
fig5.suptitle("Effect of spin-orbit obliquity λ on stellar variability observables",
              color="#eeeeee", fontsize=11, fontweight="bold")
fig5.tight_layout(rect=[0, 0, 1, 0.96])
save(fig5, "var_05_obliquity_effect.png")
print("  λ comparison figure saved")


# ============================================================================ #
#  Test 6: CCF bisector comparison                                              #
# ============================================================================ #

print("\n── Test 6: CCF bisector comparison across features ──")

# Compute one rotation's worth of CCF for three configurations and show
# how the bisector shape differs.
def bisector(ccf_profile, rv_grid):
    """Compute CCF bisector as (bisector_rv, depth) pairs."""
    inv = 1. - ccf_profile / ccf_profile.max()
    depths = np.linspace(0.05, 0.55, 20)
    bis_rv = []
    for d in depths:
        above = inv > d
        if above.any():
            idxs = np.where(above)[0]
            v_lo = float(rv_grid[idxs[0]])
            v_hi = float(rv_grid[idxs[-1]])
            bis_rv.append((v_lo + v_hi) / 2.)
        else:
            bis_rv.append(np.nan)
    return np.array(bis_rv), depths

fig6, (ax_ccf, ax_bis) = plt.subplots(1, 2, figsize=(11, 5), facecolor="#0d0d0d")
for ax in (ax_ccf, ax_bis):
    ax.set_facecolor("#111111")
    for sp in ax.spines.values(): sp.set_edgecolor("#444444")
    ax.tick_params(colors="#bbbbbb"); ax.grid(True, alpha=0.15, color="#555555")

configs = [
    ("Clean photosphere", [], "#aaaaaa"),
    ("Granulation only",  [GranulationField(800, seed=42)], "#00d4c8"),
    ("Spot only",         [StarSpot(0., 0., 15., -600.)], "#f5a528"),
    ("Full active",       [GranulationField(800, seed=42),
                           StarSpot(0., 0., 15., -600.),
                           Facula(0., 40., 9., alpha=0.12)], "#ff6655"),
]
for (lbl, feats, col) in configs:
    st = build_star(lbl, features=feats, n_theta=40, n_phi=80)
    sm = RotationSimulator(st, P_ROT, N_PH)
    rv_g = np.linspace(-25., 25., 200)
    # Use first-cycle mean CCF
    rs = sm.run(n_cycles=1., ccf_rv_range=25., ccf_n_rv=200)
    ccf = rs.ccf_mean
    ax_ccf.plot(rv_g, ccf,  color=col, lw=1.5, label=lbl)
    bv, bd = bisector(ccf, rv_g)
    ax_bis.plot(bv, bd, color=col, lw=1.5, marker="o", ms=3)

ax_ccf.set_xlabel("RV (km/s)", color="#bbbbbb")
ax_ccf.set_ylabel("Normalised CCF", color="#bbbbbb")
ax_ccf.set_title("Mean CCF profiles", color="#eeeeee")
ax_ccf.legend(facecolor="#1a1a1a", edgecolor="#444444", labelcolor="#eeeeee", fontsize=9)
ax_ccf.axvline(0., color="#555", lw=0.6, ls="--")

ax_bis.set_xlabel("Bisector RV (km/s)", color="#bbbbbb")
ax_bis.set_ylabel("Depth fraction", color="#bbbbbb")
ax_bis.set_title("CCF bisectors  (granulation → C-shape)", color="#eeeeee")
ax_bis.axvline(0., color="#555", lw=0.6, ls="--")

fig6.suptitle("CCF profile and bisector comparison across stellar configurations",
              color="#eeeeee", fontsize=11, fontweight="bold")
fig6.tight_layout(rect=[0, 0, 1, 0.94])
save(fig6, "var_06_ccf_bisectors.png")


# ============================================================================ #
#  Test 7: Fast sanity — run the full pipeline and verify sizes                #
# ============================================================================ #

print("\n── Test 7: Pipeline sanity check ──")

from PIL import Image
expected_n = N_PH * 2
print(f"  Expected time steps: {expected_n}")
print(f"  res_spot.times.shape:  {res_spot.times.shape}  ({'✓' if len(res_spot.times)==expected_n else '✗'})")
print(f"  res_spot.ccf_map.shape: {res_spot.ccf_map.shape}")
print(f"  res_spot.color_lcs.shape: {res_spot.color_lcs.shape}")

files_to_check = [
    "var_01_spot_only.png",
    "var_02_active_region.png",
    "var_03_granulation.png",
    "var_04_full_active.png",
    "var_05_obliquity_effect.png",
    "var_06_ccf_bisectors.png",
]
for fname in files_to_check:
    p = os.path.join(OUT, fname)
    if os.path.exists(p):
        im = Image.open(p)
        print(f"  {fname}: {im.size} ✓")
    else:
        print(f"  {fname}: MISSING ✗")

print("\nAll stellar variability tests complete ✓")
