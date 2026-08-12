"""
test_transit_viewer.py
======================
Full integration test and visual demonstration of the starmodel transit
epoch viewer.  Exercises:

  * Quadratic limb darkening
  * Stellar rotation with inclination
  * Spin-orbit obliquity (λ = 0°, 60°, 90°, 180°)
  * Differential rotation
  * Spectral lines with Doppler shifts
  * Transit chord, planet disk, lat/lon grid aligned with rotation axis
  * All four contact-time epochs (T1, mid-transit, T3, out-of-transit)
  * Eccentric orbit variant
  * Obliquity comparison figure (RM curves side-by-side)

Run from the repository root:
    PYTHONPATH=. python starmodel/test_transit_viewer.py
"""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from starmodel import Star, TransitModel, OrbitalParameters
from starmodel.transit_viewer import plot_transit_epoch

OUT = os.path.dirname(os.path.abspath(__file__))


# =========================================================================== #
#  Helper                                                                      #
# =========================================================================== #

def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {path}")


def _make_star(name, obliquity=0., inclination=90., v_eq=5.,
               diff_rot=0., n_theta=40, n_phi=80):
    """Build a fully configured Star with spectrum and temperature gradient."""
    import math as _math
    wl = np.linspace(6300, 6900, 400)   # centred on Hα for good line profile
    star = (
        Star(radius=1.0, n_theta=n_theta, n_phi=n_phi, name=name)
        .set_brightness(law="quadratic", coefficients={"a": 0.45, "b": 0.30})
        .set_rotation(v_eq=v_eq, inclination=inclination,
                      obliquity=obliquity,
                      differential_rotation=diff_rot)
        .set_spectrum(wl)
        .add_spectral_line(center=6563., depth=0.65, width=1.5,
                           kind="absorption")   # Hα
        # Temperature gradient: equator cooler, poles hotter
        # mimics a real star's centre-to-limb temperature variation
        .set_temperature_map(
            lambda e: 5500. + 600. * abs(_math.cos(e.theta)),
            key="T_eff",
        )
        .compute()
    )
    print(f"  {star}")
    return star


def _make_orbit(obliquity=0., impact_b=0.3, e=0., omega=90.):
    return OrbitalParameters(
        period=3.5, t0=0., semi_major_axis=12.,
        inclination=89., planet_radius=0.12,
        impact_parameter=impact_b,
        eccentricity=e, omega=omega,
        obliquity=obliquity,
    )


# =========================================================================== #
#  Test 1: aligned system – four transit epochs                                #
# =========================================================================== #

print("\n── Test 1: Aligned system (λ=0°), four transit epochs ──")
star_a = _make_star("Aligned  λ=0°", obliquity=0.)
orbit_a = _make_orbit(obliquity=0.)
model_a = TransitModel(star_a, orbit_a)
result_a = model_a.compute(n_times=600, compute_ccf=True, compute_spectrum=True)
print(f"  {model_a}")
print(result_a.summary())

ct = result_a.contact_times
epochs_a = {
    "out_of_transit": orbit_a.t0 - 0.12,
    "ingress_T1":     ct.get("T1", orbit_a.t0 - 0.05) + 0.005,
    "mid_transit":    orbit_a.t0,
    "egress_T4":      ct.get("T4",  orbit_a.t0 + 0.05) - 0.005,
}

for label, t_ep in epochs_a.items():
    if math.isnan(t_ep):
        t_ep = orbit_a.t0
    print(f"  rendering epoch: {label}  t={t_ep:.5f} d")
    fig = plot_transit_epoch(
        star_a, model_a, result_a,
        t_epoch=t_ep,
        time_format="hours",
    )
    _save(fig, f"test_aligned_{label}.png")


# =========================================================================== #
#  Test 2: misaligned λ=60° – mid-transit with visible grid tilt              #
# =========================================================================== #

print("\n── Test 2: Misaligned system (λ=60°) ──")
star_60 = _make_star("Misaligned  λ=60°", obliquity=60.)
orbit_60 = _make_orbit(obliquity=60.)
model_60 = TransitModel(star_60, orbit_60)
result_60 = model_60.compute(n_times=600, compute_ccf=True, compute_spectrum=True)
print(f"  {model_60}")

fig = plot_transit_epoch(
    star_60, model_60, result_60,
    t_epoch=orbit_60.t0,
    time_format="hours",
)
_save(fig, "test_lambda60_mid.png")

# Also ingress to see the asymmetric RM
t_ing = result_60.contact_times.get("T1", orbit_60.t0 - 0.05) + 0.01
fig = plot_transit_epoch(
    star_60, model_60, result_60,
    t_epoch=t_ing,
    time_format="hours",
)
_save(fig, "test_lambda60_ingress.png")


# =========================================================================== #
#  Test 3: retrograde (λ=180°) – RM sign reversal                             #
# =========================================================================== #

print("\n── Test 3: Retrograde system (λ=180°) ──")
star_180 = _make_star("Retrograde  λ=180°", obliquity=180.)
orbit_180 = _make_orbit(obliquity=180.)
model_180 = TransitModel(star_180, orbit_180)
result_180 = model_180.compute(n_times=600, compute_ccf=True, compute_spectrum=True)
print(f"  {model_180}")

fig = plot_transit_epoch(
    star_180, model_180, result_180,
    t_epoch=orbit_180.t0,
    time_format="hours",
)
_save(fig, "test_lambda180_mid.png")


# =========================================================================== #
#  Test 4: pole-on inclination (i=30°) with obliquity                         #
# =========================================================================== #

print("\n── Test 4: Inclined pole (i=30°, λ=45°) ──")
star_i30 = _make_star("i=30°  λ=45°", inclination=30., obliquity=45., v_eq=8.)
orbit_i30 = _make_orbit(obliquity=45.)
model_i30 = TransitModel(star_i30, orbit_i30)
result_i30 = model_i30.compute(n_times=600, compute_ccf=True, compute_spectrum=True)
print(f"  {model_i30}")

fig = plot_transit_epoch(
    star_i30, model_i30, result_i30,
    t_epoch=orbit_i30.t0,
    time_format="hours",
)
_save(fig, "test_i30_lambda45_mid.png")


# =========================================================================== #
#  Test 5: differential rotation with obliquity                                #
# =========================================================================== #

print("\n── Test 5: Differential rotation (α=0.2, λ=30°) ──")
star_dr = _make_star("DiffRot  α=0.2  λ=30°",
                     obliquity=30., diff_rot=0.2, v_eq=6.)
orbit_dr = _make_orbit(obliquity=30.)
model_dr = TransitModel(star_dr, orbit_dr)
result_dr = model_dr.compute(n_times=600, compute_ccf=True, compute_spectrum=True)
print(f"  {model_dr}")

fig = plot_transit_epoch(
    star_dr, model_dr, result_dr,
    t_epoch=orbit_dr.t0,
    time_format="hours",
)
_save(fig, "test_diffrot_lambda30_mid.png")


# =========================================================================== #
#  Test 6: eccentric orbit                                                     #
# =========================================================================== #

print("\n── Test 6: Eccentric orbit (e=0.35, λ=0°) ──")
star_ecc = _make_star("Eccentric  e=0.35", obliquity=0.)
orbit_ecc = OrbitalParameters(
    period=3.5, t0=0., semi_major_axis=12.,
    inclination=89., planet_radius=0.12,
    eccentricity=0.35, omega=60.,
    obliquity=0.,
)
model_ecc = TransitModel(star_ecc, orbit_ecc)
result_ecc = model_ecc.compute(n_times=600, compute_ccf=True, compute_spectrum=True)
print(f"  {model_ecc}")
print(result_ecc.summary())

fig = plot_transit_epoch(
    star_ecc, model_ecc, result_ecc,
    t_epoch=orbit_ecc.t0,
    time_format="hours",
)
_save(fig, "test_eccentric_mid.png")


# =========================================================================== #
#  Test 7: pole_tilt_deg on top of obliquity                                   #
# =========================================================================== #

print("\n── Test 7: pole_tilt_deg=45° on top of λ=30° ──")
# pole_tilt_deg adds an additional sky-plane rotation beyond what obliquity
# already sets.  Should result in a further 45° rotation of the grid.
fig = plot_transit_epoch(
    star_dr, model_dr, result_dr,
    t_epoch=orbit_dr.t0,
    pole_tilt_deg=45.,
    time_format="hours",
)
_save(fig, "test_extra_pole_tilt45.png")


# =========================================================================== #
#  Test 8: RM comparison plot for all obliquities                              #
# =========================================================================== #

print("\n── Test 8: RM anomaly comparison across obliquities ──")

systems = [
    (0,   star_a,   result_a),
    (60,  star_60,  result_60),
    (90,  None,     None),       # build on-the-fly
    (180, star_180, result_180),
]

# Build λ=90°
star_90  = _make_star("λ=90°", obliquity=90.)
orbit_90 = _make_orbit(obliquity=90.)
model_90 = TransitModel(star_90, orbit_90)
result_90 = model_90.compute(n_times=600, compute_ccf=True)
systems[2] = (90, star_90, result_90)

fig_rm, (ax_lc, ax_rv) = plt.subplots(
    2, 1, figsize=(11, 7), facecolor="#111111")
for ax in (ax_lc, ax_rv):
    ax.set_facecolor("#111111")
    for sp in ax.spines.values(): sp.set_edgecolor("#444444")
    ax.tick_params(colors="#bbbbbb")
    ax.xaxis.label.set_color("#bbbbbb")
    ax.yaxis.label.set_color("#bbbbbb")
    ax.title.set_color("#eeeeee")
    ax.grid(True, alpha=0.15, color="#555555")

colors_obl = ["#5599ff", "#ffaa22", "#ee44ff", "#ff4455"]
for (lam, _star, res), col in zip(systems, colors_obl):
    t_h = (res.times - 0.) * 24.
    ax_lc.plot(t_h, res.flux,     color=col, lw=1.6, label=f"λ = {lam}°")
    ax_rv.plot(t_h, res.delta_rv, color=col, lw=1.6, label=f"λ = {lam}°")

# Contact lines from aligned result
for lbl, col_c, ls_c in [("T1","#44ff88","-"),("T2","#44ff88","--"),
                          ("T3","#44ff88","--"),("T4","#44ff88","-")]:
    tv = result_a.contact_times.get(lbl, float("nan"))
    if not math.isnan(tv):
        xv = tv * 24.
        for ax in (ax_lc, ax_rv):
            ax.axvline(xv, color=col_c, lw=0.8, ls=ls_c, alpha=0.45)

ax_lc.axhline(1., color="#555555", lw=0.6, ls="--")
ax_rv.axhline(0., color="#555555", lw=0.7, ls="--")

ax_lc.set_ylabel("Normalised flux",    color="#bbbbbb")
ax_lc.set_title("Light curve — obliquity comparison", color="#eeeeee")
ax_lc.legend(facecolor="#1a1a1a", edgecolor="#444444",
             labelcolor="#eeeeee", fontsize=9)

ax_rv.set_xlabel("Time from mid-transit (h)", color="#bbbbbb")
ax_rv.set_ylabel("ΔRV (km/s)",               color="#bbbbbb")
ax_rv.set_title("Rossiter-McLaughlin anomaly — obliquity comparison",
                color="#eeeeee")
ax_rv.legend(facecolor="#1a1a1a", edgecolor="#444444",
             labelcolor="#eeeeee", fontsize=9)

fig_rm.tight_layout(rect=[0, 0, 1, 0.97])
fig_rm.suptitle("Spin-orbit obliquity effect on transit observables",
                color="#eeeeee", fontsize=11, fontweight="bold")
_save(fig_rm, "test_rm_obliquity_comparison.png")


# =========================================================================== #
#  Test 9: custom lat/lon grid style                                           #
# =========================================================================== #

print("\n── Test 9: Custom grid style (denser grid, yellow) ──")
fig = plot_transit_epoch(
    star_60, model_60, result_60,
    t_epoch=orbit_60.t0,
    latlon_dlat=15., latlon_dlon=15.,
    latlon_color="#ffee55",
    latlon_alpha=0.65,
    latlon_lw=0.9,
    time_format="hours",
)
_save(fig, "test_custom_grid_style.png")


# =========================================================================== #
#  Test 10: external high-resolution template spectrum                         #
# =========================================================================== #

print("\n── Test 10: External template spectrum loaded from file ──")

# Generate a synthetic high-resolution template and save it as ASCII
wl_hr   = np.linspace(5700, 6800, 8000)          # 0.14 Å sampling
# Synthetic G-star continuum with multiple absorption lines
flux_hr = np.ones_like(wl_hr)
for lc, depth, sigma in [
    (6563.0, 0.80, 0.8),   # Hα
    (5893.0, 0.45, 0.5),   # Na D
    (6302.5, 0.15, 0.3),   # telluric-like
    (5889.9, 0.30, 0.4),   # Na D2
    (5875.6, 0.20, 0.4),   # He I
]:
    flux_hr *= 1.0 - depth * np.exp(-0.5 * ((wl_hr - lc) / sigma)**2)

# Write to a two-column ASCII file
template_path = os.path.join(OUT, "synthetic_template.dat")
np.savetxt(template_path,
           np.column_stack([wl_hr, flux_hr]),
           header="wavelength_A  flux_norm",
           fmt="%.5f  %.8f")
print(f"  Template saved → {template_path}  ({len(wl_hr)} points)")

# Build star that loads the external template
wl_model = np.linspace(5800, 6700, 400)
star_tmpl = (
    Star(radius=1.0, n_theta=40, n_phi=80, name="Template star  λ=0°")
    .set_brightness(law="quadratic", coefficients={"a": 0.45, "b": 0.30})
    .set_rotation(v_eq=5.0, inclination=90.0, obliquity=0.)
    .set_spectrum(wl_model)
    .add_spectral_line(center=6563., depth=0.65, width=1.5)   # needed for element spectra
    .add_spectral_line(center=5893., depth=0.35, width=1.2)
    .load_template(template_path)               # ← use HR template for CCF
    .compute()
)
print(f"  {star_tmpl}")
print(f"  has_template = {star_tmpl._spectrum.has_template}")

orbit_tmpl = _make_orbit(obliquity=0.)
model_tmpl = TransitModel(star_tmpl, orbit_tmpl)
result_tmpl = model_tmpl.compute(
    n_times=600, compute_ccf=True, compute_spectrum=True
)
print(f"  {model_tmpl}")
print(result_tmpl.summary())

fig = plot_transit_epoch(
    star_tmpl, model_tmpl, result_tmpl,
    t_epoch=orbit_tmpl.t0,
    time_format="hours",
)
_save(fig, "test_template_mid.png")

# Compare CCF with Gaussian-mask version
print("\n  CCF peak comparison (template vs Gaussian mask):")
print(f"    Template  CCF max  : {result_tmpl.ccf_oot.max():.4f}")
print(f"    Gaussian  CCF max  : {result_a.ccf_oot.max():.4f}")


# =========================================================================== #
#  Test 11: PlanetarySystem file loader — WASP-108 end-to-end                 #
# =========================================================================== #

print("\n── Test 11: PlanetarySystem loader (WASP-108.json) ──")
from starmodel import PlanetarySystem

wasp108_json = os.path.join(OUT, "WASP-108.json")
sys108 = PlanetarySystem(wasp108_json)
print(sys108.summary())

# Round-trip: write back to JSON and CSV
sys108.to_json(os.path.join(OUT, "WASP-108_roundtrip.json"))
sys108.to_csv (os.path.join(OUT, "WASP-108.csv"))

# Build Star and OrbitalParameters from the file
wl_108 = np.linspace(6300, 6900, 400)
star108 = sys108.build_star(n_theta=40, n_phi=80, wavelengths=wl_108)
star108.add_spectral_line(center=6563., depth=0.65, width=1.5,
                           kind="absorption")
star108.compute()
print(f"\n  {star108}")

orbit108 = sys108.build_orbit()
print(orbit108.summary())

# Full transit computation
model108  = TransitModel(star108, orbit108)
result108 = model108.compute(n_times=500, compute_ccf=True,
                             compute_spectrum=True)
print(result108.summary())

# Epoch figure
fig = plot_transit_epoch(star108, model108, result108,
                          t_epoch=orbit108.t0,
                          time_format="hours",
                          figsize=(14, 16))
_save(fig, "wasp108_mid_transit.png")

# Also load from the CSV round-trip to verify CSV loading
sys108_csv = PlanetarySystem(os.path.join(OUT, "WASP-108.csv"))
print(f"\n  CSV round-trip: {sys108_csv}")
star108c  = sys108_csv.build_star(n_theta=20, n_phi=40, wavelengths=wl_108)
star108c.add_spectral_line(6563., 0.65, 1.5)
star108c.compute()
orbit108c = sys108_csv.build_orbit()
print(f"  v_eq from CSV: {sys108_csv.star.v_eq:.4f} km/s")
print(f"  Period from CSV: {sys108_csv.planet.orbital_period_days.value:.8f} days")


# =========================================================================== #
#  Size consistency check (updated)                                            #
# =========================================================================== #

print("\n── Size consistency check ──")
from PIL import Image
files = [
    "test_aligned_mid_transit.png",
    "test_lambda60_mid.png",
    "test_lambda180_mid.png",
    "test_i30_lambda45_mid.png",
]
sizes = set()
for f in files:
    p = os.path.join(OUT, f)
    if os.path.exists(p):
        im = Image.open(p)
        sizes.add(im.size)
        print(f"  {f}: {im.size}")

if len(sizes) == 1:
    print("  ✓ All figures are the same size.")
else:
    print("  ✗ Size mismatch detected!", sizes)

print("\nAll tests complete.")
