# starmodel

**Stellar surface & planetary transit modelling framework**

`starmodel` is a Python library for modelling physical quantities on the surface of a spherical star and simulating the observable effects of planetary transits, stellar rotation, and surface activity.

---

## Features

| Module | Capability |
|---|---|
| `element`, `grid` | Spherical surface discretised into (θ,φ) grid cells |
| `quantities` | Brightness (5 limb-darkening laws), rotational velocity with obliquity, per-element spectra |
| `star` | Top-level model combining grid + physics |
| `surface_features` | Granulation, starspots, faculae, plages |
| `transit` | Keplerian transit: light curve, RM anomaly, CCF map, colour LCs |
| `transit_viewer` | 5-panel epoch figure |
| `transit_overview` | Compact 4-panel figure |
| `stellar_variability` | Full-rotation variability simulator |
| `system` | Load/save system parameters from JSON or CSV |

---

## Install

```bash
pip install starmodel
```

Or from source:

```bash
git clone https://github.com/youruser/starmodel
cd starmodel
pip install .
```

**Dependencies:** `numpy`, `scipy`, `matplotlib`  
**Optional:** `astropy` (for FITS template loading)

---

## Quick start

### Transit simulation (WASP-108)

```python
from starmodel import PlanetarySystem, TransitModel, plot_transit_overview, get_data_path
import numpy as np

# Load published system parameters
sys = PlanetarySystem(get_data_path("WASP-108.json"))

star  = sys.build_star(wavelengths=np.linspace(6300, 6900, 400))
star.add_spectral_line(6563., depth=0.65, width=1.5)
star.compute()

orbit  = sys.build_orbit()
result = TransitModel(star, orbit).compute(n_times=500, compute_ccf=True)

fig = plot_transit_overview(star, TransitModel(star, orbit), result, t_epoch=orbit.t0)
fig.savefig("transit.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
```

### Stellar rotation variability

```python
from starmodel import Star, StarSpot, Facula, GranulationField
from starmodel.stellar_variability import RotationSimulator

star = (Star(n_theta=40, n_phi=80, name="Active star")
        .set_brightness(law="quadratic", coefficients={"a": 0.45, "b": 0.30})
        .set_rotation(v_eq=2., inclination=90.)
        .set_temperature_map(lambda e: 5778.))

star.add_feature(GranulationField(n_cells=800, seed=42))
star.add_feature(StarSpot(lat_deg=20., lon_deg=0., radius_deg=12., T_contrast=-550.))
star.add_feature(Facula(lat_deg=20., lon_deg=35., radius_deg=8., alpha=0.10))
star.compute()

sim    = RotationSimulator(star, rotation_period_days=25., n_phases_per_cycle=300)
result = sim.run(n_cycles=2.)
print(result.summary())

fig = sim.plot(result)
fig.savefig("variability.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
```

### Parameter files

```python
from starmodel import PlanetarySystem, get_data_path, list_data_files

# See bundled systems
print(list_data_files())           # ['WASP-108.csv', 'WASP-108.json', ...]

# Load & inspect
sys = PlanetarySystem(get_data_path("WASP-108.json"))
print(sys.summary())

# Export after fitting
sys.to_json("WASP-108_updated.json")
sys.to_csv ("WASP-108_updated.csv")
```

---

## Coordinate conventions

- **lon = 0°** always faces the observer (sub-observer longitude).
- **pole_tilt_deg** in the viewer adds extra sky-plane rotation on top of the obliquity.
- **North pole** follows the right-hand rule with respect to the rotation direction.

---

## Run the examples

```bash
# Basic surface model
python -m starmodel.examples.example

# Transit simulation
python -m starmodel.examples.example_transit

# Rotation variability
python -m starmodel.examples.example_variability

# Full test suites
python -m starmodel.examples.test_transit_viewer
python -m starmodel.examples.test_stellar_variability
```
