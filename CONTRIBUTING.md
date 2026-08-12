# Contributing to starmodel

Thank you for your interest in contributing!  This project is released under
**GPL-3.0-or-later**, so all contributions must be compatible with that licence.

---

## Getting started

```bash
git clone https://github.com/youruser/starmodel
cd starmodel
pip install -e ".[dev]"   # editable install with dev extras
```

---

## Running the test suites

```bash
python -m starmodel.examples.test_transit_viewer
python -m starmodel.examples.test_stellar_variability
```

All tests should pass with no errors before submitting a PR.

---

## Code style

* **Python ≥ 3.10** — use type hints, `match`/`case`, `X | Y` union syntax.
* **Docstrings** — every public class and function must have a NumPy-style
  docstring with `Parameters`, `Returns`, and at least one `Example`.
* **Physical units** — always state units in docstrings.  Internally:
  - lengths in stellar radii R★
  - velocities in km/s
  - temperatures in Kelvin
  - angles in degrees (parameters) or radians (internal computation)
  - times in days

---

## Adding a new surface feature

1. Subclass `SurfaceFeature` in `surface_features.py`.
2. Implement `apply(normals, mus, T_eff_arr, velocity_model, line_of_sight)`.
3. Return a dict with keys `brightness_factor`, `T_eff_delta`, `v_conv`,
   `line_depth_factor` — all `(n_elements,)` arrays.
4. Add a test in `examples/test_stellar_variability.py`.
5. Export the class from `__init__.py`.

---

## Adding a bundled system parameter file

Drop a JSON file following the canonical schema into `starmodel/data/`.
The schema is documented in `system.py`; `WASP-108.json` is the reference
example.

---

## Pull request checklist

- [ ] Tests pass locally
- [ ] New public API has docstrings
- [ ] `CHANGELOG.md` updated
- [ ] No breaking changes to existing notebooks / examples (or clearly documented)

---

## Reporting bugs

Open an issue on GitHub with:
* Python version and OS
* Minimal reproducible example
* Full traceback
