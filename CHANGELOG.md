# Changelog

All notable changes are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.1.0] — 2025-08

### Added
- `SurfaceGrid` — spherical discretisation with uniform and equal-area modes
- `Brightness` — 5 limb-darkening laws (uniform, linear, quadratic, sqrt, Claret 4-term)
- `Velocity` — rotational velocity with inclination, obliquity λ, differential rotation
- `Spectrum` — per-element Planck continuum + Gaussian lines; external template loading
- `Star` — top-level model with `add_feature()` pipeline
- `surface_features` — `GranulationField`, `StarSpot`, `Facula`, `Plage`
- `TransitModel` — Keplerian transit with light curve, RM anomaly, CCF map, SDSS colour LCs, transmission spectrum
- `OrbitalParameters` — full Keplerian elements including spin-orbit obliquity
- `plot_transit_epoch` — 5-panel epoch viewer with lat/lon grid aligned to rotation axis
- `plot_transit_overview` — compact 4-panel figure
- `RotationSimulator` — full-rotation variability simulator (flux, colour LCs, RV, CCF)
- `PlanetarySystem` — JSON/CSV parameter file loader with `[value, uncertainty]` pairs
- Bundled data files: `WASP-108.json`, `WASP-108.csv`, `synthetic_template.dat`
- `get_data_path()` / `list_data_files()` package helpers
- Educational Jupyter notebooks: TESS, SPARC4, GHOST/Gemini

### Fixed
- Limb-darkening recomputed at each rotation phase in `RotationSimulator` (no spurious 100% flux modulation)
- Lat/lon grid and surface feature coordinates share the same LOS-projected frame (lon = 0° always faces observer)
- Planet moves from negative to positive x with increasing time (prograde convention)
- RHR North pole: `N` label placed at `−rotation_axis` (correct right-hand convention)
