# Mapping from the cleaned research notebook to this package

This file explains where the research-code ideas live in the library.

| Research idea | Library location |
|---|---|
| `1 - score` nonconformity | `missingness.py -> transform_scores` |
| missingness / NaN-preserving logic | `missingness.py` and each envelope module |
| finite-sample conformal quantile | `calibration.py` |
| `collapsed_build_envelope` | `envelopes/collapsed.py -> build_collapsed` |
| `collapsed_tau` | `envelopes/collapsed.py -> collapsed_tau` |
| positive sphere directions | `envelopes/radial.py -> sample_positive_sphere` |
| radial angular smoothing (`kappa`) | `radial.py -> smoothing` |
| radial angular bandwidth | `radial.py -> angle_deg` |
| `strip_shape_discovery` | `envelopes/strip.py` |
| strip bin indices | `envelopes/strip.py -> get_bin_indices` |
| strip NaN fallback rules | `envelopes/strip.py -> strip_tau_scores` |
| one envelope per true label | `model.py -> ConformalSetModel.fit` |
| prediction sets | `model.py -> predict` |
| forced nonempty fallback | `model.py -> predict` |
| 2D envelope view | `model.py -> plot` |

## Naming changes made for a general library

Some notebook names were intentionally made more readable:

- `M` -> `n_directions`
- `kappa` -> `smoothing`
- `angular_bandwidth_deg` -> `angle_deg`
- `NB` / `number_of_bins` -> `n_bins`
- `monotonic_window` -> `smoothing_window`

The constructor still accepts several of the old names as aliases where useful.
