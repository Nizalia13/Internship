# Changelog

## 0.2.0

- Simplified the public API to match the professor's generic CSV workflow.
- Renamed the main public class to `ConformalSetModel`.
- Made one learned envelope per label explicit through `model.envelopes_`.
- Added `score_direction="higher_is_better" | "lower_is_better"`.
- Renamed radial parameters for readability:
  - `M` -> `n_directions`
  - `kappa` -> `smoothing`
  - `angular_bandwidth_deg` -> `angle_deg`
- Renamed strip parameters:
  - `NB` -> `n_bins`
  - `monotonic_window` -> `smoothing_window`
- Retained advanced `label_to_columns` support for the original host-style data.
- Added documentation mapping the professor's handwritten workflow to the API.
- Added tests, example CSV generator, save/load, metrics and 2D visualization.

## 0.1.0

- Initial package draft extracted from the cleaned host-classification notebook.
