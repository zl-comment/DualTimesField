# Joint historical correction

Local implementation; no anniversary retrieval. See [the temporal contract and design](../../docs/full_v5_joint_history_20261019.md).

The LightGBM implementation is the [official library](https://github.com/microsoft/LightGBM), invoked through the existing local aligned adapter. The differentiable historical dual field is an existing local implementation; this package creates a fresh instance and jointly optimizes it with calibration, state matching, event alignment and adoption gates. Tree fitting stays separate and chronological.

Entry points: `python -m forecasting.full_v5_joint.prepare`, `.train`, `.smoke` (use the full module prefix for each).

Configuration: `configs/full_v5_joint/protocol.yaml`. Formal development runs require seven authenticated OOF bundles per region; archived short-window smoke results do not satisfy that contract.
