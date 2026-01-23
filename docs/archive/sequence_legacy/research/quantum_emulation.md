# Quantum Emulation (Experimental)

This directory documents the **quantum-inspired emulation** modules that can be
plugged into Sequence without changing the core training loop. The goal is to
isolate experimental quantum techniques behind stable interfaces and enable
classical fallbacks by default.

## Scope

- **QRC (Quantum Reservoir Computing)** as a feature block for volatility/regime
  signals.
- **Quantum-walk ADG** as a distribution generator for risk/scenario analysis.
- **Discrete selection helpers** for constrained universe selection.

These modules are **opt-in** and intended for research-mode experiments.

## Usage

Enable QRC features by passing the feature group flag:

```
python train/run_training.py --pairs gbpusd --feature-groups quantum_reservoir
```

Defaults are controlled in `run/config/config.py` under `FeatureConfig` (look
for the `qrc_*` knobs).

## Notes

- The current emulation backend uses small-N statevector simulation when
  `n_qubits <= 10`, and falls back to a lightweight recurrent approximation
  for larger configurations.
- ADG distributions currently feed risk metrics only; they do not alter the
  training set unless explicitly wired into experiments.
