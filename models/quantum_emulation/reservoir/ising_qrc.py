"""Quantum reservoir computing emulation using an Ising Hamiltonian."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.quantum_emulation.backends.approximate import ApproximateConfig, ApproximateReservoir
from models.quantum_emulation.backends.numpy_statevector import HamiltonianSpec, StateVectorSimulator
from models.quantum_emulation.interfaces import LatentStateModel


@dataclass
class IsingQRCConfig:
    n_qubits: int = 8
    n_input_qubits: int = 4
    n_memory_qubits: int = 4
    dt: float = 0.1
    steps: int = 1
    coupling_scale: float = 0.5
    field_scale: float = 0.5
    input_scale: float = 1.0
    seed: int = 42
    measure_pairs: bool = False
    burn_in: int = 0
    max_statevector_qubits: int = 10


class IsingQuantumReservoir(LatentStateModel):
    """Fixed-reservoir feature generator with trained readout downstream."""

    def __init__(self, cfg: IsingQRCConfig | None = None):
        self.cfg = cfg or IsingQRCConfig()
        if self.cfg.n_input_qubits > self.cfg.n_qubits:
            self.cfg.n_input_qubits = self.cfg.n_qubits
            self.cfg.n_memory_qubits = 0
        if self.cfg.n_input_qubits + self.cfg.n_memory_qubits != self.cfg.n_qubits:
            self.cfg.n_memory_qubits = max(self.cfg.n_qubits - self.cfg.n_input_qubits, 0)
        self._rng = np.random.default_rng(self.cfg.seed)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._backend = self._build_backend()

    def _build_backend(self):
        if self.cfg.n_qubits <= self.cfg.max_statevector_qubits:
            couplings = self._rng.normal(scale=self.cfg.coupling_scale,
                                         size=(self.cfg.n_qubits, self.cfg.n_qubits))
            couplings = (couplings + couplings.T) / 2.0
            np.fill_diagonal(couplings, 0.0)
            transverse = self._rng.normal(scale=self.cfg.field_scale, size=self.cfg.n_qubits)
            spec = HamiltonianSpec(
                n_qubits=self.cfg.n_qubits,
                couplings=couplings,
                transverse_field=transverse,
                dt=self.cfg.dt,
            )
            return StateVectorSimulator(spec)
        cfg = ApproximateConfig(
            n_units=self.cfg.n_qubits,
            spectral_radius=0.9,
            input_scale=self.cfg.input_scale,
            seed=self.cfg.seed,
        )
        return ApproximateReservoir(cfg)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "IsingQuantumReservoir":
        X = np.asarray(X, dtype=np.float64)
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self._mean is None or self._std is None:
            self.fit(X)
        X_norm = (X - self._mean) / self._std
        X_norm = np.clip(X_norm, -3.0, 3.0) * self.cfg.input_scale

        n_steps = X_norm.shape[0]
        features = []
        if hasattr(self._backend, "reset"):
            self._backend.reset()

        for t in range(n_steps):
            if isinstance(self._backend, StateVectorSimulator):
                n_inputs = min(self.cfg.n_input_qubits, self.cfg.n_qubits)
                for q in range(n_inputs):
                    angle = X_norm[t, q % X_norm.shape[1]]
                    self._backend.apply_rotation_x(angle, q)
                self._backend.evolve(self.cfg.steps)
                z_feats = self._backend.z_expectations()
                if self.cfg.measure_pairs:
                    zz_feats = self._backend.zz_expectations()
                    step_feats = np.concatenate([z_feats, zz_feats])
                else:
                    step_feats = z_feats
            else:
                step_feats = self._backend.step(X_norm[t])
            features.append(step_feats)

        feature_matrix = np.vstack(features)
        if self.cfg.burn_in > 0:
            feature_matrix[: self.cfg.burn_in, :] = np.nan
        return feature_matrix

    def feature_names(self) -> list[str]:
        n = self.cfg.n_qubits
        names = [f"qrc_z_{i}" for i in range(n)]
        if self.cfg.measure_pairs:
            for i in range(n):
                for j in range(i + 1, n):
                    names.append(f"qrc_zz_{i}_{j}")
        return names


__all__ = ["IsingQRCConfig", "IsingQuantumReservoir"]
