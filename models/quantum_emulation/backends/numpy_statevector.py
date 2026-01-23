"""Small-N statevector simulator for quantum emulation.

This backend is intentionally minimal and intended for research-scale experiments
(<= 10 qubits). It uses dense numpy arrays and eigendecomposition for unitary
propagation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _pauli() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    identity = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return identity, x, y, z


def _kron_all(ops: list[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def _single_qubit_operator(op: np.ndarray, target: int, n_qubits: int) -> np.ndarray:
    identity, _, _, _ = _pauli()
    ops = [identity for _ in range(n_qubits)]
    ops[target] = op
    return _kron_all(ops)


def _two_qubit_operator(op_a: np.ndarray, op_b: np.ndarray, i: int, j: int, n_qubits: int) -> np.ndarray:
    identity, _, _, _ = _pauli()
    ops = [identity for _ in range(n_qubits)]
    ops[i] = op_a
    ops[j] = op_b
    return _kron_all(ops)


def rotation_x(theta: float) -> np.ndarray:
    cos = np.cos(theta / 2.0)
    sin = np.sin(theta / 2.0)
    return np.array([[cos, -1.0j * sin], [-1.0j * sin, cos]], dtype=np.complex128)


@dataclass
class HamiltonianSpec:
    n_qubits: int
    couplings: np.ndarray
    transverse_field: np.ndarray
    dt: float


class StateVectorSimulator:
    """Dense statevector simulator for small qubit counts."""

    def __init__(self, spec: HamiltonianSpec):
        self.spec = spec
        self.n_qubits = spec.n_qubits
        self.dim = 2 ** self.n_qubits
        self.state = np.zeros(self.dim, dtype=np.complex128)
        self.state[0] = 1.0 + 0.0j
        self._z_signs = self._build_z_signs()
        self._u_step = self._build_unitary()

    def reset(self) -> None:
        self.state[:] = 0.0 + 0.0j
        self.state[0] = 1.0 + 0.0j

    def _build_unitary(self) -> np.ndarray:
        identity, x, _, z = _pauli()
        hamiltonian = np.zeros((self.dim, self.dim), dtype=np.complex128)
        # Transverse field terms.
        for idx in range(self.n_qubits):
            hamiltonian += self.spec.transverse_field[idx] * _single_qubit_operator(x, idx, self.n_qubits)
        # ZZ couplings.
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                if self.spec.couplings[i, j] == 0.0:
                    continue
                hamiltonian += self.spec.couplings[i, j] * _two_qubit_operator(z, z, i, j, self.n_qubits)
        # Unitary from eigendecomposition (Hermitian).
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        phases = np.exp(-1.0j * self.spec.dt * eigenvalues)
        return (eigenvectors * phases) @ eigenvectors.conj().T

    def apply_rotation_x(self, theta: float, target: int) -> None:
        op = rotation_x(theta)
        full_op = _single_qubit_operator(op, target, self.n_qubits)
        self.state = full_op @ self.state

    def evolve(self, steps: int = 1) -> None:
        if steps <= 0:
            return
        if steps == 1:
            self.state = self._u_step @ self.state
            return
        u_power = np.linalg.matrix_power(self._u_step, steps)
        self.state = u_power @ self.state

    def z_expectations(self) -> np.ndarray:
        probs = np.abs(self.state) ** 2
        return probs @ self._z_signs

    def zz_expectations(self) -> np.ndarray:
        probs = np.abs(self.state) ** 2
        n_qubits = self.n_qubits
        features = []
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                features.append((probs * self._z_signs[:, i] * self._z_signs[:, j]).sum())
        return np.array(features, dtype=np.float64)

    def _build_z_signs(self) -> np.ndarray:
        signs = np.empty((self.dim, self.n_qubits), dtype=np.float64)
        for idx in range(self.dim):
            for q in range(self.n_qubits):
                bit = (idx >> (self.n_qubits - q - 1)) & 1
                signs[idx, q] = 1.0 if bit == 0 else -1.0
        return signs


__all__ = ["HamiltonianSpec", "StateVectorSimulator", "rotation_x"]
