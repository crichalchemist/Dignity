import numpy as np
import pytest


@pytest.mark.unit
class TestQuantumReservoir:
    def test_qrc_feature_shape_and_names(self):
        from models.quantum_emulation.reservoir.ising_qrc import IsingQRCConfig, IsingQuantumReservoir

        rng = np.random.default_rng(123)
        X = rng.normal(size=(12, 4))
        cfg = IsingQRCConfig(n_qubits=4, n_input_qubits=2, n_memory_qubits=2, seed=123)
        model = IsingQuantumReservoir(cfg)
        features = model.fit_transform(X)

        assert features.shape == (12, 4)
        assert np.isfinite(features).all()
        assert len(model.feature_names()) == 4

    def test_qrc_pair_measurements(self):
        from models.quantum_emulation.reservoir.ising_qrc import IsingQRCConfig, IsingQuantumReservoir

        rng = np.random.default_rng(42)
        X = rng.normal(size=(6, 3))
        cfg = IsingQRCConfig(
            n_qubits=3,
            n_input_qubits=2,
            n_memory_qubits=1,
            measure_pairs=True,
            seed=42,
        )
        model = IsingQuantumReservoir(cfg)
        features = model.fit_transform(X)

        # z features (3) + zz pairs (3) = 6
        assert features.shape == (6, 6)
        assert len(model.feature_names()) == 6


@pytest.mark.unit
class TestQuantumWalkADG:
    def test_adg_sample_shape_and_pdf(self):
        from models.quantum_emulation.walks.adg import SplitStepQuantumWalkADG

        rng = np.random.default_rng(7)
        returns = rng.normal(size=200)
        model = SplitStepQuantumWalkADG()
        model.fit(returns)

        samples = model.sample(None, n_paths=8, horizon=5)
        assert samples.shape == (8, 5)
        assert np.isfinite(samples).all()

        grid = np.linspace(-1.0, 1.0, 11)
        pdf = model.pdf(None, grid)
        assert pdf.shape == grid.shape
        assert (pdf >= 0).all()
        assert pdf.sum() > 0
