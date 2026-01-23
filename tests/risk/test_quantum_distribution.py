import numpy as np
import pytest


@pytest.mark.unit
def test_compute_var_cvar_ordering():
    from risk.quantum_distribution import compute_var_cvar

    rng = np.random.default_rng(0)
    samples = rng.normal(size=(1000, 1))
    var, cvar = compute_var_cvar(samples, alpha=0.05)

    assert cvar <= var + 1e-6
    assert np.isfinite(var)
    assert np.isfinite(cvar)


@pytest.mark.unit
def test_estimate_adg_risk_outputs():
    from risk.quantum_distribution import estimate_adg_risk

    rng = np.random.default_rng(1)
    returns = rng.normal(size=300)
    metrics = estimate_adg_risk(returns)

    assert "var" in metrics
    assert "cvar" in metrics
    assert np.isfinite(metrics["var"])
    assert np.isfinite(metrics["cvar"])
