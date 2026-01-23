# Sequence FX Trading System Documentation

**Last Updated**: 2026-01-17
**Version**: Phase 3 Complete

Welcome to the Sequence FX trading system documentation. This directory contains comprehensive guides, API references,
and implementation details for the reinforcement learning-based forex trading platform.

---

## Quick Navigation

### 🚀 Getting Started

- **[Architecture & API Reference](api/ARCHITECTURE_API_REFERENCE.md)** - System overview, module reference, training
  workflow
- **[Configuration Reference](CONFIGURATION_REFERENCE.md)** - Production-ready configurations and parameter tuning

### 📊 Testing & Validation

- **[Testing & Validation Report](TESTING_VALIDATION_REPORT.md)** - Comprehensive test results (25/25 tests passing)

### 🔧 Implementation Guides

- **[Phase 3 Implementation Summary](implementation/PHASE_3_IMPLEMENTATION_SUMMARY.md)** - Transaction costs, position
  sizing, risk management
- **[Phase 3 Quick Start](guides/PHASE_3_QUICK_START.md)** - 5-minute setup guide
- **[FX Signals Integration](guides/FX_SIGNALS_RL_INTEGRATION.md)** - Integrate FX patterns into RL pipeline
- **[RL Implementation Comparison](implementation/RL_IMPLEMENTATION_COMPARISON.md)** - "Fake RL" vs "Real RL" migration
  guide

### 🔍 Specialized Guides

- **[Backtesting Integration](guides/BACKTESTING_INTEGRATION_GUIDE.md)** - Backtesting.py integration
- **[Tracing Implementation](guides/TRACING_IMPLEMENTATION.md)** - OpenTelemetry setup and usage
- **[Tracing Quick Reference](guides/TRACING_QUICK_REF.md)** - Quick start for tracing

### 📊 Data Sources

- **[New Data Sources](NEW_DATA_SOURCES.md)** - FRED, Comtrade, ECB fundamental data integration
- **[Hyperparameter Tuning](guides/HYPERPARAMETER_TUNING_GUIDE.md)** - Bayesian optimization guide

### 📚 Research & Analysis

- **[Research Evaluation](research/RESEARCH_EVALUATION.md)** - Research paper concepts mapped to implementation
- **[Quantum Emulation](research/quantum_emulation.md)** - Experimental quantum-inspired features

### 📦 Archives

- **[Archived Documentation](archive/)** - Superseded or historical documentation

---

## Documentation Structure

```
docs/
├── README.md                                    ✅ You are here
├── CONFIGURATION_REFERENCE.md                   ✅ Current
├── NEW_DATA_SOURCES.md                          ✅ Current
├── TESTING_AND_LINTING.md                       ✅ Current
├── TESTING_VALIDATION_REPORT.md                 ✅ Current
│
├── api/
│   └── ARCHITECTURE_API_REFERENCE.md            ✅ Current
│
├── guides/
│   ├── BACKTESTING_INTEGRATION_GUIDE.md         ✅ Current
│   ├── FX_SIGNALS_RL_INTEGRATION.md             ✅ Current
│   ├── HYPERPARAMETER_TUNING_GUIDE.md           ✅ Current
│   ├── PHASE_3_QUICK_START.md                   ✅ Current
│   ├── TRACING_IMPLEMENTATION.md                ✅ Current
│   └── TRACING_QUICK_REF.md                     ✅ Current
│
├── implementation/
│   ├── PHASE_3_IMPLEMENTATION_SUMMARY.md        ✅ Current
│   └── RL_IMPLEMENTATION_COMPARISON.md          ✅ Current
│
├── research/
│   ├── RESEARCH_EVALUATION.md                   ✅ Current
│   └── quantum_emulation.md                     ✅ Current
│
└── archive/                                     📦 Historical docs
```

---

## Documentation by Use Case

### For New Users

1. Start with **[Architecture & API Reference](ARCHITECTURE_API_REFERENCE.md)** for system overview
2. Review **[Testing & Validation Report](TESTING_VALIDATION_REPORT.md)** to understand current status
3. Follow **[Phase 3 Quick Start](guides/PHASE_3_QUICK_START.md)** for hands-on setup

### For Researchers

1. Review **[Research Evaluation](research/RESEARCH_EVALUATION.md)** for concept mapping
2. Explore **[FX Signals Integration](guides/FX_SIGNALS_RL_INTEGRATION.md)** for feature engineering
3. Study **[Phase 3 Implementation](implementation/PHASE_3_IMPLEMENTATION_SUMMARY.md)** for production features

### For Developers

1. Reference **[Configuration Guide](CONFIGURATION_REFERENCE.md)** for production configs
2. Follow **[RL Implementation Comparison](implementation/RL_IMPLEMENTATION_COMPARISON.md)** for migration
3. Use **[Tracing Implementation](guides/TRACING_IMPLEMENTATION.md)** for observability

### For Operations

1. Review **[Testing & Validation Report](TESTING_VALIDATION_REPORT.md)** for system health
2. Reference **[Configuration Guide](CONFIGURATION_REFERENCE.md)** for deployment configs
3. Use **[Backtesting Integration](guides/BACKTESTING_INTEGRATION_GUIDE.md)** for validation

---

## Phase Overview

### Phase 1: Real RL Training Infrastructure ✅

- GDELT sentiment pipeline integration
- Real execution environment with market simulation
- Policy-gradient RL agent training

### Phase 2: Feature Engineering ✅

- Microstructure features (24 features)
- Regime detection (GMM-based 4-state classifier)
- Intrinsic time features (directional change events)
- FX patterns (23 features: sessions, S/R, ADX, price action)

### Phase 3: Production Enhancements ✅

- **Transaction costs**: Commission modeling, variable spreads
- **Position sizing**: Dynamic Kelly-criterion-inspired sizing
- **Risk management**: Stop-loss, take-profit, drawdown limits

---

## Key Features

### Trading Environment

- Simulated retail execution with realistic friction
- Transaction costs: Commission + spread + slippage
- Variable spreads during high volatility
- FIFO position tracking with realized P&L

### RL Training

- Policy-gradient agent (A2C-style updates)
- Real execution environment (not supervised learning)
- Dynamic position sizing (2% risk per trade)
- Multi-pair training support

### Feature Engineering

- 51+ features across multiple domains
- Microstructure: Order flow, execution quality
- Regime detection: Market state classification
- Intrinsic time: Event-driven features
- FX patterns: Sessions, support/resistance, trends

### Risk Management

- Configurable stop-loss and take-profit
- Portfolio-level drawdown monitoring
- Position limits per trading pair
- Cash constraint enforcement

---

## Testing Status

**All 25 tests passing** ✅

- **Phase 1 & 2**: 7/7 integration tests passing
- **Phase 3**: 16/16 validation tests passing
- **End-to-End**: 2/2 full pipeline tests passing

See **[Testing & Validation Report](TESTING_VALIDATION_REPORT.md)** for details.

---

## Configuration Quick Reference

### Conservative Training (High Safety)

```python
ExecutionConfig(
    commission_pct=0.0001,
    variable_spread=True,
    enable_stop_loss=True,
    stop_loss_pct=0.01,
    enable_take_profit=True,
    take_profit_pct=0.02,
    enable_drawdown_limit=True,
    max_drawdown_pct=0.10,
)
```

### Production-Like (Recommended)

```python
ExecutionConfig(
    commission_pct=0.0001,
    variable_spread=True,
    enable_stop_loss=False,  # Let agent learn
    enable_drawdown_limit=True,
    max_drawdown_pct=0.20,
)

ActionConverter(
    max_position=10.0,
    risk_per_trade=0.015,
    use_dynamic_sizing=True,
)
```

See **[Configuration Reference](CONFIGURATION_REFERENCE.md)** for complete options.

---

## Recent Updates

### 2025-12-29

- ✅ Completed Phase 3 implementation (transaction costs, position sizing, risk management)
- ✅ Comprehensive testing validation (25/25 tests passing)
- ✅ Documentation consolidation and reorganization
- ✅ Configuration reference guide created
- ✅ Phase 3 quick start guide added

---

## Contributing

When adding new documentation:

1. Place in appropriate subdirectory (`guides/`, `implementation/`, `research/`)
2. Update this README.md with new document link
3. Add cross-references to related documents
4. Include "Last Updated" date stamp
5. Follow existing formatting conventions

---

## Support & Resources

- **Issues**: https://github.com/your-org/sequence/issues
- **Discussions**: https://github.com/your-org/sequence/discussions
- **Documentation**: You're reading it! 📖

---

## License

[Your License Here]

---

**Document Status Legend**:

- ✅ **Current** - Up-to-date with latest implementation
- ⚠️ **Needs Update** - Scheduled for refresh
- 📦 **Archived** - Historical reference, superseded by newer docs
