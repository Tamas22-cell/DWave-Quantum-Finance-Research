é

# D-Wave Quantum Finance Research Engine

## Quantum-Inspired Portfolio Optimization & Multi-Agent Research

A Python-based research platform combining quantum-inspired
optimization, financial risk management, walk-forward backtesting,
and a rule-based multi-agent analysis architecture.

## Key Features

- 30 financial assets, including equities and cryptocurrencies
- QUBO portfolio optimization
- D-Wave Ocean SDK integration
- Local Simulated Annealing optimization
- Exact combinatorial benchmark
- Adaptive parameter tuning
- Walk-forward historical backtesting
- Portfolio volatility constraints
- Asset concentration limits
- Transaction cost modeling
- Three-agent research architecture
- FastAPI REST API
- Interactive financial dashboard

## Architecture

### Portfolio Optimization Engine

Uses a Quadratic Unconstrained Binary Optimization (QUBO)
formulation to select a portfolio of four assets.

The system compares simulated annealing results with an
exact constrained benchmark.

### Risk Management

The V7 engine implements:

- Maximum 35% estimated annualized training volatility
- Maximum one cryptocurrency
- Maximum one quantum technology stock
- Historical risk-adjusted performance analysis

### Multi-Agent Research

The V8 architecture includes:

1. Market Agent
2. Risk Agent
3. Portfolio Agent

A Research Coordinator combines their findings into
JSON and Markdown reports.

The agents are currently rule-based and do not use LLMs.

### API and Dashboard

The V9 application provides:

- `/health`
- `/api/v7`
- `/api/v8`
- `/api/equity`
- `/api/summary`
- `/docs`

The dashboard visualizes portfolio performance,
risk metrics, and multi-agent research findings.

## Technology Stack

- Python
- NumPy
- Pandas
- D-Wave Ocean SDK
- dimod
- Simulated Annealing
- FastAPI
- Uvicorn
- Chart.js

## Installation

Clone the repository and create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Dashboard

The dashboard reads saved V7 and V8 research outputs.

First, generate the required results using the
optimization and multi-agent scripts.

Then run:

```bash
python quantum_dashboard_v9.py
```

Open:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

## Research Results

In one historical V7 backtest:

| Metric | Simulated Annealing | Exact |
|---|---:|---:|
| Annualized return | 85.24% | 70.50% |
| Annualized volatility | 35.54% | 35.52% |
| Sharpe ratio | 1.9128 | 1.6801 |
| Maximum drawdown | -17.63% | -21.85% |
| Final equity | 2.5212 | 2.2264 |

These results reflect a historical simulation,
not expected future investment performance.

The exact solver optimizes the training objective,
not future realized returns.

## Important Limitations

- Optimization currently runs on classical hardware.
- No D-Wave quantum processing unit is used.
- Historical performance does not guarantee future results.
- Risk constraints are estimated from training data.
- Backtesting uses simplified transaction cost assumptions.
- Additional robustness testing is required before
  considering any live financial application.

## Development Roadmap

- Advanced automated testing
- Extended risk analytics
- LLM-powered agent integration
- Cloud API deployment
- Interactive public research dashboard
- Quantum hardware experimentation when access is available

## Disclaimer

This project is intended for educational and research purposes.

It does not constitute investment advice.
