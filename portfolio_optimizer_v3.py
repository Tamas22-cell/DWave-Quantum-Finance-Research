
import json
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import dimod
import numpy as np
import pandas as pd
import yfinance as yf
from dwave.samplers import SimulatedAnnealingSampler

# ==========================================
# D-WAVE QUANTUM PORTFOLIO OPTIMIZER V3
# Real Market Data + QUBO + Benchmark
# ==========================================

ASSETS = [
    "BTC-USD", "ETH-USD", "SOL-USD",
    "ADA-USD", "SUI20947-USD",
    "AVAX-USD", "LINK-USD", "XRP-USD",
    "DOGE-USD", "LTC-USD",
    "NVDA", "MSFT", "GOOGL", "AMZN",
    "META", "AMD", "TSLA", "PLTR",
    "IONQ", "QBTS", "RGTI", "QUBT",
    "IBM", "GOOG", "AAPL", "AVGO",
    "TSM", "MU", "INTC", "AMAT"
]

TARGET_ASSETS = 4
RISK_AVERSION = 0.5
PENALTY = 10.0
NUM_READS = 1000
NUM_SWEEPS = 1000

OUTPUT = Path(__file__).resolve().parent / "results_v3.json"

# ==========================================
# DOWNLOAD MARKET DATA
# ==========================================

print("=== D-WAVE PORTFOLIO OPTIMIZER V3 ===")
print("Downloading historical market data...")

raw = yf.download(
    ASSETS,
    period="1y",
    interval="1d",
    auto_adjust=True,
    progress=False,
    threads=True
)

if raw.empty:
    raise RuntimeError("Market data download failed.")

prices = raw["Close"].copy()

if isinstance(prices, pd.Series):
    prices = prices.to_frame()

# Remove assets with insufficient data
prices = prices.dropna(axis=1, thresh=150)

# Align observations across all assets
prices = prices.ffill().dropna()

# Crypto trades on weekends; stocks do not.
# Keep business days for a common observation calendar.
prices = prices.loc[prices.index.dayofweek < 5]

if prices.shape[0] < 100:
    raise RuntimeError("Insufficient aligned price history.")

assets = prices.columns.tolist()

print("Available assets:", len(assets))
print("Observations:", len(prices))

if len(assets) < TARGET_ASSETS:
    raise RuntimeError("Not enough assets for optimization.")

# ==========================================
# RETURNS AND COVARIANCE
# ==========================================

daily_returns = prices.pct_change(fill_method=None).dropna()

mu = daily_returns.mean().to_numpy() * 252
covariance = daily_returns.cov().to_numpy() * 252

n = len(assets)

# Equal weights among selected assets.
# Objective is mean-variance optimization.
risk_weight = RISK_AVERSION / (TARGET_ASSETS ** 2)
return_weight = 1.0 / TARGET_ASSETS

# ==========================================
# BUILD QUBO
# ==========================================

Q = {}

for i in range(n):
    Q[(i, i)] = (
        risk_weight * covariance[i, i]
        - return_weight * mu[i]
        + PENALTY * (1 - 2 * TARGET_ASSETS)
    )

    for j in range(i + 1, n):
        Q[(i, j)] = (
            2 * risk_weight * covariance[i, j]
            + 2 * PENALTY
        )

bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

# ==========================================
# SIMULATED ANNEALING
# ==========================================

print("\nRunning D-Wave Simulated Annealing...")

start = time.perf_counter()

sampler = SimulatedAnnealingSampler()

samples = sampler.sample(
    bqm,
    num_reads=NUM_READS,
    num_sweeps=NUM_SWEEPS,
    seed=42
)

sa_runtime = time.perf_counter() - start

valid_samples = [
    row for row in samples.data(["sample", "energy"])
    if sum(row.sample.values()) == TARGET_ASSETS
]

if not valid_samples:
    raise RuntimeError("No feasible SA solution found.")

best_sa = min(valid_samples, key=lambda row: row.energy)

sa_indices = [
    i for i in range(n)
    if best_sa.sample[i] == 1
]

sa_assets = [assets[i] for i in sa_indices]

# ==========================================
# EXACT CLASSICAL BENCHMARK
# Enumerate all C(n,4) combinations
# ==========================================

print("Running classical exact benchmark...")

start = time.perf_counter()

best_exact_energy = float("inf")
best_exact_combo = None

for combo in combinations(range(n), TARGET_ASSETS):
    idx = np.array(combo)

    # Feasible objective, excluding constant penalty
    portfolio_return = float(mu[idx].mean())

    portfolio_risk = float(
        covariance[np.ix_(idx, idx)].sum()
        / (TARGET_ASSETS ** 2)
    )

    objective = (
        RISK_AVERSION * portfolio_risk
        - portfolio_return
    )

    if objective < best_exact_energy:
        best_exact_energy = objective
        best_exact_combo = combo

exact_runtime = time.perf_counter() - start

exact_assets = [
    assets[i] for i in best_exact_combo
]

# Convert SA QUBO energy to objective without
# the constant -PENALTY * TARGET_ASSETS**2.
sa_objective = (
    float(best_sa.energy)
    + PENALTY * TARGET_ASSETS ** 2
)

energy_gap = sa_objective - best_exact_energy

# ==========================================
# PORTFOLIO METRICS
# ==========================================

selected_mu = mu[sa_indices]
selected_cov = covariance[
    np.ix_(sa_indices, sa_indices)
]

weights = np.full(TARGET_ASSETS, 1.0 / TARGET_ASSETS)

expected_return = float(weights @ selected_mu)

variance = float(
    weights @ selected_cov @ weights
)

volatility = float(np.sqrt(max(variance, 0.0)))

# Illustrative zero risk-free rate
sharpe_ratio = (
    expected_return / volatility
    if volatility > 0 else None
)

# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n=== SELECTED PORTFOLIO ===")
print("Assets:", sa_assets)
print("Selected count:", len(sa_assets))

print("\n=== RISK & RETURN ===")
print("Annual return:", round(expected_return * 100, 2), "%")
print("Annual volatility:", round(volatility * 100, 2), "%")
print(
    "Sharpe ratio:",
    round(sharpe_ratio, 4)
    if sharpe_ratio is not None else "N/A"
)

print("\n=== BENCHMARK ===")
print("SA objective:", round(sa_objective, 8))
print("Exact objective:", round(best_exact_energy, 8))
print("Exact assets:", exact_assets)
print("Objective gap:", round(energy_gap, 8))
print("SA runtime:", round(sa_runtime, 4), "seconds")
print("Exact runtime:", round(exact_runtime, 4), "seconds")

if abs(energy_gap) < 1e-7:
    status = "GLOBAL OPTIMUM FOUND"
else:
    status = "SUBOPTIMAL SOLUTION"

print("STATUS:", status)

# ==========================================
# EXPORT RESULTS TO JSON
# ==========================================

result = {
    "project": "D-Wave Quantum Portfolio Optimizer V3",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "solver": "D-Wave Simulated Annealing",
    "hardware": "Local CPU",
    "market_data": "Yahoo Finance historical prices",
    "assets_available": len(assets),
    "observations": len(prices),
    "selected_assets": sa_assets,
    "exact_selected_assets": exact_assets,
    "target_assets": TARGET_ASSETS,
    "annual_return": expected_return,
    "annual_volatility": volatility,
    "sharpe_ratio": sharpe_ratio,
    "sa_objective": sa_objective,
    "exact_objective": best_exact_energy,
    "objective_gap": energy_gap,
    "sa_runtime_seconds": sa_runtime,
    "exact_runtime_seconds": exact_runtime,
    "status": status
}

with OUTPUT.open("w", encoding="utf-8") as f:
    json.dump(result, f, indent=4, allow_nan=False)

print("\nJSON saved:", OUTPUT)
print("Optimization completed successfully.")
