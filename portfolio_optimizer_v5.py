
import json
import time
from itertools import combinations
from pathlib import Path

import dimod
import numpy as np
import pandas as pd
import yfinance as yf
from dwave.samplers import SimulatedAnnealingSampler

# ==========================================
# D-WAVE PORTFOLIO OPTIMIZER V5
# AUTO-TUNING + EXACT BENCHMARK
# ==========================================

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "results_v5.json"

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

TARGET = 4
RISK_AVERSION = 0.5

# Parameter search
READS_OPTIONS = [100, 500, 1000]
SWEEPS_OPTIONS = [500, 2000, 5000]
SEEDS = [42, 123, 777]

# ==========================================
# DOWNLOAD MARKET DATA
# ==========================================

print("=== D-WAVE PORTFOLIO OPTIMIZER V5 ===")
print("Downloading market data...")

raw = yf.download(
    ASSETS,
    period="1y",
    interval="1d",
    auto_adjust=True,
    progress=False,
    threads=False
)

if raw.empty:
    raise RuntimeError("Market data download failed.")

prices = raw["Close"].copy()
prices = prices.dropna(axis=1, thresh=150)
prices = prices.ffill().dropna()
prices = prices.loc[prices.index.dayofweek < 5]

if len(prices) < 100:
    raise RuntimeError("Insufficient price observations.")

assets = prices.columns.tolist()
n = len(assets)

if n < TARGET:
    raise RuntimeError("Not enough available assets.")

returns = prices.pct_change(fill_method=None).dropna()

mu = returns.mean().to_numpy() * 252
covariance = returns.cov().to_numpy() * 252

print("Available assets:", n)
print("Observations:", len(prices))

# ==========================================
# BUILD QUBO
# ==========================================

penalty = 10.0

Q = {}

for i in range(n):
    Q[(i, i)] = (
        RISK_AVERSION * covariance[i, i] / TARGET**2
        - mu[i] / TARGET
        + penalty * (1 - 2 * TARGET)
    )

    for j in range(i + 1, n):
        Q[(i, j)] = (
            2 * RISK_AVERSION
            * covariance[i, j] / TARGET**2
            + 2 * penalty
        )

bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

# ==========================================
# EXACT BENCHMARK
# ==========================================

print("\nRunning Exact Benchmark...")

start = time.perf_counter()

exact_energy = float("inf")
exact_combo = None

for combo in combinations(range(n), TARGET):
    idx = np.array(combo)

    annual_return = float(mu[idx].mean())

    variance = float(
        covariance[np.ix_(idx, idx)].sum()
        / TARGET**2
    )

    objective = (
        RISK_AVERSION * variance
        - annual_return
    )

    if objective < exact_energy:
        exact_energy = objective
        exact_combo = combo

exact_runtime = time.perf_counter() - start

print("Exact assets:", [assets[i] for i in exact_combo])
print("Exact objective:", round(exact_energy, 8))

# ==========================================
# AUTO-TUNING
# ==========================================

sampler = SimulatedAnnealingSampler()

results = []
best_global = None

for reads in READS_OPTIONS:
    for sweeps in SWEEPS_OPTIONS:
        for seed in SEEDS:

            start = time.perf_counter()

            samples = sampler.sample(
                bqm,
                num_reads=reads,
                num_sweeps=sweeps,
                seed=seed
            )

            runtime = time.perf_counter() - start

            valid = [
                row for row in samples.data(
                    ["sample", "energy"]
                )
                if sum(row.sample.values()) == TARGET
            ]

            if not valid:
                print(
                    f"Reads={reads}, Sweeps={sweeps}, "
                    f"Seed={seed}: NO FEASIBLE SOLUTION"
                )
                continue

            best = min(valid, key=lambda r: r.energy)

            indices = [
                i for i in range(n)
                if best.sample[i] == 1
            ]

            objective = float(
                best.energy + penalty * TARGET**2
            )

            gap = objective - exact_energy

            record = {
                "reads": reads,
                "sweeps": sweeps,
                "seed": seed,
                "assets": [assets[i] for i in indices],
                "objective": objective,
                "gap": gap,
                "runtime": runtime
            }

            results.append(record)

            print(
                f"Reads={reads:4d} "
                f"Sweeps={sweeps:5d} "
                f"Seed={seed:3d} "
                f"Gap={gap:.8f} "
                f"Time={runtime:.3f}s"
            )

            if (
                best_global is None
                or objective < best_global["objective"]
            ):
                best_global = record

# ==========================================
# FINAL RESULTS
# ==========================================

if best_global is None:
    raise RuntimeError("No feasible solution found.")

print("\n=== BEST AUTO-TUNING RESULT ===")
print("Selected assets:", best_global["assets"])
print("Objective:", round(best_global["objective"], 8))
print("Exact objective:", round(exact_energy, 8))
print("Objective gap:", round(best_global["gap"], 8))
print("Reads:", best_global["reads"])
print("Sweeps:", best_global["sweeps"])
print("Seed:", best_global["seed"])

status = (
    "GLOBAL OPTIMUM FOUND"
    if abs(best_global["gap"]) < 1e-7
    else "SUBOPTIMAL SOLUTION"
)

print("STATUS:", status)

# ==========================================
# JSON EXPORT
# ==========================================

output = {
    "project": "D-Wave Portfolio Optimizer V5",
    "hardware": "Local CPU",
    "assets_available": n,
    "exact_assets": [assets[i] for i in exact_combo],
    "exact_objective": exact_energy,
    "exact_runtime": exact_runtime,
    "best_auto_tuning": best_global,
    "experiments": results,
    "status": status
}

with OUTPUT.open("w", encoding="utf-8") as file:
    json.dump(output, file, indent=4, allow_nan=False)

print("\nJSON saved:", OUTPUT)
print("Auto-tuning completed successfully.")
