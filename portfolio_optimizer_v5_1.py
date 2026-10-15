
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
# D-WAVE PORTFOLIO OPTIMIZER V5.1
# STABLE DATA + CACHE + AUTO-TUNING
# ==========================================

BASE = Path(__file__).resolve().parent
CACHE = BASE / "market_cache_v5_1.csv"
OUTPUT = BASE / "results_v5_1.json"

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
PENALTY = 10.0
MIN_OBSERVATIONS = 150
MAX_RETRIES = 3

READS_OPTIONS = [100, 500, 1000]
SWEEPS_OPTIONS = [500, 2000, 5000]
SEEDS = [42, 123, 777]

print("\n=== D-WAVE PORTFOLIO OPTIMIZER V5.1 ===")

# ==========================================
# MARKET DATA CACHE
# ==========================================

if CACHE.exists():
    print("Loading local market cache...")
    prices = pd.read_csv(
        CACHE,
        index_col=0,
        parse_dates=True
    )
else:
    prices = pd.DataFrame()

prices = prices.sort_index()
prices = prices.loc[~prices.index.duplicated(keep="last")]

def download_asset(ticker):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = yf.download(
                ticker,
                period="1y",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=20
            )

            if data.empty:
                raise ValueError("Empty market data")

            close = data["Close"]

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            close = pd.to_numeric(
                close, errors="coerce"
            ).dropna()

            if len(close) < MIN_OBSERVATIONS:
                raise ValueError(
                    f"Only {len(close)} observations"
                )

            close.name = ticker
            return close

        except Exception as error:
            print(
                f"  {ticker} attempt "
                f"{attempt}/{MAX_RETRIES}: {error}"
            )
            time.sleep(1)

    return None

failed = []

for index, ticker in enumerate(ASSETS, 1):
    print(f"[{index}/{len(ASSETS)}] {ticker}")

    if (
        ticker in prices.columns
        and prices[ticker].notna().sum()
        >= MIN_OBSERVATIONS
    ):
        print("  Loaded from cache")
        continue

    series = download_asset(ticker)

    if series is None:
        failed.append(ticker)
        continue

    # Align by calendar date to avoid timezone differences.
    series.index = pd.to_datetime(
        series.index, utc=True
    ).tz_convert(None).normalize()

    prices[ticker] = series

    # Save progress after every successful download.
    prices.to_csv(CACHE)

# ==========================================
# PREPARE COMMON TRADING CALENDAR
# ==========================================

available = [
    ticker for ticker in ASSETS
    if ticker in prices.columns
    and prices[ticker].notna().sum()
    >= MIN_OBSERVATIONS
]

if len(available) < TARGET:
    raise RuntimeError(
        "Not enough assets for optimization."
    )

prices = prices[available].copy()
prices = prices.loc[prices.index.dayofweek < 5]
prices = prices.sort_index().ffill().dropna()

if len(prices) < 100:
    raise RuntimeError(
        "Insufficient common price history."
    )

returns = prices.pct_change(fill_method=None).dropna()

mu = returns.mean().to_numpy() * 252
cov = returns.cov().to_numpy() * 252

n = len(available)

print("\nAvailable assets:", n)
print("Failed downloads:", failed)
print("Historical observations:", len(prices))

# ==========================================
# BUILD QUBO
# ==========================================

Q = {}

for i in range(n):
    Q[(i, i)] = (
        RISK_AVERSION * cov[i, i] / TARGET**2
        - mu[i] / TARGET
        + PENALTY * (1 - 2 * TARGET)
    )

    for j in range(i + 1, n):
        Q[(i, j)] = (
            2 * RISK_AVERSION
            * cov[i, j] / TARGET**2
            + 2 * PENALTY
        )

bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

def portfolio_objective(indices):
    idx = np.asarray(indices, dtype=int)

    annual_return = float(mu[idx].mean())

    variance = float(
        cov[np.ix_(idx, idx)].sum()
        / TARGET**2
    )

    return RISK_AVERSION * variance - annual_return

# ==========================================
# EXACT BENCHMARK
# ==========================================

print("\nRunning Exact Benchmark...")

start = time.perf_counter()
exact_combo = None
exact_objective = float("inf")

for combo in combinations(range(n), TARGET):
    objective = portfolio_objective(combo)

    if objective < exact_objective:
        exact_objective = objective
        exact_combo = combo

exact_runtime = time.perf_counter() - start

print("Exact assets:", [available[i] for i in exact_combo])
print("Exact objective:", round(exact_objective, 8))
print("Exact runtime:", round(exact_runtime, 4), "s")

# ==========================================
# SIMULATED ANNEALING AUTO-TUNING
# ==========================================

sampler = SimulatedAnnealingSampler()

best_global = None
experiments = []

print("\nRunning Auto-Tuning (27 experiments)...")

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

            feasible = [
                row for row in samples.data(
                    ["sample", "energy"]
                )
                if sum(row.sample.values()) == TARGET
            ]

            if not feasible:
                print(
                    f"Reads={reads} Sweeps={sweeps} "
                    f"Seed={seed}: NO FEASIBLE SOLUTION"
                )
                continue

            best = min(
                feasible,
                key=lambda row: row.energy
            )

            indices = [
                i for i in range(n)
                if best.sample[i] == 1
            ]

            objective = portfolio_objective(indices)
            gap = objective - exact_objective

            record = {
                "reads": reads,
                "sweeps": sweeps,
                "seed": seed,
                "assets": [
                    available[i] for i in indices
                ],
                "objective": objective,
                "gap": gap,
                "runtime": runtime
            }

            experiments.append(record)

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

if best_global is None:
    raise RuntimeError(
        "Auto-tuning found no feasible solution."
    )

# ==========================================
# FINAL VALIDATION
# ==========================================

status = (
    "GLOBAL OPTIMUM FOUND"
    if abs(best_global["gap"]) < 1e-7
    else "SUBOPTIMAL SOLUTION"
)

print("\n=== BEST AUTO-TUNING RESULT ===")
print("Selected assets:", best_global["assets"])
print("Objective:", round(best_global["objective"], 8))
print("Exact objective:", round(exact_objective, 8))
print("Objective gap:", round(best_global["gap"], 8))
print("Reads:", best_global["reads"])
print("Sweeps:", best_global["sweeps"])
print("Seed:", best_global["seed"])
print("STATUS:", status)

# ==========================================
# EXPORT JSON
# ==========================================

result = {
    "project": "D-Wave Portfolio Optimizer V5.1",
    "solver": "Local Simulated Annealing",
    "assets_available": n,
    "failed_downloads": failed,
    "historical_observations": len(prices),
    "exact_assets": [
        available[i] for i in exact_combo
    ],
    "exact_objective": exact_objective,
    "exact_runtime": exact_runtime,
    "best_auto_tuning": best_global,
    "experiments": experiments,
    "status": status
}

with OUTPUT.open("w", encoding="utf-8") as file:
    json.dump(result, file, indent=4, allow_nan=False)

print("\nJSON saved:", OUTPUT)
print("Optimization completed successfully.")
