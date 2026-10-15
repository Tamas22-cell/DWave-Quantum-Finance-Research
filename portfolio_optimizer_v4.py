
import json
import time
import warnings
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import dimod
import numpy as np
import pandas as pd
import yfinance as yf
from dwave.samplers import SimulatedAnnealingSampler

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================
# D-WAVE QUANTUM PORTFOLIO OPTIMIZER V4
# ============================================

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
NUM_READS = 1000
NUM_SWEEPS = 1000
MAX_RETRIES = 3

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "results_v4.json"

# ============================================
# MARKET DATA DOWNLOAD
# ============================================

def download_prices(tickers):
    collected = {}
    failed = []

    for number, ticker in enumerate(tickers, 1):
        print(f"[{number}/{len(tickers)}] {ticker}")

        success = False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                data = yf.download(
                    ticker,
                    period="1y",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    timeout=15
                )

                if data.empty:
                    raise ValueError("Empty market data")

                close = data["Close"]

                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]

                close = close.dropna()

                if len(close) < 150:
                    raise ValueError(
                        "Insufficient historical data"
                    )

                collected[ticker] = close
                success = True
                break

            except Exception as error:
                print(
                    f"Attempt {attempt}/{MAX_RETRIES}: "
                    f"{str(error)[:100]}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(2)

        if not success:
            failed.append(ticker)

    if not collected:
        raise RuntimeError(
            "No market data available. Check connection."
        )

    prices = pd.DataFrame(collected)
    prices.index = pd.to_datetime(prices.index, utc=True)
    prices = prices.sort_index()

    # Common trading calendar: weekdays only
    prices = prices.loc[prices.index.dayofweek < 5]

    # No backfilling from future observations
    prices = prices.ffill().dropna()

    if len(prices) < 100:
        raise RuntimeError(
            "Insufficient aligned historical observations."
        )

    return prices, failed


# ============================================
# PORTFOLIO METRICS
# ============================================

def portfolio_metrics(indices, mu, covariance):
    weights = np.full(len(indices), 1 / len(indices))

    selected_mu = mu[list(indices)]
    selected_cov = covariance[
        np.ix_(indices, indices)
    ]

    annual_return = float(weights @ selected_mu)

    variance = float(
        weights @ selected_cov @ weights
    )

    volatility = float(np.sqrt(max(variance, 0)))

    sharpe = (
        annual_return / volatility
        if volatility > 0 else None
    )

    return {
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe_ratio": sharpe
    }


# ============================================
# START
# ============================================

print("\n=== D-WAVE PORTFOLIO OPTIMIZER V4 ===")
print("Downloading real historical market data...")

prices, failed_assets = download_prices(ASSETS)

assets = prices.columns.tolist()
n = len(assets)

if n < TARGET:
    raise RuntimeError(
        "Not enough assets for portfolio optimization."
    )

print("\nAvailable assets:", n)
print("Failed downloads:", failed_assets)
print("Historical observations:", len(prices))

daily_returns = prices.pct_change(
    fill_method=None
).dropna()

if len(daily_returns) < 90:
    raise RuntimeError("Insufficient daily returns.")

mu = daily_returns.mean().to_numpy() * 252
covariance = daily_returns.cov().to_numpy() * 252

# ============================================
# QUBO MODEL
# ============================================

Q = {}

for i in range(n):
    Q[(i, i)] = (
        RISK_AVERSION * covariance[i, i] / TARGET**2
        - mu[i] / TARGET
        + PENALTY * (1 - 2 * TARGET)
    )

    for j in range(i + 1, n):
        Q[(i, j)] = (
            2 * RISK_AVERSION
            * covariance[i, j] / TARGET**2
            + 2 * PENALTY
        )

bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

# ============================================
# SOLVER 1: SIMULATED ANNEALING
# ============================================

print("\nRunning Simulated Annealing...")

start = time.perf_counter()

sa_samples = SimulatedAnnealingSampler().sample(
    bqm,
    num_reads=NUM_READS,
    num_sweeps=NUM_SWEEPS,
    seed=42
)

sa_runtime = time.perf_counter() - start

valid_sa = [
    row for row in sa_samples.data(
        ["sample", "energy"]
    )
    if sum(row.sample.values()) == TARGET
]

if not valid_sa:
    raise RuntimeError("No feasible SA portfolio found.")

best_sa = min(valid_sa, key=lambda r: r.energy)

sa_indices = tuple(
    i for i in range(n)
    if best_sa.sample[i] == 1
)

sa_objective = float(
    best_sa.energy + PENALTY * TARGET**2
)

# ============================================
# SOLVER 2: GREEDY BASELINE
# ============================================

print("Running Greedy Baseline...")

start = time.perf_counter()

# Rank by standalone mean-variance score
standalone_scores = (
    RISK_AVERSION * np.diag(covariance) / TARGET**2
    - mu / TARGET
)

greedy_indices = tuple(
    sorted(np.argsort(standalone_scores)[:TARGET])
)

greedy_metrics = portfolio_metrics(
    greedy_indices, mu, covariance
)

greedy_objective = (
    RISK_AVERSION
    * greedy_metrics["annual_volatility"]**2
    - greedy_metrics["annual_return"]
)

greedy_runtime = time.perf_counter() - start

# ============================================
# SOLVER 3: EXACT COMBINATORIAL BENCHMARK
# ============================================

print("Running Exact Benchmark...")

start = time.perf_counter()

best_exact_objective = float("inf")
best_exact_indices = None

for combo in combinations(range(n), TARGET):
    idx = np.array(combo)

    expected_return = float(mu[idx].mean())

    variance = float(
        covariance[np.ix_(idx, idx)].sum()
        / TARGET**2
    )

    objective = (
        RISK_AVERSION * variance
        - expected_return
    )

    if objective < best_exact_objective:
        best_exact_objective = objective
        best_exact_indices = combo

exact_runtime = time.perf_counter() - start

# ============================================
# RESULTS
# ============================================

sa_metrics = portfolio_metrics(
    sa_indices, mu, covariance
)

exact_metrics = portfolio_metrics(
    best_exact_indices, mu, covariance
)

gap = sa_objective - best_exact_objective

def names(indices):
    return [assets[i] for i in indices]


print("\n=== SOLVER COMPARISON ===")

print("\nSIMULATED ANNEALING")
print("Assets:", names(sa_indices))
print("Objective:", round(sa_objective, 8))
print("Runtime:", round(sa_runtime, 4), "s")

print("\nGREEDY BASELINE")
print("Assets:", names(greedy_indices))
print("Objective:", round(greedy_objective, 8))
print("Runtime:", round(greedy_runtime, 4), "s")

print("\nEXACT BENCHMARK")
print("Assets:", names(best_exact_indices))
print("Objective:", round(best_exact_objective, 8))
print("Runtime:", round(exact_runtime, 4), "s")

print("\n=== PORTFOLIO RISK ===")
print(
    "Annual return:",
    round(sa_metrics["annual_return"] * 100, 2),
    "%"
)
print(
    "Annual volatility:",
    round(sa_metrics["annual_volatility"] * 100, 2),
    "%"
)
print(
    "Sharpe Ratio:",
    round(sa_metrics["sharpe_ratio"], 4)
    if sa_metrics["sharpe_ratio"] is not None
    else "N/A"
)

print("\n=== VALIDATION ===")
print("Objective gap:", round(gap, 8))

status = (
    "GLOBAL OPTIMUM FOUND"
    if abs(gap) < 1e-7
    else "SUBOPTIMAL SOLUTION"
)

print("STATUS:", status)

# ============================================
# JSON EXPORT
# ============================================

result = {
    "project": "D-Wave Portfolio Optimizer V4",
    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "hardware": "Local CPU",
    "data_source": "Yahoo Finance",
    "available_assets": n,
    "failed_assets": failed_assets,
    "observations": len(prices),
    "selected_assets": names(sa_indices),
    "portfolio_metrics": sa_metrics,
    "solvers": {
        "simulated_annealing": {
            "assets": names(sa_indices),
            "objective": sa_objective,
            "runtime": sa_runtime
        },
        "greedy": {
            "assets": names(greedy_indices),
            "objective": greedy_objective,
            "runtime": greedy_runtime
        },
        "exact": {
            "assets": names(best_exact_indices),
            "objective": best_exact_objective,
            "runtime": exact_runtime
        }
    },
    "objective_gap": gap,
    "status": status
}

with OUTPUT.open("w", encoding="utf-8") as file:
    json.dump(
        result,
        file,
        indent=4,
        allow_nan=False
    )

print("\nJSON saved:", OUTPUT)
print("Optimization completed successfully.")
