
import json
import time
from itertools import combinations
from pathlib import Path

import dimod
import numpy as np
import pandas as pd
import yfinance as yf
from dwave.samplers import SimulatedAnnealingSampler

# ============================================
# D-WAVE PORTFOLIO OPTIMIZER V6
# QUANTUM FINANCE RESEARCH ENGINE
# ============================================

BASE = Path(__file__).resolve().parent
CACHE = BASE / "market_cache_v6.csv"
OUTPUT = BASE / "results_v6.json"

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

TRAIN_DAYS = 126
TEST_DAYS = 21

NUM_READS = 1000
NUM_SWEEPS = 2000
SEED = 123

# ============================================
# DATA DOWNLOAD AND CACHE
# ============================================

def download_asset(ticker):
    for attempt in range(1, 4):
        try:
            data = yf.download(
                ticker,
                period="2y",
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
                raise ValueError("Insufficient history")

            close.index = pd.to_datetime(
                close.index, utc=True
            ).tz_convert(None).normalize()

            close = close.loc[
                ~close.index.duplicated(keep="last")
            ]

            return close.rename(ticker)

        except Exception as error:
            print(
                f"{ticker}: attempt {attempt}/3 "
                f"failed: {error}"
            )
            time.sleep(1)

    return None


def load_market_data():
    if CACHE.exists():
        print("Loading market data from cache...")
        prices = pd.read_csv(
            CACHE,
            index_col=0,
            parse_dates=True
        )
    else:
        prices = pd.DataFrame()

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    failed = []

    for i, ticker in enumerate(ASSETS, 1):
        print(f"[{i}/{len(ASSETS)}] {ticker}")

        if (
            ticker in prices.columns
            and prices[ticker].notna().sum()
            >= MIN_OBSERVATIONS
        ):
            print("  Cached")
            continue

        series = download_asset(ticker)

        if series is None:
            failed.append(ticker)
            continue

        prices[ticker] = series
        prices.to_csv(CACHE)

    available = [
        ticker for ticker in ASSETS
        if ticker in prices.columns
        and prices[ticker].notna().sum()
        >= MIN_OBSERVATIONS
    ]

    prices = prices[available].sort_index()
    prices = prices.loc[prices.index.dayofweek < 5]
    prices = prices.ffill().dropna()

    if len(prices) < TRAIN_DAYS + TEST_DAYS + 1:
        raise RuntimeError(
            "Insufficient common historical observations."
        )

    return prices, failed


# ============================================
# QUBO MODEL
# ============================================

def objective(indices, mu, cov):
    idx = np.asarray(indices, dtype=int)

    annual_return = float(mu[idx].mean())
    variance = float(
        cov[np.ix_(idx, idx)].sum() / TARGET**2
    )

    return RISK_AVERSION * variance - annual_return


def build_bqm(mu, cov):
    n = len(mu)
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

    return dimod.BinaryQuadraticModel.from_qubo(Q)


# ============================================
# SOLVER BENCHMARK
# ============================================

def optimize_portfolio(train_returns, assets):
    mu = train_returns.mean().to_numpy() * 252
    cov = train_returns.cov().to_numpy() * 252
    n = len(assets)

    bqm = build_bqm(mu, cov)
    sampler = SimulatedAnnealingSampler()

    start = time.perf_counter()

    samples = sampler.sample(
        bqm,
        num_reads=NUM_READS,
        num_sweeps=NUM_SWEEPS,
        seed=SEED
    )

    feasible = [
        row for row in samples.data(
            ["sample", "energy"]
        )
        if sum(row.sample.values()) == TARGET
    ]

    if not feasible:
        raise RuntimeError("No feasible SA solution.")

    best_sa = min(feasible, key=lambda x: x.energy)
    sa_indices = [
        i for i in range(n)
        if best_sa.sample[i] == 1
    ]

    sa_runtime = time.perf_counter() - start

    start = time.perf_counter()

    exact_indices = min(
        combinations(range(n), TARGET),
        key=lambda combo: objective(combo, mu, cov)
    )

    exact_runtime = time.perf_counter() - start

    sa_objective = objective(sa_indices, mu, cov)
    exact_objective = objective(exact_indices, mu, cov)

    return {
        "sa_indices": sa_indices,
        "exact_indices": list(exact_indices),
        "sa_assets": [assets[i] for i in sa_indices],
        "exact_assets": [assets[i] for i in exact_indices],
        "sa_objective": sa_objective,
        "exact_objective": exact_objective,
        "objective_gap": max(
            0.0, sa_objective - exact_objective
        ),
        "sa_runtime": sa_runtime,
        "exact_runtime": exact_runtime
    }


# ============================================
# RISK ANALYTICS
# ============================================

def calculate_metrics(daily_returns):
    r = np.asarray(daily_returns, dtype=float)

    if len(r) < 2:
        return {}

    annual_return = float(
        np.prod(1 + r) ** (252 / len(r)) - 1
    )

    volatility = float(
        np.std(r, ddof=1) * np.sqrt(252)
    )

    sharpe = (
        annual_return / volatility
        if volatility > 0 else 0.0
    )

    downside = np.minimum(r, 0)
    downside_dev = float(
        np.sqrt(np.mean(downside**2) * 252)
    )

    sortino = (
        annual_return / downside_dev
        if downside_dev > 0 else 0.0
    )

    equity = np.cumprod(1 + r)
    peaks = np.maximum.accumulate(
        np.concatenate(([1.0], equity))
    )[1:]

    drawdowns = equity / peaks - 1
    max_drawdown = float(np.min(drawdowns))

    return {
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown
    }


# ============================================
# WALK-FORWARD BACKTEST
# ============================================

def run_backtest(prices):
    returns = prices.pct_change(
        fill_method=None
    ).dropna()

    assets = returns.columns.tolist()
    all_returns = []
    all_exact_returns = []
    windows = []

    for start in range(
        TRAIN_DAYS,
        len(returns) - TEST_DAYS + 1,
        TEST_DAYS
    ):
        train = returns.iloc[
            start - TRAIN_DAYS:start
        ]

        test = returns.iloc[
            start:start + TEST_DAYS
        ]

        print(
            f"\nWindow {len(windows) + 1}: "
            f"{test.index[0].date()} -> "
            f"{test.index[-1].date()}"
        )

        result = optimize_portfolio(train, assets)

        sa_test = test.iloc[
            :, result["sa_indices"]
        ].mean(axis=1)

        exact_test = test.iloc[
            :, result["exact_indices"]
        ].mean(axis=1)

        all_returns.extend(sa_test.tolist())
        all_exact_returns.extend(exact_test.tolist())

        result["test_start"] = str(
            test.index[0].date()
        )
        result["test_end"] = str(
            test.index[-1].date()
        )

        windows.append(result)

        print("SA:", result["sa_assets"])
        print("Exact:", result["exact_assets"])
        print(
            "Objective gap:",
            round(result["objective_gap"], 8)
        )

    return {
        "windows": windows,
        "sa_metrics": calculate_metrics(all_returns),
        "exact_metrics": calculate_metrics(
            all_exact_returns
        ),
        "test_observations": len(all_returns)
    }


# ============================================
# MAIN
# ============================================

def main():
    print("\n=== D-WAVE PORTFOLIO OPTIMIZER V6 ===")
    print("Quantum Finance Research Engine")
    print("Mode: Local Simulated Annealing\n")

    prices, failed = load_market_data()

    print("\nAvailable assets:", len(prices.columns))
    print("Failed downloads:", failed)
    print("Historical observations:", len(prices))

    print("\nRunning Walk-Forward Backtesting...")

    results = run_backtest(prices)

    print("\n=== FINAL BACKTEST RESULTS ===")

    for name, metrics in [
        ("SIMULATED ANNEALING", results["sa_metrics"]),
        ("EXACT BENCHMARK", results["exact_metrics"])
    ]:
        print(f"\n{name}")

        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

    output = {
        "project": "D-Wave Portfolio Optimizer V6",
        "solver": "Local Simulated Annealing",
        "assets": prices.columns.tolist(),
        "failed_downloads": failed,
        "historical_observations": len(prices),
        **results
    }

    with OUTPUT.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            output,
            file,
            indent=4,
            allow_nan=False
        )

    print("\nJSON saved:", OUTPUT)
    print("V6 completed successfully.")


if __name__ == "__main__":
    main()
