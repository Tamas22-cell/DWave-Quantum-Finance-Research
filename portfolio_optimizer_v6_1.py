
import json
import time
from itertools import combinations
from pathlib import Path

import dimod
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from dwave.samplers import SimulatedAnnealingSampler

# ============================================
# D-WAVE PORTFOLIO OPTIMIZER V6.1
# ADAPTIVE AUTO-TUNING RESEARCH ENGINE
# ============================================

BASE = Path(__file__).resolve().parent

CACHE = BASE / "market_cache_v6.csv"
OUTPUT = BASE / "results_v6_1.json"
EQUITY_CSV = BASE / "equity_curve_v6_1.csv"
EQUITY_PNG = BASE / "equity_curve_v6_1.png"

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

TRAIN_DAYS = 126
TEST_DAYS = 21
MIN_OBSERVATIONS = 150

TRANSACTION_COST = 0.001

READS_OPTIONS = [250, 500, 1000]
SWEEPS_OPTIONS = [500, 1000, 2000]
SEEDS = [42, 123, 777]

# ============================================
# MARKET DATA
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

            return close.rename(ticker)

        except Exception as error:
            print(
                f"{ticker}: attempt {attempt}/3: {error}"
            )
            time.sleep(1)

    return None


def load_market_data():
    if CACHE.exists():
        print("Loading V6 market cache...")
        prices = pd.read_csv(
            CACHE,
            index_col=0,
            parse_dates=True
        )
    else:
        prices = pd.DataFrame()

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

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

        if series is not None:
            prices[ticker] = series
            prices.to_csv(CACHE)

    available = [
        ticker for ticker in ASSETS
        if ticker in prices.columns
        and prices[ticker].notna().sum()
        >= MIN_OBSERVATIONS
    ]

    if len(available) < TARGET:
        raise RuntimeError("Not enough assets.")

    prices = prices[available].sort_index()
    prices = prices.loc[prices.index.dayofweek < 5]
    prices = prices.ffill().dropna()

    if len(prices) < TRAIN_DAYS + TEST_DAYS + 1:
        raise RuntimeError("Insufficient common history.")

    return prices


# ============================================
# OBJECTIVE AND QUBO
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
# ADAPTIVE SOLVER
# ============================================

def optimize_portfolio(train_returns, assets):
    mu = train_returns.mean().to_numpy() * 252
    cov = train_returns.cov().to_numpy() * 252
    n = len(assets)

    bqm = build_bqm(mu, cov)
    sampler = SimulatedAnnealingSampler()

    best_sa = None
    experiments = []

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

                elapsed = time.perf_counter() - start

                feasible = [
                    row for row in samples.data(
                        ["sample", "energy"]
                    )
                    if sum(row.sample.values()) == TARGET
                ]

                if not feasible:
                    continue

                candidate = min(
                    feasible, key=lambda row: row.energy
                )

                indices = [
                    i for i in range(n)
                    if candidate.sample[i] == 1
                ]

                score = objective(indices, mu, cov)

                record = {
                    "reads": reads,
                    "sweeps": sweeps,
                    "seed": seed,
                    "objective": score,
                    "runtime": elapsed
                }
                experiments.append(record)

                if (
                    best_sa is None
                    or score < best_sa["objective"]
                ):
                    best_sa = {
                        **record,
                        "indices": indices
                    }

    if best_sa is None:
        raise RuntimeError("No feasible SA solution.")

    start = time.perf_counter()

    exact_indices = min(
        combinations(range(n), TARGET),
        key=lambda combo: objective(combo, mu, cov)
    )

    exact_runtime = time.perf_counter() - start
    exact_score = objective(exact_indices, mu, cov)

    return {
        "sa_indices": best_sa["indices"],
        "exact_indices": list(exact_indices),
        "sa_assets": [
            assets[i] for i in best_sa["indices"]
        ],
        "exact_assets": [
            assets[i] for i in exact_indices
        ],
        "sa_objective": best_sa["objective"],
        "exact_objective": exact_score,
        "objective_gap": max(
            0.0, best_sa["objective"] - exact_score
        ),
        "best_parameters": {
            "reads": best_sa["reads"],
            "sweeps": best_sa["sweeps"],
            "seed": best_sa["seed"]
        },
        "experiments": experiments,
        "exact_runtime": exact_runtime
    }


# ============================================
# PERFORMANCE METRICS
# ============================================

def calculate_metrics(daily_returns):
    r = np.asarray(daily_returns, dtype=float)

    if len(r) < 2:
        return {}

    equity = np.cumprod(1 + r)
    annual_return = float(
        equity[-1] ** (252 / len(r)) - 1
    )

    volatility = float(
        np.std(r, ddof=1) * np.sqrt(252)
    )

    sharpe = (
        float(np.mean(r) * 252 / volatility)
        if volatility > 0 else 0.0
    )

    downside = np.minimum(r, 0)
    downside_dev = float(
        np.sqrt(np.mean(downside**2) * 252)
    )

    sortino = (
        float(np.mean(r) * 252 / downside_dev)
        if downside_dev > 0 else 0.0
    )

    running_peak = np.maximum.accumulate(
        np.concatenate(([1.0], equity))
    )[1:]

    max_drawdown = float(
        np.min(equity / running_peak - 1)
    )

    return {
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "final_equity": float(equity[-1])
    }


# ============================================
# WALK-FORWARD BACKTEST
# ============================================

def run_backtest(prices):
    returns = prices.pct_change(
        fill_method=None
    ).dropna()

    assets = returns.columns.tolist()
    sa_returns = []
    exact_returns = []
    dates = []
    windows = []

    previous_sa = set()
    previous_exact = set()

    for end in range(
        TRAIN_DAYS,
        len(returns) - TEST_DAYS + 1,
        TEST_DAYS
    ):
        train = returns.iloc[
            end - TRAIN_DAYS:end
        ]

        test = returns.iloc[
            end:end + TEST_DAYS
        ]

        result = optimize_portfolio(train, assets)

        current_sa = set(result["sa_indices"])
        current_exact = set(result["exact_indices"])

        # Equal-weight portfolios; turnover includes sells and buys.
        sa_turnover = (
            len(current_sa.symmetric_difference(previous_sa))
            / (2 * TARGET)
        )

        exact_turnover = (
            len(current_exact.symmetric_difference(previous_exact))
            / (2 * TARGET)
        )

        sa_daily = test.iloc[
            :, result["sa_indices"]
        ].mean(axis=1).copy()

        exact_daily = test.iloc[
            :, result["exact_indices"]
        ].mean(axis=1).copy()

        # Charge trading costs at the beginning of each window.
        sa_daily.iloc[0] -= (
            sa_turnover * TRANSACTION_COST
        )
        exact_daily.iloc[0] -= (
            exact_turnover * TRANSACTION_COST
        )

        sa_returns.extend(sa_daily.tolist())
        exact_returns.extend(exact_daily.tolist())
        dates.extend(test.index.tolist())

        previous_sa = current_sa
        previous_exact = current_exact

        result["test_start"] = str(
            test.index[0].date()
        )
        result["test_end"] = str(
            test.index[-1].date()
        )
        result["sa_turnover"] = sa_turnover
        result["exact_turnover"] = exact_turnover

        windows.append(result)

        print(
            f"Window {len(windows):02d} | "
            f"Gap={result['objective_gap']:.8f} | "
            f"SA={result['sa_assets']}"
        )

    curve = pd.DataFrame(
        {
            "SA": np.cumprod(1 + np.array(sa_returns)),
            "Exact": np.cumprod(
                1 + np.array(exact_returns)
            )
        },
        index=pd.DatetimeIndex(dates)
    )

    curve.index.name = "Date"
    curve.to_csv(EQUITY_CSV)

    plt.figure(figsize=(12, 6))
    plt.plot(
        curve.index, curve["SA"],
        label="Simulated Annealing"
    )
    plt.plot(
        curve.index, curve["Exact"],
        label="Exact Benchmark"
    )
    plt.title("D-Wave V6.1 – Walk-Forward Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value (start = 1.0)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(EQUITY_PNG, dpi=160)
    plt.close()

    return {
        "windows": windows,
        "sa_metrics": calculate_metrics(sa_returns),
        "exact_metrics": calculate_metrics(exact_returns),
        "test_observations": len(sa_returns)
    }


# ============================================
# MAIN
# ============================================

def main():
    print("\n=== D-WAVE PORTFOLIO OPTIMIZER V6.1 ===")
    print("Adaptive Research Engine")
    print("Solver: Local Simulated Annealing\n")

    prices = load_market_data()

    print("\nAvailable assets:", len(prices.columns))
    print("Historical observations:", len(prices))

    print("\nRunning Adaptive Walk-Forward Backtesting...")

    results = run_backtest(prices)

    print("\n=== FINAL RESULTS ===")

    for label, metrics in [
        ("SIMULATED ANNEALING", results["sa_metrics"]),
        ("EXACT BENCHMARK", results["exact_metrics"])
    ]:
        print(f"\n{label}")

        for key, value in metrics.items():
            print(f"{key}: {value:.6f}")

    output = {
        "project": "D-Wave Portfolio Optimizer V6.1",
        "solver": "Local Simulated Annealing",
        "assets": prices.columns.tolist(),
        "historical_observations": len(prices),
        "transaction_cost": TRANSACTION_COST,
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
    print("Equity CSV:", EQUITY_CSV)
    print("Equity chart:", EQUITY_PNG)
    print("V6.1 completed successfully.")


if __name__ == "__main__":
    main()
