
import json
import time
from itertools import combinations
from pathlib import Path

import dimod
import numpy as np
import pandas as pd
from dwave.samplers import SimulatedAnnealingSampler

# ============================================
# D-WAVE PORTFOLIO OPTIMIZER V7
# ADVANCED RISK MANAGEMENT
# ============================================

BASE = Path(__file__).resolve().parent
CACHE = BASE / "market_cache_v6.csv"
OUTPUT = BASE / "results_v7.json"
CURVE_FILE = BASE / "equity_curve_v7.csv"

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

CRYPTO = {
    "BTC-USD", "ETH-USD", "SOL-USD",
    "ADA-USD", "SUI20947-USD",
    "AVAX-USD", "LINK-USD", "XRP-USD",
    "DOGE-USD", "LTC-USD"
}

QUANTUM = {"IONQ", "QBTS", "RGTI", "QUBT"}

TARGET = 4
TRAIN_DAYS = 126
TEST_DAYS = 21

RISK_AVERSION = 0.5
MAX_VOLATILITY = 0.35
TRANSACTION_COST = 0.001

CARDINALITY_PENALTY = 20.0
GROUP_PENALTY = 20.0

EXPERIMENTS = [
    (500, 1000, 42),
    (1000, 2000, 123),
    (1500, 3000, 777)
]

# ============================================
# MARKET DATA
# ============================================

def load_data():
    if not CACHE.exists():
        raise FileNotFoundError(
            f"Market cache not found: {CACHE}"
        )

    prices = pd.read_csv(
        CACHE,
        index_col=0,
        parse_dates=True
    )

    prices.index = pd.to_datetime(
        prices.index, utc=True
    ).tz_convert(None).normalize()

    prices = prices.loc[
        ~prices.index.duplicated(keep="last")
    ].sort_index()

    available = [
        a for a in ASSETS
        if a in prices.columns
        and prices[a].notna().sum() >= 150
    ]

    prices = prices[available]
    prices = prices.loc[prices.index.dayofweek < 5]
    prices = prices.ffill().dropna()

    if len(available) < TARGET:
        raise RuntimeError("Not enough assets.")

    if len(prices) < TRAIN_DAYS + TEST_DAYS + 2:
        raise RuntimeError("Insufficient history.")

    return prices


# ============================================
# RISK AND OBJECTIVE
# ============================================

def portfolio_stats(indices, mu, cov):
    idx = np.array(indices, dtype=int)

    annual_return = float(mu[idx].mean())

    variance = float(
        cov[np.ix_(idx, idx)].sum()
        / TARGET**2
    )

    volatility = float(
        np.sqrt(max(variance, 0.0))
    )

    objective = (
        RISK_AVERSION * variance
        - annual_return
    )

    return objective, annual_return, volatility


def valid_categories(indices, assets):
    names = [assets[i] for i in indices]

    crypto_count = sum(
        name in CRYPTO for name in names
    )

    quantum_count = sum(
        name in QUANTUM for name in names
    )

    return (
        len(indices) == TARGET
        and crypto_count <= 1
        and quantum_count <= 1
    )


def valid_portfolio(indices, assets, mu, cov):
    if not valid_categories(indices, assets):
        return False

    _, _, volatility = portfolio_stats(
        indices, mu, cov
    )

    return volatility <= MAX_VOLATILITY + 1e-10


# ============================================
# QUBO WITH CATEGORY PENALTIES
# ============================================

def build_bqm(mu, cov, assets):
    n = len(assets)
    Q = {}

    for i in range(n):
        Q[(i, i)] = (
            RISK_AVERSION * cov[i, i] / TARGET**2
            - mu[i] / TARGET
            + CARDINALITY_PENALTY * (1 - 2 * TARGET)
        )

        for j in range(i + 1, n):
            value = (
                2 * RISK_AVERSION
                * cov[i, j] / TARGET**2
                + 2 * CARDINALITY_PENALTY
            )

            if (
                assets[i] in CRYPTO
                and assets[j] in CRYPTO
            ):
                value += GROUP_PENALTY

            if (
                assets[i] in QUANTUM
                and assets[j] in QUANTUM
            ):
                value += GROUP_PENALTY

            Q[(i, j)] = value

    return dimod.BinaryQuadraticModel.from_qubo(Q)


# ============================================
# EXACT CONSTRAINED BENCHMARK
# ============================================

def exact_solver(mu, cov, assets):
    start = time.perf_counter()

    best = None
    best_score = float("inf")
    feasible_count = 0

    for combo in combinations(
        range(len(assets)), TARGET
    ):
        if not valid_portfolio(
            combo, assets, mu, cov
        ):
            continue

        feasible_count += 1

        score, _, _ = portfolio_stats(
            combo, mu, cov
        )

        if score < best_score:
            best_score = score
            best = list(combo)

    runtime = time.perf_counter() - start

    return best, best_score, runtime, feasible_count


# ============================================
# CONSTRAINED SIMULATED ANNEALING
# ============================================

def sa_solver(bqm, mu, cov, assets):
    sampler = SimulatedAnnealingSampler()

    best = None
    best_score = float("inf")
    best_parameters = None
    total_runtime = 0.0

    for reads, sweeps, seed in EXPERIMENTS:
        start = time.perf_counter()

        samples = sampler.sample(
            bqm,
            num_reads=reads,
            num_sweeps=sweeps,
            seed=seed
        )

        total_runtime += (
            time.perf_counter() - start
        )

        for row in samples.data(["sample"]):
            indices = [
                i for i in range(len(assets))
                if row.sample[i] == 1
            ]

            if not valid_portfolio(
                indices, assets, mu, cov
            ):
                continue

            score, _, _ = portfolio_stats(
                indices, mu, cov
            )

            if score < best_score:
                best_score = score
                best = indices
                best_parameters = {
                    "reads": reads,
                    "sweeps": sweeps,
                    "seed": seed
                }

    return (
        best,
        best_score,
        total_runtime,
        best_parameters
    )


# ============================================
# METRICS
# ============================================

def metrics(values):
    r = np.asarray(values, dtype=float)

    if len(r) < 2:
        return {}

    equity = np.cumprod(1 + r)

    volatility = float(
        np.std(r, ddof=1) * np.sqrt(252)
    )

    mean_annual = float(np.mean(r) * 252)

    downside = np.minimum(r, 0)
    downside_vol = float(
        np.sqrt(np.mean(downside**2) * 252)
    )

    peaks = np.maximum.accumulate(
        np.concatenate(([1.0], equity))
    )[1:]

    return {
        "annual_return": float(
            equity[-1] ** (252 / len(r)) - 1
        ),
        "annual_volatility": volatility,
        "sharpe": (
            mean_annual / volatility
            if volatility > 0 else None
        ),
        "sortino": (
            mean_annual / downside_vol
            if downside_vol > 0 else None
        ),
        "max_drawdown": float(
            np.min(equity / peaks - 1)
        ),
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

        mu = train.mean().to_numpy() * 252
        cov = train.cov().to_numpy() * 252

        bqm = build_bqm(mu, cov, assets)

        exact, exact_score, exact_time, count = (
            exact_solver(mu, cov, assets)
        )

        sa, sa_score, sa_time, params = (
            sa_solver(bqm, mu, cov, assets)
        )

        window_number = len(windows) + 1

        if exact is None:
            print(
                f"Window {window_number:02d}: "
                "NO FEASIBLE PORTFOLIO"
            )
            continue

        # If SA cannot find a feasible portfolio,
        # record this explicitly and skip this window.
        if sa is None:
            print(
                f"Window {window_number:02d}: "
                "SA FOUND NO FEASIBLE SOLUTION"
            )
            continue

        sa_set = set(sa)
        exact_set = set(exact)

        sa_turnover = len(
            sa_set.symmetric_difference(previous_sa)
        ) / (2 * TARGET)

        exact_turnover = len(
            exact_set.symmetric_difference(previous_exact)
        ) / (2 * TARGET)

        sa_daily = test.iloc[:, sa].mean(axis=1).copy()
        exact_daily = test.iloc[
            :, exact
        ].mean(axis=1).copy()

        sa_daily.iloc[0] -= (
            sa_turnover * TRANSACTION_COST
        )

        exact_daily.iloc[0] -= (
            exact_turnover * TRANSACTION_COST
        )

        sa_returns.extend(sa_daily.tolist())
        exact_returns.extend(exact_daily.tolist())
        dates.extend(test.index.tolist())

        previous_sa = sa_set
        previous_exact = exact_set

        _, sa_expected, sa_vol = portfolio_stats(
            sa, mu, cov
        )

        _, exact_expected, exact_vol = (
            portfolio_stats(exact, mu, cov)
        )

        gap = max(0.0, sa_score - exact_score)

        record = {
            "window": window_number,
            "test_start": str(test.index[0].date()),
            "test_end": str(test.index[-1].date()),
            "sa_assets": [assets[i] for i in sa],
            "exact_assets": [
                assets[i] for i in exact
            ],
            "sa_objective": sa_score,
            "exact_objective": exact_score,
            "objective_gap": gap,
            "sa_train_return": sa_expected,
            "exact_train_return": exact_expected,
            "sa_train_volatility": sa_vol,
            "exact_train_volatility": exact_vol,
            "sa_turnover": sa_turnover,
            "exact_turnover": exact_turnover,
            "sa_runtime": sa_time,
            "exact_runtime": exact_time,
            "feasible_portfolios": count,
            "best_parameters": params
        }

        windows.append(record)

        print(
            f"Window {window_number:02d} | "
            f"Gap={gap:.8f} | "
            f"SA Vol={sa_vol:.2%} | "
            f"Exact Vol={exact_vol:.2%}"
        )

    if not sa_returns:
        raise RuntimeError(
            "No comparable backtest windows."
        )

    curve = pd.DataFrame(
        {
            "SA": np.cumprod(
                1 + np.array(sa_returns)
            ),
            "Exact": np.cumprod(
                1 + np.array(exact_returns)
            )
        },
        index=pd.DatetimeIndex(dates)
    )

    curve.index.name = "Date"
    curve.to_csv(CURVE_FILE)

    return {
        "windows": windows,
        "sa_metrics": metrics(sa_returns),
        "exact_metrics": metrics(exact_returns),
        "completed_windows": len(windows),
        "test_observations": len(sa_returns)
    }


# ============================================
# MAIN
# ============================================

def main():
    print("\n=== D-WAVE PORTFOLIO V7 ===")
    print("Advanced Risk Management")
    print("Solver: Local Simulated Annealing\n")

    prices = load_data()

    print("Available assets:", len(prices.columns))
    print("Historical observations:", len(prices))
    print("Maximum volatility:", MAX_VOLATILITY)
    print("Maximum crypto assets: 1")
    print("Maximum quantum stocks: 1")

    print("\nRunning constrained backtesting...")

    results = run_backtest(prices)

    print("\n=== FINAL RISK REPORT ===")

    for label, values in [
        ("SIMULATED ANNEALING", results["sa_metrics"]),
        ("EXACT BENCHMARK", results["exact_metrics"])
    ]:
        print("\n" + label)

        for key, value in values.items():
            if value is None:
                print(f"{key}: N/A")
            else:
                print(f"{key}: {value:.6f}")

    output = {
        "project": "D-Wave Portfolio V7",
        "solver": "Local Simulated Annealing",
        "assets": prices.columns.tolist(),
        "max_train_volatility": MAX_VOLATILITY,
        "max_crypto": 1,
        "max_quantum": 1,
        "transaction_cost": TRANSACTION_COST,
        "note": (
            "Historical research only. Volatility "
            "constraints are estimated on training data."
        ),
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
    print("Equity CSV:", CURVE_FILE)
    print("V7 completed successfully.")


if __name__ == "__main__":
    main()
