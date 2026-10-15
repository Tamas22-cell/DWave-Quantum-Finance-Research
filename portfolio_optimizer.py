
import numpy as np
import dimod
from dwave.samplers import SimulatedAnnealingSampler
from itertools import combinations
from time import perf_counter

# ==========================================
# D-WAVE QUANTUM PORTFOLIO OPTIMIZER V2
# ==========================================

assets = [
    "BTC", "ETH", "SOL", "ADA", "SUI",
    "NVDA", "MSFT", "GOOGL", "AMZN", "META",
    "AMD", "TSLA", "PLTR", "IONQ", "QBTS"
]

# Demonstracios eves varhato hozamok
returns = np.array([
    0.20, 0.16, 0.25, 0.14, 0.28,
    0.18, 0.12, 0.15, 0.14, 0.17,
    0.19, 0.16, 0.21, 0.24, 0.27
])

# Demonstracios eves volatilitasi adatok
volatility = np.array([
    0.65, 0.75, 0.90, 0.70, 1.00,
    0.45, 0.25, 0.35, 0.40, 0.42,
    0.55, 0.65, 0.70, 0.95, 1.10
])

# Pozitiv szemidefinit kovarianciamatrix
# Egyszerusitett kozos piaci faktormodell
market_correlation = 0.30

correlation = (
    market_correlation * np.ones((15, 15))
    + (1 - market_correlation) * np.eye(15)
)

covariance = np.outer(
    volatility, volatility
) * correlation

# Optimalizacios parameterek
target_assets = 4
risk_aversion = 0.5
penalty = 5.0

n = len(assets)

# ==========================================
# QUBO MODELL
# ==========================================

Q = {}

for i in range(n):
    Q[(i, i)] = (
        risk_aversion * covariance[i, i]
        - returns[i]
        + penalty * (1 - 2 * target_assets)
    )

    for j in range(i + 1, n):
        Q[(i, j)] = (
            2 * risk_aversion * covariance[i, j]
            + 2 * penalty
        )

bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

# ==========================================
# SIMULATED ANNEALING
# ==========================================

start = perf_counter()

sampler = SimulatedAnnealingSampler()

samples = sampler.sample(
    bqm,
    num_reads=1000,
    num_sweeps=1000
)

sa_time = perf_counter() - start

valid = [
    row
    for row in samples.data(["sample", "energy"])
    if sum(row.sample.values()) == target_assets
]

if not valid:
    raise RuntimeError(
        "No valid portfolio found by Simulated Annealing."
    )

best = min(valid, key=lambda row: row.energy)

selected_indices = [
    i for i in range(n)
    if best.sample[i] == 1
]

selected = [assets[i] for i in selected_indices]

# ==========================================
# EXACT BENCHMARK
# ==========================================

start = perf_counter()

# Pontosan 4 eszkoz osszes kombinacioja
# C(15,4) = 1365
best_exact_energy = float("inf")
best_exact_indices = None

for combo in combinations(range(n), target_assets):
    sample = {i: int(i in combo) for i in range(n)}
    energy = bqm.energy(sample)

    if energy < best_exact_energy:
        best_exact_energy = energy
        best_exact_indices = combo

exact_time = perf_counter() - start

exact_selected = [
    assets[i] for i in best_exact_indices
]

# ==========================================
# PORTFOLIO METRICS
# ==========================================

weights = np.zeros(n)
weights[selected_indices] = 1.0 / target_assets

portfolio_return = float(weights @ returns)
portfolio_variance = float(
    weights @ covariance @ weights
)
portfolio_volatility = np.sqrt(portfolio_variance)

# ==========================================
# RESULTS
# ==========================================

print("\n=== D-WAVE PORTFOLIO OPTIMIZER V2 ===")

print("Total assets:", n)
print("Selected assets:", selected)
print("Selected count:", len(selected))

print("\n=== PORTFOLIO METRICS ===")
print(
    "Expected annual return:",
    round(portfolio_return * 100, 2), "%"
)
print(
    "Annual volatility:",
    round(portfolio_volatility * 100, 2), "%"
)

print("\n=== SIMULATED ANNEALING ===")
print("QUBO energy:", round(best.energy, 6))
print("Runtime:", round(sa_time, 4), "seconds")

print("\n=== EXACT BENCHMARK ===")
print("Selected assets:", exact_selected)
print("Exact energy:", round(best_exact_energy, 6))
print("Runtime:", round(exact_time, 4), "seconds")

difference = best.energy - best_exact_energy

print("\n=== COMPARISON ===")
print("Energy difference:", round(difference, 8))

if abs(difference) < 1e-8:
    print("STATUS: GLOBAL OPTIMUM FOUND")
else:
    print("STATUS: SUBOPTIMAL SOLUTION")

print("\nSolver: D-Wave Simulated Annealing")
print("Hardware: Local CPU Simulator")
print("Optimization completed successfully.")
