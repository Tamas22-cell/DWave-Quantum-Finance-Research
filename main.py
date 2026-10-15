import dimod
from dwave.samplers import SimulatedAnnealingSampler

# QUBO modell
Q = {
    (0, 0): -3,
    (1, 1): -2,
    (0, 1): 4,
}

bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

# 1. Exact Solver
exact = dimod.ExactSolver().sample(bqm)

# 2. Simulated Annealing
sa = SimulatedAnnealingSampler().sample(
    bqm,
    num_reads=1000,
    num_sweeps=1000
)

print("=== D-WAVE BENCHMARK ===")

print("\nExactSolver:")
print(exact.first)

print("\nSimulated Annealing:")
print(sa.first)

print("\nEnergy difference:")
print(sa.first.energy - exact.first.energy)