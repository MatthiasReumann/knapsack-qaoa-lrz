import os
import time
import dotenv

import numpy as np

from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution

import mqp.qiskit_provider as mqp_qiskit

from qiskit_aer import Aer

from qiskit_optimization.applications import Knapsack
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.minimum_eigensolvers import QAOA

from qiskit.transpiler import PassManager, StagedPassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit.primitives import BackendSampler

from passes import PauliTwirl

dotenv.load_dotenv()


class CustomKnapsack(Knapsack):
    def __init__(self, values: list[int], weights: list[int], max_weight: int):
        super().__init__(values, weights, max_weight)

    def to_docplex(self) -> Model:
        """Convert a knapsack problem instance into a
        :class:`~docplex.mp.model.Model`"""
        mdl = Model(name="Knapsack")
        x = {i: mdl.binary_var(name=f"x_{i}") for i in range(len(self._values))}
        mdl.maximize(mdl.sum(self._values[i] * x[i] for i in x))
        mdl.add_constraint(
            mdl.sum(self._weights[i] * x[i] for i in x) <= self._max_weight
        )
        return mdl


def qaoa_callback(
    it: int, params: np.ndarray, obj: float, options: dict[str, any]
) -> None:
    print(it, params, obj)


def run_benchmark(backend, sack: CustomKnapsack, repeats: int = 5):
    # Setup QAOA and different types of optimization.
    base_pm = generate_preset_pass_manager(
        backend=backend, optimization_level=0, seed_transpiler=42
    )
    final_pm = PassManager([PauliTwirl()])

    staged_pm = StagedPassManager(
        stages=["base", "final"], base=base_pm, final=final_pm
    )
    sampler = BackendSampler(
        backend=backend, options={"shots": 1024}, bound_pass_manager=staged_pm
    )
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA())
    problem = sack.to_quadratic_program()

    # Classical optimization.
    docplex = sack.to_docplex()
    sol: SolveSolution = docplex.solve()
    obj_classical = sol.get_objective_value()
    params_classical = np.array(
        sol.get_value_list(docplex.iter_binary_vars()), dtype=np.int64
    )
    rt_classical = sol.solve_details.time
    print(
        f"Classical[value={obj_classical}, params={params_classical}] ~ {rt_classical:.3f}s"
    )

    # Run Optimization.
    sum_hamming, sum_delta = 0, 0
    for _ in range(repeats):
        start = time.time()
        result = MinimumEigenOptimizer(qaoa).solve(problem)
        end = time.time()

        obj_quantum = result.fval
        params_quantum = np.array(result.x, dtype=np.int64)
        rt_quantum = end - start
        hamming = np.count_nonzero(params_classical != params_quantum)
        obj_delta = abs(obj_classical - obj_quantum)
        print(
            f"Quantum[value={obj_quantum}, params={params_quantum},"
            f" hamming={hamming}, delta={obj_delta}] ~ {rt_quantum:.3f}s"
        )

        sum_hamming += hamming
        sum_delta += obj_delta

    return sum_hamming / repeats, sum_delta / repeats


def run_benchmarks(backends: dict, problems: dict[str, CustomKnapsack]):
    print("backend; qubits; avg_hamming; avg_delta")
    for backend_name, backend in backends.items():
        for q, sack in problems.items():
            avg_hamming, avg_delta = run_benchmark(backend, sack, repeats=1)
            print(f"{backend_name}; {q}; {avg_hamming}; {avg_delta}")


if __name__ == "__main__":
    # Setup backends.
    provider = mqp_qiskit.MQPProvider(token=os.getenv("MQP_TOKEN"))
    backends = {
        # "aer": Aer.get_backend("aer_simulator"),
        # "qexa20": provider.get_backend("QExa20"),
        # "q20": provider.get_backend("Q20"),
        "aqt20": provider.get_backend("AQT20"),
    }

    # Define the problems.
    problems = {
        "5": CustomKnapsack(
            values=[7, 8, 2],
            weights=[3, 2, 1],  # 3
            max_weight=3,  # ceil(log(3)) = 2
        ),
        "10": CustomKnapsack(
            values=[7, 8, 3, 10, 11, 10],
            weights=[3, 2, 9, 7, 4, 10],  # 6
            max_weight=14,  # ceil(log(14)) = 4
        ),
        "15": CustomKnapsack(
            values=[7, 8, 3, 10, 11, 10, 12, 3],
            weights=[10, 9, 12, 14, 9, 10, 15, 16],  # 8
            max_weight=72,  # ceil(log(72)) = 7
        ),
        "20": CustomKnapsack(
            values=[8, 7, 9, 6, 12, 3, 10, 11, 10, 7, 1, 2, 3],
            weights=[10, 12, 9, 7, 10, 10, 11, 12, 7, 15, 8, 3, 7],  # 13
            max_weight=95,  # ceil(log(95)) = 7
        ),
    }

    # Run benchmarks.
    run_benchmarks(backends, problems)
