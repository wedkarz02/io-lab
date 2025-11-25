import pygad
import math
import numpy as np


def endurance(x, y, z, u, v, w):
    return math.exp(-2 * (y - math.sin(x)) ** 2) + math.sin(z * u) + math.cos(v * w)


def fitness_func(ga_instance, solution, solution_idx):
    x, y, z, u, v, w = solution
    return endurance(x, y, z, u, v, w)


gene_space = [{"low": 0, "high": 1} for _ in range(6)]

sol_per_pop = 50
num_genes = 6
num_parents_mating = 20
num_generations = 100
keep_parents = 10
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 15

ga_instance = pygad.GA(
    gene_space=gene_space,
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    parent_selection_type=parent_selection_type,
    keep_parents=keep_parents,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    mutation_percent_genes=mutation_percent_genes,
)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()

print(
    f"Parametry: x={solution[0]:.4f}, y={solution[1]:.4f}, z={solution[2]:.4f}, u={solution[3]:.4f}, v={solution[4]:.4f}, w={solution[5]:.4f}"
)
print(f"Najlepsza wytrzymałość: {solution_fitness:.6f}")

ga_instance.plot_fitness().get_figure().savefig("zad3_res/fitness_plot.png")

for i in range(2):
    ga_instance = pygad.GA(
        gene_space=gene_space,
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,
    )

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    print(f"\nUruchomienie {i+2}:")
    print(
        f"Parametry: x={solution[0]:.4f}, y={solution[1]:.4f}, z={solution[2]:.4f}, u={solution[3]:.4f}, v={solution[4]:.4f}, w={solution[5]:.4f}"
    )
    print(f"Najlepsza wytrzymałość: {solution_fitness:.6f}")

    ga_instance.plot_fitness().get_figure().savefig(f"zad3_res/fitness_plot_{i+2}.png")
