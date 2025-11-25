import pygad
import numpy as np
import time

labirynt = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

labirynt = np.array(labirynt)
start = (1, 1)
cel = (10, 10)

ruch_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def fitness_func(ga_instance, solution, solution_idx):
    x, y = start
    odwiedzone = set()
    odwiedzone.add((x, y))

    for i, gen in enumerate(solution):
        if i >= 30:
            break

        dx, dy = ruch_map[gen % 4]
        new_x, new_y = x + dx, y + dy

        if 0 <= new_x < labirynt.shape[0] and 0 <= new_y < labirynt.shape[1]:
            if labirynt[new_x, new_y] == 0:
                x, y = new_x, new_y
                odwiedzone.add((x, y))

        if (x, y) == cel:
            break

    odleglosc_do_celu = abs(x - cel[0]) + abs(y - cel[1])

    if (x, y) == cel:
        fitness = 1000 + (30 - len(odwiedzone))
    else:
        fitness = 1.0 / (1.0 + odleglosc_do_celu) + len(odwiedzone) * 0.1

    return fitness


gene_space = [0, 1, 2, 3]
num_genes = 30
sol_per_pop = 300
num_parents_mating = 150
num_generations = 400
keep_parents = 75
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 4

czasy_wykonania = []
znalezione_rozwiazania = 0

for i in range(10):
    start_time = time.time()

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
        stop_criteria="reach_1000",
    )

    ga_instance.run()
    end_time = time.time()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    x, y = start
    sciezka = [start]
    for gen in solution:
        dx, dy = ruch_map[gen % 4]
        new_x, new_y = x + dx, y + dy

        if 0 <= new_x < labirynt.shape[0] and 0 <= new_y < labirynt.shape[1]:
            if labirynt[new_x, new_y] == 0:
                x, y = new_x, new_y
                sciezka.append((x, y))

        if (x, y) == cel:
            break

    if (x, y) == cel:
        znalezione_rozwiazania += 1
        print(
            f"Uruchomienie {i+1}: Znaleziono rozwiązanie! Długość ścieżki: {len(sciezka)}"
        )
    else:
        print(
            f"Uruchomienie {i+1}: Nie znaleziono pełnej ścieżki. Pozycja końcowa: ({x}, {y}), Odległość do celu: {abs(x-cel[0]) + abs(y-cel[1])}"
        )

    czasy_wykonania.append(end_time - start_time)
    ga_instance.plot_fitness().get_figure().savefig(f"zad4_res/fitness_plot_{i+1}.png")

sredni_czas = np.mean(czasy_wykonania)
print(f"\nŚredni czas wykonania: {sredni_czas:.2f} sekund")
print(f"Znalezione rozwiązania: {znalezione_rozwiazania}/10")

with open("zad4_res/wyniki.txt", "w") as f:
    f.write(f"Średni czas wykonania: {sredni_czas:.2f} sekund\n")
    f.write(f"Znalezione rozwiązania: {znalezione_rozwiazania}/10\n")
    for i, czas in enumerate(czasy_wykonania):
        f.write(f"Uruchomienie {i+1}: {czas:.2f} sekund\n")
