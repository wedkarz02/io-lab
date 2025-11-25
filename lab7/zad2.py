import pygad
import numpy
import time
import os

przedmioty = [
    {"nazwa": "zegar", "waga": 7, "wartosc": 100},
    {"nazwa": "obraz-pejzaż", "waga": 7, "wartosc": 300},
    {"nazwa": "obraz-portret", "waga": 6, "wartosc": 200},
    {"nazwa": "radio", "waga": 2, "wartosc": 40},
    {"nazwa": "laptop", "waga": 5, "wartosc": 500},
    {"nazwa": "lampka nocna", "waga": 6, "wartosc": 70},
    {"nazwa": "srebrne sztućce", "waga": 1, "wartosc": 100},
    {"nazwa": "porcelana", "waga": 3, "wartosc": 250},
    {"nazwa": "figura z brazu", "waga": 10, "wartosc": 300},
    {"nazwa": "skórzana torebka", "waga": 3, "wartosc": 280},
    {"nazwa": "odkurzacz", "waga": 15, "wartosc": 300},
]

wagi = [item["waga"] for item in przedmioty]
wartosci = [item["wartosc"] for item in przedmioty]
nazwy = [item["nazwa"] for item in przedmioty]

max_waga = 25

gene_space = [0, 1]


def fitness_func(model, solution, solution_idx):
    suma_wartosci = numpy.sum(solution * wartosci)
    suma_wag = numpy.sum(solution * wagi)

    if suma_wag > max_waga:
        return 0

    return suma_wartosci


fitness_function = fitness_func

sol_per_pop = 20
num_genes = len(przedmioty)

num_parents_mating = 8
num_generations = 50
keep_parents = 4

parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 10


def on_generation(ga_instance):
    if ga_instance.best_solution()[1] >= 1630:
        return "stop"


success_count = 0
total_time = 0
successful_runs = 0

for i in range(10):
    start_time = time.time()

    ga_instance = pygad.GA(
        gene_space=gene_space,
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_function,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,
        on_generation=on_generation,
        stop_criteria=["reach_1630"],
    )

    ga_instance.run()

    end_time = time.time()
    run_time = end_time - start_time

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    if solution_fitness >= 1630:
        success_count += 1
        total_time += run_time
        successful_runs += 1

        if successful_runs == 1:
            selected_items = []
            total_weight = 0
            total_value = 0

            for j in range(len(solution)):
                if solution[j] == 1:
                    selected_items.append(nazwy[j])
                    total_weight += wagi[j]
                    total_value += wartosci[j]

            print(
                "Parameters of the best solution : {solution}".format(solution=solution)
            )
            print(
                "Fitness value of the best solution = {solution_fitness}".format(
                    solution_fitness=solution_fitness
                )
            )
            print("Total weight: {weight}".format(weight=total_weight))
            print("Total value: {value}".format(value=total_value))
            print("Selected items: {items}".format(items=selected_items))

            if not os.path.exists("zad2_res"):
                os.makedirs("zad2_res")
            ga_instance.plot_fitness().savefig("zad2_res/fitness_plot.png")

success_rate = (success_count / 10) * 100
average_time = total_time / success_count if success_count > 0 else 0

print("Success rate: {rate}%".format(rate=success_rate))
print("Average time for successful runs: {time} seconds".format(time=average_time))
