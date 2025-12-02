import numpy as np
import pygad
import time
from PIL import Image, ImageDraw

maze = np.array(
    [
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
)

start = (1, 1)
goal = (10, 10)
moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
chrom_length = 30
gene_space = [0, 1, 2, 3]


def run_path(chrom):
    x, y = start
    path = [(x, y)]
    for g in chrom:
        g = int(g)
        dx, dy = moves[g]
        nx, ny = x + dx, y + dy
        if maze[nx][ny] == 0:
            x, y = nx, ny
            path.append((x, y))
        if (x, y) == goal:
            break
    return path


def fitness_func(ga_instance, solution, sol_idx):
    solution = [int(g) for g in solution]
    path = run_path(solution)
    end = path[-1]
    dist = abs(end[0] - goal[0]) + abs(end[1] - goal[1])
    if end == goal:
        return 1000 - len(path)
    return 1 / (1 + dist)


def on_generation(ga):
    if ga.best_solution()[1] > 500:
        ga.stop_generation = True


def save_image(path, idx):
    cell = 20
    img = Image.new(
        "RGB", (maze.shape[1] * cell, maze.shape[0] * cell), (255, 255, 255)
    )
    d = ImageDraw.Draw(img)
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            color = (0, 0, 0) if maze[i][j] == 1 else (255, 255, 255)
            d.rectangle(
                [j * cell, i * cell, (j + 1) * cell, (i + 1) * cell], fill=color
            )
    for x, y in path:
        d.rectangle(
            [y * cell, x * cell, (y + 1) * cell, (x + 1) * cell], fill=(255, 0, 0)
        )
    img.save(f"zad4_res/path_{idx}.png")


times = []
with open("zad4_res/run_times.txt", "w") as f:
    for i in range(10):
        t0 = time.time()
        ga = pygad.GA(
            num_generations=500,
            sol_per_pop=300,
            num_parents_mating=50,
            fitness_func=fitness_func,
            gene_space=gene_space,
            num_genes=chrom_length,
            mutation_probability=0.1,
            on_generation=on_generation,
        )
        ga.run()
        sol, fit, _ = ga.best_solution()
        sol = [int(g) for g in sol]
        path = run_path(sol)
        save_image(path, i)
        t = time.time() - t0
        times.append(t)
        f.write(f"{t}\n")

print("avg time:", sum(times) / len(times))
