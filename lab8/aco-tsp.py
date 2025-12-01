import matplotlib.pyplot as plt
import random
import sys

from aco import AntColony

plt.style.use("dark_background")

# COORDS = ((20, 52), (43, 50), (20, 84), (70, 65), (29, 90), (87, 83), (73, 23))


def random_coords(count, min, max):
    return tuple(
        [(random.randrange(min, max), random.randrange(min, max)) for _ in range(count)]
    )


coords_len = 15
COORDS = random_coords(coords_len, 0, 100)

ant_count = 1
iterations = 1

if len(sys.argv) > 1 and sys.argv[1] == "grid":
    COORDS = (
        (0, 0),
        (10, 0),
        (20, 0),
        (30, 0),
        (40, 0),
        (0, 10),
        (10, 10),
        (20, 10),
        (30, 10),
        (40, 10),
        (0, 20),
        (10, 20),
        (20, 20),
        (30, 20),
        (40, 20),
        (0, 30),
        (10, 30),
        (20, 30),
        (30, 30),
        (40, 30),
        (0, 40),
        (10, 40),
        (20, 40),
        (30, 40),
        (40, 40),
    )
    ant_count = 10
    iterations = 100
    coords_len = len(COORDS)


def plot_nodes(w=12, h=8):
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches(w, h)


def plot_all_edges():
    paths = ((a, b) for a in COORDS for b in COORDS)

    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))


plot_nodes()


colony = AntColony(
    COORDS,
    ant_count=ant_count,
    alpha=0.5,
    beta=1.2,
    pheromone_evaporation_rate=0.40,
    pheromone_constant=1000.0,
    iterations=iterations,
)

optimal_nodes = colony.get_path()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )

plt.savefig(f"zad2_res/antc_{ant_count}_iter_{iterations}_clen_{coords_len}.png")

