import math

import matplotlib.pyplot as plt
import numpy as np
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history

bounds = (np.zeros(6), np.ones(6))
options = {"c1": 0.5, "c2": 0.3, "w": 0.9}


def endurance(el):
    x, y, z, u, v, w = el
    return math.exp(-2 * (y - math.sin(x)) ** 2) + math.sin(z * u) + math.cos(v * w)


def f(x):
    return [-endurance(el) for el in x]


optimizer = ps.single.GlobalBestPSO(
    n_particles=10, dimensions=6, options=options, bounds=bounds
)

optimizer.optimize(f, iters=1000)

cost_history = optimizer.cost_history
plot_cost_history(cost_history)
plt.savefig("zad1_res/cost_hist.png")
