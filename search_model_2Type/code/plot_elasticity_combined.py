import numpy as np
import matplotlib.pyplot as plt
from solvemodel import computeElasticity, solveModel

#Params

delta = 0.98
k1 = 12.0
gamma = 0.2
mu1 = 4.0
sigma = 0.5
kappa = 24.0
pi = 0.0

eta = 0.1
p0 = 0.5
a = 10.0
s_bar = 0.3

T = 31
P1 = 12

ben = np.ones(T) * 800 / 30
ben[:P1] = 1100 / 30
inst = (T, ben)

timevec = np.arange(T) + 1

# Alpha values to compare
alphas = [0.1, 0.5, 0.9]

# Plot

plt.figure()

for alpha in alphas:
    xi = np.array([
        delta, k1, gamma, mu1, sigma, kappa, pi,
        eta, alpha, p0, a, s_bar
    ])

    elasticity = computeElasticity(xi, inst)

    plt.plot(timevec, elasticity, label=f"alpha={alpha}")

plt.axvline(x=12, linestyle="dashed")
plt.ylim(-3, 0)

plt.xlabel("Months")
plt.ylabel("Elasticity")
plt.title("Elasticity of Search Effort across Alpha")
plt.legend()

plt.savefig("fig_elasticity_combined.png", bbox_inches="tight")
plt.show()
