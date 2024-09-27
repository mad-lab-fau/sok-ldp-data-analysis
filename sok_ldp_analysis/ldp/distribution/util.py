import numpy as np


def project_onto_prob_simplex(y):
    """
    Projection onto the probability simplex. Based on pseudocode from: http://arxiv.org/abs/1309.1541.

    Args:
        y: The vector to project onto the probability simplex

    Returns: The projection of y onto the probability simplex
    """
    u = np.sort(y)[::-1]
    sum_u = np.cumsum(u)
    j = np.arange(len(u)) + 1

    # Find the highest possible index for which this is still positive
    term = u + (1 / j) * (1 - sum_u)
    rho = j[term > 0][-1] - 1

    # calculate lambda
    l = 1 / (rho + 1) * (1 - sum_u[rho])

    return np.maximum(y + l, 0)
