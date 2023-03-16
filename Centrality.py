from numpy.linalg import inv
import numpy as np


def PN_score(adj, beta=None):
    """Calculates the PN centrality score for each firm within the network.

    If no value for `beta` is given, the optimal value as determined by [...] is used.

    Parameters
    ----------
    adj : numpy.array
        The signed adjacency matrix of the network.
    beta : float, optional
        The normalization factor. (default is None)

    Returns
    -------
    numpy.array
        A vector of PN centrality scores.
    """
    # TODO: add a reference to the original paper.
    # TODO: does the matrix need to be ony -1 and 1, or are the daily trade volume values allowed?
    # TODO: this is only for symmetric adjacency matrices, also implement for non-symmetric matrices?
    P = np.copy(adj)
    N = np.copy(adj)

    P[P < 0] = 0
    N[N > 0] = 0
    N = -1 * N

    A = P - 2 * N
    n = A.shape[0]

    ones = np.ones((n, 1))
    identity = np.diag(np.ones(n))

    if beta is None:
        # beta = 1 / (2 * n - 2)
        maxdegree = np.max((np.sum(A, axis=0), np.sum(A, axis=1)))
        beta = 1 / (2 * maxdegree)

    scores = inv(identity - beta * A) @ ones
    return scores

