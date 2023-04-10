from numpy.linalg import inv
import numpy as np
import networkx as nx
import pandas as pd


def PN_score(P, N, beta=None):
    """Calculates the PN centrality score for each firm within the network.

    If no value for `beta` is given, the optimal value as determined by [...] is used.
    The entries for `N` should be non-positive.

    Parameters
    ----------
    P : numpy.array
        Matrix of positive links.
    N : numpy array
        Matrix of negative links.
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
    # entries of P are positive or zero, entries of N are negative or zero
    A = P + 2 * N
    n = A.shape[0]

    ones = np.ones((n, 1))
    identity = np.diag(np.ones(n))

    if beta is None:
        # beta = 1 / (2 * n - 2)
        maxdegree = np.max((np.sum(A, axis=0), np.sum(A, axis=1)))
        beta = 1 / (2 * maxdegree)

    scores = inv(identity - beta * A) @ ones
    return scores.reshape(1, -1)[0]


def eigenv_score(A):
    G = nx.from_numpy_array(A)
    centrality = nx.eigenvector_centrality(G)
    # df = pd.DataFrame([[v, c] for v, c in centrality.items()], columns=["firm","eigenv_score"])
    # sort by firm index
    sorted_list = sorted([[v, c] for v, c in centrality.items()], key=lambda tup: tup[0])
    # extract centrality scores
    scores = np.transpose(sorted_list)[1]
    return scores
