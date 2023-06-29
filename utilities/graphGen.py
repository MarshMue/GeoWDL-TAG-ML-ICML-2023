import torch


def KDSGraph(weights: torch.Tensor):
    n = weights.shape[1]
    k = weights.shape[0]

    A = torch.zeros((n + k, n + k))

    # make adjacency matrix with block 0s on the diagonal
    A[:n, n:] = weights.T
    A[n:, :n] = weights

    D = A.sum(dim=1)

    return D, A


def atomDistGraph(weights, atoms, OTSolver):
    n = weights.shape[0]

    A = torch.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            A[i, j] = A[j, i] = OTSolver(weights[:, i], weights[:, j])
