import torch


def gridCost(height, width):
    """
    make the cost function for a 2D grid

    :param height:
    :param width:
    :return:
    """
    norm = max(height, width)

    x = torch.linspace(0, width / norm, width)
    y = torch.linspace(0, height / norm, height)

    X, Y = torch.meshgrid(x, y, indexing="ij")

    C = torch.pow(X.T.reshape(-1, 1) - X.T.reshape(1, -1), 2) + torch.pow(Y.T.reshape(-1, 1) - Y.T.reshape(1, -1), 2)

    return C


def atomCost(D: torch.Tensor, OTSolver):
    """
    make the cost matrix between atom distributions where C[i,j] = W(D[:,i], D[:, j])
    :param atoms:
    :return:
    """

    n = D.shape[1]
    C = torch.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # arrays need to contiguous for W K means
            dist = OTSolver(D[:, i].contiguous(), D[:, j].contiguous())
            C[i, j] = dist
            C[j, i] = dist

    return C
