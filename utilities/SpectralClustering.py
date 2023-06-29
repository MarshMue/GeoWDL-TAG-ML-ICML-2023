import torch
from sklearn.cluster import KMeans


def spectralClustering(D: torch.Tensor, A: torch.Tensor, k: int,
                       method: str = "normalized-symmetric", device: str = None):
    """
    meta function to perform spectral clustering according to the specified method

    :param D: The (n x 1) degree matrix (stored as a vector)
    :param A: the (n x n) adjacency matrix
    :param k: the number of classes to find
    :param device: the device to perform computations on (should match the device of D and A)
    :param method: the method of spectral clustering to perform
    :return: an (n) tensor of class labels
    """

    if device == None:
        device = D.device.type

    # make laplacian
    L = makeLaplacian(D, A, method)
    n = L.shape[0]

    # find eigenvectors
    if method == "normalized-randomwalk":
        eigenvectors = findEigenvectorsSymmetric(L, k).div(D.sqrt().view(n, 1))
    else:
        eigenvectors = findEigenvectorsSymmetric(L, k)

    # perform kmeans on eigenvectors
    kmeans = KMeans(k, n_init=100)
    kmeans.fit(eigenvectors.type(torch.float))
    return torch.tensor(kmeans.labels_)


def makeLaplacian(D: torch.Tensor, A: torch.Tensor, method: str = "normalized-symmetric"):
    n = A.shape[0]
    if method == "unnormalized":
        return torch.diag(D) - A
    elif method == "normalized-symmetric" or method == "normalized-randomwalk":
        Dsqrt = 1.0 / D.sqrt()
        return torch.eye(n) - Dsqrt.view(-1, 1).mul(A.mul(Dsqrt.view(1, -1)))
    else:
        raise NotImplementedError(f"Method \"{method}\" not implemented.")


def findEigenvectorsSymmetric(L, k):
    n = L.shape[0]
    if n < 500:
        _, eigenvectors = torch.linalg.eigh(L)
    else:
        _, eigenvectors = torch.lobpcg(L, k + 1, largest=False)

    return eigenvectors[:, 1:k + 1]
