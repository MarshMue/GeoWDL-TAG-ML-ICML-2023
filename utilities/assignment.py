from scipy.optimize import linear_sum_assignment
import torch


def makeCostMatrix(pred_labels: torch.Tensor, true_labels: torch.Tensor):
    """
    pred and true labels me be of different cardinalities.
    This cost matrix entries are the number of misclassifications that would occur if you were to put predicted label i
    to true label j

    :param pred_labels:
    :param true_labels:
    :param k:
    :return:
    """

    # number of pred labels
    unique_pred_labels = pred_labels.unique().sort()[0]
    kp = len(unique_pred_labels)

    unique_true_labels = true_labels.unique().sort()[0]
    kt = len(unique_true_labels)

    C = torch.zeros((kp, kt))

    for i in range(kp):
        for j in range(kt):
            # compute number of mislabels from label i to j

            # get indices of predicted labels i
            pred_idx = pred_labels == unique_pred_labels[i]

            true_idx = true_labels == unique_true_labels[j]

            # among true labels at position corresponding to predicted label i, how many are not label j
            # total number of labelings of the predicted label minus the number of pairs that coincide with the target label
            num_mislabels = pred_idx.sum() - torch.bitwise_and(pred_idx, true_idx).sum()
            C[i, j] = num_mislabels

    return C


def relabelPredictions(pred_labels: torch.Tensor, true_labels: torch.Tensor, method: str = "linear-assign"):
    """
    return a new vector of labels that correspond to the true labels with as few mislabelings as possible

    :param C:
    :param pred:
    :return:
    """

    C = makeCostMatrix(pred_labels, true_labels)
    unique_pred_labels = pred_labels.unique().sort()[0]
    unique_true_labels = true_labels.unique().sort()[0]
    new_pred = torch.zeros_like(pred_labels, dtype=int)

    if method == "min-error":
        assignments = C.argmin(dim=1)
        for i in range(C.shape[0]):
            new_pred[pred_labels == unique_pred_labels[i]] = unique_true_labels[assignments[i]]
    elif method == "linear-assign":
        if len(unique_pred_labels) != len(unique_true_labels):
            raise ValueError("differing numbers of unique labels")

        old_assignments, assignments = linear_sum_assignment(C)

        for i in old_assignments:
            # take old assignment i and give it new assignment
            new_pred[pred_labels == unique_pred_labels[i]] = unique_true_labels[assignments[i]]

    return new_pred


def labelAccuracy(pred_labels: torch.Tensor, true_labels: torch.Tensor):
    """
    determine the accuracy

    :param pred_labels:
    :param true_labels:
    :return:
    """

    n = pred_labels.view(-1, 1).shape[0]

    # non zeros in difference will be mislabels
    return 1 - (pred_labels - true_labels).count_nonzero() / n


def relabelAccuracy(pred_labels: torch.Tensor, true_labels: torch.Tensor, method: str = "linear-assign"):
    """
    Same as label accuracy but first solves the relabeling problem

    :return:
    """

    relabeled_pred = relabelPredictions(pred_labels, true_labels, method)

    return labelAccuracy(relabeled_pred, true_labels)
