import torch
import torch.nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment


class LocalDictionaryLoss(torch.nn.Module):
    def __init__(self, penalty):
        super(LocalDictionaryLoss, self).__init__()
        self.penalty = penalty

    def forward(self, A, y, x):
        return self.forward_detailed(A, y, x)[2]

    def forward_detailed(self, A, y, x):
        weight = (y.unsqueeze(1) - A.unsqueeze(0)).pow(2).sum(dim=2)
        a = 0.5 * (y - x @ A).pow(2).sum(dim=1).mean()
        b = (weight * x).sum(dim=1).mean()
        return a, b, a + b * self.penalty


def clustering_accuracy(targets, predictions, k=None, return_matching=False):
    assert len(targets) == len(predictions)
    n = len(targets)
    if k is None:
        k = int(targets.max() + 1)
    cost = torch.zeros(k, k)
    for i in range(n):
        cost[targets[i].item(), predictions[i].item()] -= 1
    matching = linear_sum_assignment(cost)
    breakdown = -cost[matching]
    total = breakdown.sum().item() / n
    for i in range(k):
        breakdown[i] /= -cost[i].sum()
    if return_matching:
        return total, breakdown, matching
    else:
        return total, breakdown
