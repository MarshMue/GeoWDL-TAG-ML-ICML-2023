import torch
import torchvision.transforms as T
from numpy.random import choice
from torchvision.datasets import MNIST


def MNISTtoHistograms(path_to_dir: str,
                      digits: [int],
                      samples_per_digit: int = 100,
                      height: int = 28,
                      width: int = 28,
                      dev: torch.device = torch.device("cpu"),
                      ):
    # get MNIST data set
    dataset = MNIST(root=path_to_dir, download=True)

    # make resize function and convert to tensor
    resize = T.Compose(
        [T.ToPILImage(), T.Resize((height, width)), T.ToTensor()])

    # take subsamples of digits
    n_digits = len(digits)
    data = torch.zeros((n_digits * samples_per_digit, height, width), device=dev)
    labels = torch.zeros((n_digits * samples_per_digit,), dtype=int)

    for j in range(n_digits):
        # get number of digits in class
        curr_digit = dataset.data[dataset.targets == digits[j]]
        n_in_class = curr_digit.shape[0]

        indices = choice(range(n_in_class), samples_per_digit, replace=False)

        # subtract 1 from labels to have them start from 0
        labels[j * samples_per_digit: (j + 1) * samples_per_digit] = digits[j]
        for i in range(samples_per_digit):
            idx = indices[i]
            data[i + j * samples_per_digit] = resize(curr_digit[idx])[0]

    data_dim = height * width

    # pad data away from 0
    padding = 2e-1
    data += padding

    # normalize images
    torch.divide(data, torch.sum(torch.sum(data, dim=2), dim=1).view(data.shape[0], 1, 1), out=data)

    # reshape data into columns of where the images are stacked by taking each row as a vector and then stacking them
    # sequentially
    return data.reshape(-1, data_dim).T, labels


def imageSupport(height: int, width: int) -> torch.Tensor:
    """
    gives the 2D tensor of image support unrolled

    :param height:
    :param width:
    :return:
    """

    x = torch.linspace(0, 1, width)
    y = torch.linspace(0, 1, height)
    ys, xs = torch.meshgrid(y, x, indexing='ij')

    points = torch.stack((ys.reshape(-1), xs.reshape(-1)), dim=1)

    return points
