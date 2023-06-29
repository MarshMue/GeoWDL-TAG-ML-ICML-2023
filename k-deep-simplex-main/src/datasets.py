import numpy as np
import scipy.io as sio

import keras
from keras.datasets import mnist

from sklearn.datasets import make_moons

import torch
from torchvision import datasets, transforms


def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_all = np.concatenate([x_train, x_test])
    y_all = np.concatenate([y_train, y_test])
    x_all = x_all.reshape(x_all.shape[0], 784).astype(float) / 255.0
    return x_all, y_all


def get_yale_data(root="CroppedYale", height=192, width=168):
    faces = datasets.ImageFolder(
        root=root,
        transform=transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1)),  # flatten image
            ]
        ),
    )
    loader = torch.utils.data.DataLoader(faces, batch_size=len(faces))
    images = next(iter(loader))[0].numpy()
    labels = np.array(faces.targets)
    return images, labels


def get_hyperspectral_data(path):
    mat_contents = sio.loadmat(path)
    name = [
        x
        for x in mat_contents.keys()
        if x not in ["__header__", "__version__", "__globals__"]
    ][0]
    data = mat_contents[name]
    # flatten image
    data = data.reshape(data.shape[0] * data.shape[1], *data.shape[2:])
    return data


# example usage:

# from datasets import get_mnist_data, get_yale_data, get_hyperspectral_data
# x, y = get_mnist_data()
# x, y = get_yale_data()
# x = get_hyperspectral_data()


def prune(data, labels, wanted):
    n = len(data)
    idx = [i for i in range(n) if labels[i] in wanted]
    data = data[idx]
    labels = labels[idx]
    lookup = np.arange(int(labels.max() + 1))
    lookup[wanted] = np.arange(len(wanted))
    labels = lookup[labels]
    return data, labels


def get_data(dataset, path, quantity=10000):
    if dataset == "moons":
        data, labels = make_moons(quantity)
        data = torch.tensor(data).float()
        data += 0.1 * torch.randn(data.shape)
        labels = torch.tensor(labels)
        return data, labels, 2
    elif dataset == "mnist":
        data, labels = get_mnist_data()
        data, labels = prune(data, labels, [0, 3, 4, 6, 7])
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        return torch.tensor(data).float(), torch.tensor(labels), 5
    elif dataset == "salinas":
        data = get_hyperspectral_data(f"{path}/SalinasA_smallNoise.mat")
        labels = get_hyperspectral_data(f"{path}/SalinasA_gt.mat")
        data, labels = prune(data, labels, [0, 1, 10, 11, 12, 13, 14])
        data = torch.tensor(data).float()
        labels = torch.tensor(labels)
        data = data.reshape(86, 83, -1).permute(1, 0, 2).reshape(83 * 86, -1)
        data -= data.min()
        data /= data.max()
        data, labels = data[np.where(labels)], labels[np.where(labels)] - 1
        return data, labels, 6
    elif dataset == "yale2":
        data, labels = get_yale_data() if path is None else get_yale_data(path)
        data, labels = prune(data, labels, [4, 8])
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        return torch.tensor(data).float(), torch.tensor(labels), 2
    elif dataset == "yale3":
        data, labels = get_yale_data() if path is None else get_yale_data(path)
        data, labels = prune(data, labels, [4, 8, 20])
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        return torch.tensor(data).float(), torch.tensor(labels), 3
    else:
        print(f"unknown dataset '{dataset}")
