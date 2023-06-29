import time
from pathlib import Path

import numpy as np
import scipy.sparse as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sacred import Experiment
from sklearn.cluster import k_means
from sklearn.manifold import spectral_embedding
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm as progress_bar

import model, datasets, utils


ex = Experiment("Application of K-Deep-Simplex to Clustering")


@ex.config
def cfg():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    hyp = {
        "num_layers": 15,
        "input_size": 2,
        "hidden_size": 24,
        "penalty": 5.0,
        "train_step": True,
    }
    lr = 1e-3
    dataset = "moons"
    quantity = 10000
    path_to_data = "../data"
    subset = None
    epochs = 1000
    batch_size = 10000
    workers = 4
    seed = 0


@ex.named_config
def moons_default():
    pass


@ex.named_config
def mnist_default():
    dataset = "mnist"
    hyp = {
        "num_layers": 100,
        "input_size": 784,
        "hidden_size": 500,
        "penalty": 0.5,
        "train_step": False,
    }
    lr = 1e-3
    epochs = 30
    batch_size = 1024


@ex.named_config
def salinas_default():
    dataset = "salinas"
    hyp = {
        "num_layers": 100,
        "input_size": 224,
        "hidden_size": 600,
        "penalty": 2.0,
        "train_step": True,
    }
    lr = 1e-4
    epochs = 30
    batch_size = 128


@ex.named_config
def yale2_default():
    dataset = "yale2"
    hyp = {
        "num_layers": 50,
        "input_size": 32256,
        "hidden_size": 64,
        "penalty": 0.1,
        "train_step": False,
    }
    lr = 1e-4
    epochs = 15
    batch_size = 1


@ex.named_config
def yale3_default():
    dataset = "yale3"
    hyp = {
        "num_layers": 50,
        "input_size": 32256,
        "hidden_size": 96,
        "penalty": 0.1,
        "train_step": False,
    }
    lr = 1e-4
    epochs = 15
    batch_size = 1


def fast_embedding(code, k):
    graph = code.T @ code
    embedding = spectral_embedding(graph, n_components=k, drop_first=False)
    embedding = np.concatenate([code @ embedding, embedding])
    return embedding


@ex.automain
def run(
    _run,
    device,
    hyp,
    lr,
    dataset,
    quantity,
    path_to_data,
    subset,
    epochs,
    batch_size,
    workers,
    seed,
):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    start_loading_data = time.time()

    data, labels, k = datasets.get_data(dataset, path_to_data, quantity)

    if subset is not None:
        data = data[:subset]
        labels = labels[:subset]

    start_k_means = time.time()

    predictions = k_means(data, k)[1]
    score = utils.clustering_accuracy(labels, predictions)[0]
    print(f"baseline accuracy = {score:.6f}")

    start_training = time.time()

    net = model.KDS(**hyp)
    with torch.no_grad():
        p = torch.randperm(len(data))[: net.hidden_size]
        net.W.data = data[p]
        net.step.fill_((net.W.data.svd()[1][0] ** -2).item())
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = utils.LocalDictionaryLoss(hyp["penalty"])

    net.train()
    for epoch in progress_bar(range(epochs)):
        shuffle = torch.randperm(len(data))
        data, labels = data[shuffle], labels[shuffle]
        for i in progress_bar(range(0, len(data), batch_size), disable=True):
            y = data[i : i + batch_size].to(device)
            x_hat = net.encode(y)
            loss = criterion(net.W, y, x_hat)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1e-4)
            optimizer.step()

    with torch.no_grad():
        net.eval()
        x_hat = []
        for i in progress_bar(range(0, len(data), batch_size)):
            y = data[i : i + batch_size].to(device)
            x_hat.append(net.encode(y).cpu())
        x_hat = torch.cat(x_hat)

    start_clustering = time.time()

    print("clustering...")
    embedding = fast_embedding(x_hat.numpy(), k)
    embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
    predictions = k_means(embedding, k)[1]
    truncated_predictions = torch.tensor(predictions)[: -net.hidden_size]
    score = utils.clustering_accuracy(labels, truncated_predictions)[0]
    print(f"accuracy = {score:.6f}")

    stop = time.time()

    print(f"time to load data = {start_k_means - start_loading_data:.2f}")
    print(f"time to do k-means = {start_training - start_k_means:.2f}")
    print(f"time to train = {start_clustering - start_training:.2f}")
    print(f"time to cluster = {stop - start_clustering:.2f}")

    print("saving network and optimizer...")
    path = Path(_run.observers[0].dir if _run.observers else ".")
    save = {"net": net.state_dict(), "opt": optimizer.state_dict()}
    torch.save(save, path / "final_state.pt")
