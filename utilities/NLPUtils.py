import matplotlib.axes
import scipy.io as sio
import torch
import numpy as np
from numpy.random import choice


def loadWordHistograms(samples_per_class: int, fname: str, classes_to_use=None, random_sample: int = 0):
    """

    :param samples_per_class: number of samples to pick from each class
    :return:
    """

    # validate input
    assert (samples_per_class >= 0)

    # load data
    embeddings, histograms, classes, word_to_idx, wordsList = loadData(fname)

    classes = torch.tensor(classes)

    if classes_to_use is None:
        classes_to_use = classes.unique()
    else:
        # validate that classes exist
        for class_label in classes_to_use:
            assert (class_label in classes)

    # first figure out how big of an array to allocate
    size = 0
    for class_label in classes_to_use:
        class_idxs = np.where(classes == class_label)[0]
        n_in_class = len(class_idxs)
        # if insufficient samples then use all that are available
        if n_in_class < samples_per_class:
            size += n_in_class
        else:
            size += samples_per_class

    samples = torch.zeros((histograms.shape[0], size))
    sample_labels = torch.zeros(size + random_sample)

    # keep track of used idxs to randomly sample other data
    sampled_idxs = torch.tensor([])

    # then add the data
    last_idx = 0
    for class_label in classes_to_use:
        class_idxs = np.where(classes == class_label)[0]
        n_in_class = len(class_idxs)
        # if insufficient samples then use all that are available
        if n_in_class < samples_per_class:
            sampled_idxs = torch.cat([sampled_idxs, torch.tensor(class_idxs)])
            samples[:, last_idx: last_idx + n_in_class] = histograms[:, class_idxs]

            # add labels
            sample_labels[last_idx: last_idx + n_in_class] = class_label

            # update last index
            last_idx += n_in_class
        else:
            # pick samples
            sample_idxs = choice(class_idxs, samples_per_class, replace=False)
            sampled_idxs = torch.cat([sampled_idxs, torch.tensor(sample_idxs)])
            samples[:, last_idx: last_idx + samples_per_class] = histograms[:, sample_idxs]

            # add labels
            sample_labels[last_idx: last_idx + samples_per_class] = class_label

            # update last index
            last_idx += samples_per_class

    # truncate sample histogram to eliminate the empty support (i.e. for every index there is some histogram that has
    # non-zero mass there
    #
    # this can be identified by summing the index across the samples and identifying the non-zero entries
    # (since mass is always non-negative)

    if random_sample > 0:
        valid_idxs = [z for z in range(histograms.shape[1]) if z not in sampled_idxs]
        sample_idxs = np.random.choice(valid_idxs, random_sample, replace=False)
        random_samples = histograms[:, sample_idxs]
        samples = torch.cat([samples, random_samples], dim=1)
        sample_labels[last_idx:] = classes[sample_idxs]

    idxs = samples.sum(dim=1) != 0.0
    samples = samples[idxs, :]
    embeddings = embeddings[:, idxs]
    wordsList = wordsList[idxs]

    # preprocess
    # nothing to do here atm
    return samples, sample_labels, embeddings, wordsList


def loadData(fname):
    # read in data from file (loadmat should throw an error if it does not exist)
    mat_contents = sio.loadmat(fname)

    # bbc sports
    if "bbcsport-emd_tr_te_split.mat" in fname:
        data = mat_contents['X'][0]  # each document contains a set of support points (word2vec)
        labels = mat_contents['Y'][0]  # each document's class
        BOW_X = mat_contents['BOW_X'][
            0]  # Same shape as data, the value shows how often a certain word is shown in the document,
        words = mat_contents["words"]
    elif "news20_with_words-test.mat" in fname:
        test = mat_contents['xte'][0]
        train = mat_contents['xtr'][0]
        data = np.concatenate([test, train])
        test_labels = mat_contents['yte'][0]
        train_labels = mat_contents['ytr'][0]
        labels = np.concatenate([test_labels, train_labels])
        BOW_xte = mat_contents['BOW_xte'][0]
        BOW_xtr = mat_contents['BOW_xtr'][0]
        BOW_X = np.concatenate([BOW_xte, BOW_xtr])
        words = mat_contents["words"]

    n_documents = words.shape[1]

    # count unique words
    wordslist = []
    for i in range(n_documents):
        n_words_per_doc = len(words[:, i][0][0])
        for j in range(n_words_per_doc):
            wordslist.append(words[:, i][0][0][j][0])

    wordslist = set(wordslist)
    n_unique_words = len(wordslist)

    # assign each unique word an index and grab the embedding

    embeddings = torch.zeros((300, n_unique_words))
    histograms = torch.zeros((n_unique_words, n_documents))

    # track index of next word to be mapped
    new_word_idx = 0

    # keep track of indexes that words are mapped to
    word_to_idx = {}

    for i in range(n_documents):
        n_words_per_doc = len(words[:, i][0][0])
        for j in range(n_words_per_doc):
            word = words[:, i][0][0][j][0]
            # if word not assigned, give it an assignment
            if word not in word_to_idx:
                word_to_idx[word] = new_word_idx
                new_word_idx += 1

                # also add the embedding
                embeddings[:, word_to_idx[word]] = torch.tensor(data[i][:, j])

            # update document's histogram with relevant word count
            # counts will be normalized afterwards
            histograms[word_to_idx[word], i] = BOW_X[i][0, j]

    # normalize histograms [columns (the histogram for each document) sum to 1]
    histograms /= histograms.sum(dim=0)

    return embeddings, histograms, labels, word_to_idx, np.array(list(wordslist))


def plotTopKWords(ax, hist, words, k):
    sorted_hist, idxs = torch.sort(hist, descending=True)

    hist = sorted_hist[:k]
    words = words[idxs][:k]

    ax.bar(x=range(k), height=hist, tick_label=words)
    ax.tick_params(axis='x', labelrotation=-60)


def plot2DConvexCombinations(ax, D, weights, classes, class_names):
    """
    pin each of the atoms in 2D space and then display each data point as the convex combination of the coordinates of atoms
    based using the barycentric weights

    :param ax:
    :param D:
    :param weights:
    :param classes:
    :param class_names:
    :return:
    """


def sampleKData(X, k, i):
    """
    returns k randomly selected columns of a matrix x while excluding index i

    :param X:
    :param k:
    :return:
    """

    # input validation
    assert len(X.shape) == 2
    assert X.shape[1] >= k and X.shape[1] > i

    idxs = list(range(X.shape[1]))
    idxs.remove(i)

    return X[:, np.random.choice(idxs, k, replace=False)]


def plotComparisonErrors(ax: matplotlib.axes.Axes, errors: torch.Tensor, labels: [str]):
    """
    plots the better errors of

    :param ax:
    :param errors:
    :param labels:
    :return:
    """

    # check that there are only 2 labels and erros
    assert len(labels) == 2
    assert errors.shape[1] == 2

    better1 = errors[:, 0] <= errors[:, 1]
    better2 = better1.logical_not()

    ax.scatter(errors[better1, 0], errors[better1, 1], color="green", label=labels[0])
    ax.scatter(errors[better2, 0], errors[better2, 1], color="red", label=labels[1])


def splitData(X, labels, k, train_size):
    """
    splits the data into a training set, test set, and set of k samples per class observed in labels
    training set uses train_size, and any remaining data (if any) goes to the test set

    :param X:
    :param labels:
    :param k:
    :return:
    """

    train = []
    train_labels = []

    test = []
    test_labels = []

    samples = []
    samples_labels = []

    idx = 0

    for label in labels.unique():
        idx += 1

        label_data = X[:, labels == label]

        # use first k as the k samples
        samples.append(label_data[:, :k])

        train.append(label_data[:, k:k + train_size])

        remaining_data = label_data[:, k + train_size:].shape[1]
        # use remaining data (if any) for test set
        if remaining_data > 0:
            test.append(label_data[:, k + train_size:])
            test_labels.append(torch.ones(remaining_data) * label)

        train_labels.append(torch.ones(train_size) * label)

        samples_labels.append(torch.ones(k) * label)

    train = torch.cat(train, dim=1)
    train_labels = torch.cat(train_labels)

    test = torch.cat(test, dim=1)
    test_labels = torch.cat(test_labels)

    samples = torch.cat(samples, dim=1)
    samples_labels = torch.cat(samples_labels)

    return train, train_labels, test, test_labels, samples, samples_labels


def sampleData(X, labels, k):
    """

    :param X:
    :param labels:
    :param k:
    :return:
    """

    n = X.shape[1]

    assert n >= k, f"k given ({k}) must be less than number of data points to sample ({n})"

    idxs = np.random.choice(n, k, replace=False)

    return X[:, idxs], labels[idxs]
