import torch
from utilities.NLPUtils import loadWordHistograms, plotTopKWords, sampleKData, plotComparisonErrors
from wdl.bcm import getBCMweights
from wdl.WDL import WDL
from wdl.bregman import barycenter, OT
from ot.utils import dist
import time
import matplotlib.pyplot as plt

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)

    # Load word histograms with n samples from each class
    mat_fname = '../../data/WMD_datasets/bbcsport-emd_tr_te_split.mat'

    samples_per_class = 20

    n_exemplar_samples = 10

    X, labels, embeddings, words = loadWordHistograms(samples_per_class=samples_per_class, fname=mat_fname)

    n_data = X.shape[1]

    localities = [0.0] + [10 ** x for x in range(-3, 3)]

    ### WDL PARAMS ###
    k = 10

    init_method = "rand"

    reg = 1e-1

    max_iters = 300

    max_sinkhorn_iters = 5

    lr = 0.25

    n_restarts = 1
    ##################

    # make cost matrix
    C = dist(embeddings.T, embeddings.T)

    barySolver = barycenter(C=C, method="bregman", reg=reg, maxsinkiter=max_sinkhorn_iters)
    otSolver = OT(C=C, method="bregman", reg=reg, maxiter=max_sinkhorn_iters)

    print(f"ground dim = {C.shape[0]}")

    start = time.time()
    for j in range(len(localities)):
        μ = localities[j]

        print(f"--------- Locality: {μ} ---------")

        wdl = WDL(n_atoms=k)

        weights, log = wdl.fit(X=X,
                               C=C,
                               init_method=init_method,
                               loss_method="bregman",
                               bary_method="bregman",
                               reg=reg,
                               mu=μ,
                               max_iters=max_iters,
                               max_sinkhorn_iters=max_sinkhorn_iters,
                               update_method="adam",
                               lr=lr,
                               verbose=True,
                               n_restarts=n_restarts,
                               log_iters=5,
                               log=True,
                               )

        print(f"Elapsed time locality = {μ} with {n_restarts} restarts: {time.time() - start} (s)")

        fig, ax = plt.subplots()
        ax.imshow(weights)
        ax.set_title(f"μ={μ:.3f} weights")
        fig.savefig(f"exemplar_images/μ={μ:.3f}weights_{max_iters}_iters_{n_restarts}_rstrts.png")

        K = 50
        fig, ax = plt.subplots(k, 1, figsize=(K // 2, 4 * k))
        for i in range(k):
            plotTopKWords(ax[i], wdl.D[:, i], words, K)

        fig.tight_layout()

        fig.savefig(f"exemplar_images/atoms_μ={μ:.3f}_{max_iters}_iters_{n_restarts}_rstrts.png")

        dict_errors = torch.zeros((n_data))
        exemplar_errors_same_class = torch.zeros((n_data))
        exemplar_errors_any_class = torch.zeros((n_data))

        # need to track which index of the subset of each label has currently been used to prevent it from being picked
        # in the BCM model
        curr_idxs = {}
        for label in labels.unique():
            curr_idxs[int(label)] = 0

        # sample exemplars from data
        for i in range(n_data):
            dict_recon = barySolver(wdl.D, weights[:, i])
            dict_errors[i] = otSolver(X[:, i], dict_recon)

            # sample exemplars from same class
            for _ in range(n_exemplar_samples):
                D = sampleKData(X[:, labels == labels[i]], k, curr_idxs[int(labels[i])])
                bcm_weights = torch.tensor(getBCMweights(D, X[:, i], embeddings, reg), dtype=torch.get_default_dtype())
                exemplar_recon = barySolver(D, bcm_weights)
                exemplar_errors_same_class[i] += otSolver(X[:, i], exemplar_recon)[0]

            # increment current index of that label
            curr_idxs[int(labels[i])] += 1

            # sample exemplars from any class
            for _ in range(n_exemplar_samples):
                D = sampleKData(X, k, i)
                bcm_weights = torch.tensor(getBCMweights(D, X[:, i], embeddings, reg), dtype=torch.get_default_dtype())
                exemplar_recon = barySolver(D, bcm_weights)
                exemplar_errors_any_class[i] += otSolver(X[:, i], exemplar_recon)[0]

        # average errors
        exemplar_errors_same_class /= n_exemplar_samples
        exemplar_errors_any_class /= n_exemplar_samples

        print(f"average error {dict_errors.mean()}")

        # compare reconstruction error of atoms vs average expected error

        compare_same_class = torch.stack([dict_errors, exemplar_errors_same_class], dim=1)
        compare_any_class = torch.stack([dict_errors, exemplar_errors_any_class], dim=1)

        # plot error comparisons
        fig, ax = plt.subplots(2, 1, figsize=(6, 8))

        ### plot for same class exemplars
        plotComparisonErrors(ax[0], compare_same_class, ["Dictionary", "Exemplars"])
        ax[0].set_xlabel("WDL error")
        ax[0].set_ylabel(f"Exemplar error (avg over {n_exemplar_samples} samples)")
        ax[0].set_title(f"Reconstruction errors using using {k} learned atoms or exemplars of the same class")

        maxval = max(
            [dict_errors.min(), dict_errors.max(), exemplar_errors_same_class.min(), exemplar_errors_same_class.max()])
        x = [-maxval, maxval]
        y = [-maxval, maxval]

        zeros = [0, 0]
        # add axes and y=x
        ax[0].plot(x, y, color="red")
        ax[0].plot(zeros, y, color="black")
        ax[0].plot(x, zeros, color="black")
        ax[0].legend()

        ### plot for any class exemplars
        ax[1].set_xlabel("WDLerror")
        ax[1].set_ylabel(f"Exemplar error (avg over {n_exemplar_samples} samples)")
        ax[1].set_title(f"Reconstruction errors using {k} learned atoms or exemplars of the any class")

        plotComparisonErrors(ax[1], compare_any_class, ["Dictionary", "Exemplars"])

        maxval = max(
            [dict_errors.min(), dict_errors.max(), exemplar_errors_any_class.min(), exemplar_errors_any_class.max()])
        x = [-maxval, maxval]
        y = [-maxval, maxval]

        # add axes and y=x
        ax[1].plot(x, y, color="red")
        ax[1].plot(zeros, y, color="black")
        ax[1].plot(x, zeros, color="black")
        ax[1].legend()

        fig.tight_layout
        fig.savefig(f"exemplar_images/error_comparison_μ={μ:.3f}_{max_iters}_iters_{n_restarts}_rstrts.png", dpi=150)

    plt.show()
