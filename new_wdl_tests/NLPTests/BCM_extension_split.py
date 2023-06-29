import torch
from utilities.NLPUtils import loadWordHistograms, splitData
from wdl.bcm import getBCMweights
from wdl.WDL import WDL
from wdl.bregman import barycenter, OT
from ot.utils import dist
import matplotlib.pyplot as plt
import argparse
import pickle
import time
from tqdm import tqdm

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int)
    parser.add_argument("--locality", type=float)
    parser.add_argument("--k", type=int)

    args = parser.parse_args()

    print(f"HPC Trial: {args.trial} (if not none)")

    # torch setup
    torch.set_default_dtype(torch.float32)

    dev = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    ### WDL PARAMS ###
    init_method = "rand"

    reg = 1e-1

    max_iters = 300

    max_sinkhorn_iters = 25

    lr = 0.25

    n_restarts = 1

    mu = args.locality
    ##################

    # setup trial information
    n_trials = 1

    n_test_samples = 100

    k = args.k
    k_idx = 0
    n_ks = 1

    # pick class based on nearest neighbor
    oneNN_accuracy_e = torch.zeros((n_trials, n_ks), device=dev)
    # pick class based on minimum average distance to other classes
    mad_accuracy_e = torch.zeros((n_trials, n_ks), device=dev)
    # pick class that achieves the smallest barycentric loss
    mbl_accuracy_e = torch.zeros((n_trials, n_ks), device=dev)
    # pick class that has the maximum coordinate
    mc_accuracy_e = torch.zeros((n_trials, n_ks), device=dev)

    # pick class that achieves the smallest barycentric loss using the loss from the QP
    mbl_accuracy_l = torch.zeros((n_trials, n_ks), device=dev)

    # pick class based on nearest neighbor but using the dictionaries as references
    oneNN_d_accuracy_e = torch.zeros((n_trials, n_ks), device=dev)
    # pick class based on minimum average distance to other classes but using the dictionaries as references
    mad_d_accuracy_e = torch.zeros((n_trials, n_ks), device=dev)
    # same as mbl but using the dictionaries as references
    mbl_d_accuracy_e = torch.zeros((n_trials, n_ks), device=dev)
    # same as mc but using the dictionaries as references
    mc_d_accuracy_e = torch.zeros((n_trials, n_ks), device=dev)

    # same as mbl but using the dictionaries as references and using the loss from the qp
    mbl_d_accuracy_l = torch.zeros((n_trials, n_ks), device=dev)

    for trial in range(n_trials):
        print(f"############# Trial {trial + 1} of {n_trials} #############\n")

        # sample data:
        # Load word histograms with n samples from each class
        # BBC Sport 5

        mat_fname = '../../data/WMD_datasets/bbcsport-emd_tr_te_split.mat'

        # 4 * k examples per class to learn a dictionary from
        n_dict_train = 4 * k

        # k examples to use as in original BCM experiments for comparison

        n_exemplar_samples = k

        samples_per_class = n_dict_train + n_exemplar_samples

        X, labels, embeddings, words = loadWordHistograms(samples_per_class=samples_per_class,
                                                          random_sample=n_test_samples,
                                                          fname=mat_fname)

        # move data to gpu
        X = X.to(dev)
        embeddings = embeddings.to(dev)

        ulabels = labels.unique()
        nlabels = ulabels.shape[0]

        # make cost matrix
        C = dist(embeddings.T, embeddings.T).to(dev)
        barySolver = barycenter(
            C=C, method="bregman", reg=reg, maxsinkiter=max_sinkhorn_iters, dev=dev)
        otSolver = OT(C=C, method="bregman", reg=reg,
                      maxiter=max_sinkhorn_iters, dev=dev)

        train, train_labels, test, test_labels, samples, samples_labels = splitData(
            X, labels, k, n_dict_train)

        # learn dictionary for each class
        Ds = []
        for label in tqdm(labels.unique()):
            data = train[:, train_labels == label]

            wdl = WDL(n_atoms=args.k)

            weights, log = wdl.fit(X=data,
                                   C=C,
                                   init_method=init_method,
                                   loss_method="bregman",
                                   bary_method="bregman",
                                   weight_update="bcm",
                                   support=embeddings.T,
                                   reg=reg,
                                   mu=mu,
                                   max_iters=max_iters,
                                   max_sinkhorn_iters=max_sinkhorn_iters,
                                   update_method="adam",
                                   lr=lr,
                                   verbose=True,
                                   n_restarts=n_restarts,
                                   log_iters=5,
                                   log=True,
                                   dev=dev
                                   )

            Ds.append(wdl.D.clone())

        Ds = torch.cat(Ds, dim=1)

        ###### perform accuracy tests ######
        print("Accuracy tests")
        for i in tqdm(range(test.shape[1])):
            distances_e = torch.zeros(samples.shape[1])

            # one nearest neighbor
            for j in range(samples.shape[1]):
                distances_e[j] = otSolver(test[:, i], samples[:, j])

            min_idx_e = distances_e.argmin()

            # check label agreement
            oneNN_accuracy_e[trial,
                             k_idx] += test_labels[i] == samples_labels[min_idx_e]

            # minimum average distance
            avg_distances_e = torch.zeros(nlabels)

            idx = 0
            for label_idx in range(nlabels):
                label = ulabels[label_idx]
                avg_distances_e[label_idx] = distances_e[samples_labels == label].mean(
                )

            min_label_e = ulabels[avg_distances_e.argmin()]

            mad_accuracy_e[trial, k_idx] += test_labels[i] == min_label_e

            # minimum barycenter loss
            recon_loss_e = torch.zeros(nlabels)
            recon_loss_l = torch.zeros(nlabels)

            for label_idx in range(nlabels):
                label = ulabels[label_idx]
                D = samples[:, samples_labels == label]
                bcm_weights, QP_loss = getBCMweights(
                    D, test[:, i], embeddings.T, reg, return_val=True)
                bcm_weights = torch.tensor(
                    bcm_weights, dtype=torch.get_default_dtype()).to(dev)
                exemplar_recon = barySolver(D, bcm_weights)
                recon_loss_e[label_idx] = otSolver(
                    test[:, i], exemplar_recon)
                recon_loss_l[label_idx] = QP_loss

            min_label_e = ulabels[recon_loss_e.argmin()]
            min_label_l = ulabels[recon_loss_l.argmin()]

            mbl_accuracy_e[trial, k_idx] += test_labels[i] == min_label_e
            mbl_accuracy_l[trial, k_idx] += test_labels[i] == min_label_l

            # maximum coordinate
            bcm_weights = torch.tensor(getBCMweights(samples, test[:, i], embeddings.T, reg, return_val=False),
                                       dtype=torch.get_default_dtype()).to(dev)

            max_mass = torch.zeros(ulabels.shape[0])
            for j, label in enumerate(ulabels):
                max_mass[j] = bcm_weights[samples_labels == label].sum()

            max_label = ulabels[max_mass.argmax()]

            mc_accuracy_e[trial, k_idx] += test_labels[i] == max_label

            ### dictionary equivalents ###
            # note samples_lables are the same for the dictionaries

            n_d = samples.shape[1]

            distances_e = torch.zeros(n_d)

            # one nearest neighbor
            for j in range(n_d):
                distances_e[j] = otSolver(test[:, i], Ds[:, j])

            min_idx_e = distances_e.argmin()

            # check label agreement
            oneNN_d_accuracy_e[trial,
                               k_idx] += test_labels[i] == samples_labels[min_idx_e]

            # minimum average distance
            avg_distances_e = torch.zeros(nlabels)

            idx = 0
            for label_idx in range(nlabels):
                label = ulabels[label_idx]
                avg_distances_e[label_idx] = distances_e[samples_labels == label].mean(
                )

            min_label_e = ulabels[avg_distances_e.argmin()]

            mad_d_accuracy_e[trial, k_idx] += test_labels[i] == min_label_e

            # minimum barycenter loss w/ dictionaries
            recon_loss_e = torch.zeros(nlabels)
            recon_loss_l = torch.zeros(nlabels)

            for label_idx in range(nlabels):
                label = ulabels[label_idx]
                D = Ds[:, samples_labels == label]
                bcm_weights, QP_loss = getBCMweights(
                    D, test[:, i], embeddings.T, reg, return_val=True)

                bcm_weights = torch.tensor(
                    bcm_weights, dtype=torch.get_default_dtype()).to(dev)
                exemplar_recon = barySolver(D, bcm_weights)
                recon_loss_e[label_idx] = otSolver(
                    test[:, i], exemplar_recon)
                recon_loss_l[label_idx] = QP_loss

            min_label_e = ulabels[recon_loss_e.argmin()]
            min_label_l = ulabels[recon_loss_l.argmin()]

            mbl_d_accuracy_e[trial, k_idx] += test_labels[i] == min_label_e
            mbl_d_accuracy_l[trial, k_idx] += test_labels[i] == min_label_l

            # maximum coordinate w/ dictionaries
            bcm_weights = torch.tensor(getBCMweights(Ds, test[:, i], embeddings.T, reg, return_val=False),
                                       dtype=torch.get_default_dtype()).to(dev)

            max_mass = torch.zeros(ulabels.shape[0])
            for j, label in enumerate(ulabels):
                max_mass[j] = bcm_weights[samples_labels == label].sum()

            max_label = ulabels[max_mass.argmax()]

            mc_d_accuracy_e[trial, k_idx] += test_labels[i] == max_label

        # normalize accuracies over number of test items
        n_test = test.shape[1]
        oneNN_accuracy_e[trial, k_idx] /= n_test

        mad_accuracy_e[trial, k_idx] /= n_test

        mbl_accuracy_e[trial, k_idx] /= n_test
        mbl_accuracy_l[trial, k_idx] /= n_test

        mc_accuracy_e[trial, k_idx] /= n_test

        oneNN_d_accuracy_e[trial, k_idx] /= n_test

        mad_d_accuracy_e[trial, k_idx] /= n_test

        mbl_d_accuracy_e[trial, k_idx] /= n_test
        mbl_d_accuracy_l[trial, k_idx] /= n_test

        mc_d_accuracy_e[trial, k_idx] /= n_test

    data_dict = {}
    data_dict["1NN - Data - e"] = oneNN_accuracy_e.to("cpu")

    data_dict["Mean Avg. Dist. - Data - e"] = mad_accuracy_e.to("cpu")

    data_dict["Min BC Loss - Data - e"] = mbl_accuracy_e.to("cpu")
    data_dict["Min BC Loss - Data - l"] = mbl_accuracy_l.to("cpu")

    data_dict["Max Coordinate - Data"] = mc_accuracy_e.to("cpu")

    data_dict["1NN - Dict. - e"] = oneNN_d_accuracy_e.to("cpu")

    data_dict["Mean Avg. Dist. - Dict. - e"] = mad_d_accuracy_e.to("cpu")

    data_dict["Min BC Loss - Dict. - e"] = mbl_d_accuracy_e.to("cpu")
    data_dict["Min BC Loss - Dict. - l"] = mbl_d_accuracy_l.to("cpu")

    data_dict["Max Coordinate - Dict."] = mc_d_accuracy_e.to("cpu")

    with open(f'data_outputs/{args.exp}_{args.trial}mu={mu:.3f}.pkl', 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"total Script elapsed time: {time.time() - start_time} (s)")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
