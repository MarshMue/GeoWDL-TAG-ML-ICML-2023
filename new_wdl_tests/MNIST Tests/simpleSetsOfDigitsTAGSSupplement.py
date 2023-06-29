import torch
from utilities.MNISTutils import MNISTtoHistograms
from utilities.cost import gridCost
from wdl.bregman import barycenter, OT
from wdl.WDL import WDL
from utilities.simpleDistributions import vec2grid, \
    sampleBaryFromDict, matchAtoms
import matplotlib.pyplot as plt
from utilities.visualizations import animateGrayImages
import pickle
import datetime
from matplotlib import rcParams

import argparse

rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

import scipy.io as sio

if __name__ == "__main__":
    dev = torch.device("cpu")
    if torch.cuda.is_available():
        dev = torch.device('cuda')
    elif not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")
        torch.device("cpu")

    else:
        dev = torch.device("mps")

    torch.set_default_dtype(torch.float32)

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_level", type=float)
    args = parser.parse_args()

    #### pick mnist samples
    height, width = 28, 28
    n_atoms_per_digit = 3
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # digits = [0, 1]
    n_digits = len(digits)
    D, _ = MNISTtoHistograms('../../data', digits=digits, samples_per_digit=n_atoms_per_digit,
                             height=height,
                             width=width)

    #### form barycenters data
    maxsinkiters = 50
    reg = 0.003
    barySolver = barycenter(None, "conv", reg=reg, height=height, width=width, maxsinkiter=maxsinkiters)

    # make tensor to store generated digits
    n_samples = 50

    X = torch.zeros((height * width), n_samples * n_digits, device=dev)
    Lambda = torch.zeros((n_atoms_per_digit * n_digits, n_samples * n_digits), device=dev)

    # generate digits as barycenters of the atoms
    for i in range(len(digits)):
        X[:, i * n_samples:(i + 1) * n_samples], Lambda[i * n_atoms_per_digit:(i + 1) * n_atoms_per_digit,
                                                 i * n_samples:(i + 1) * n_samples] = sampleBaryFromDict(
            D[:, i * n_atoms_per_digit:(i + 1) * n_atoms_per_digit], n_samples, barySolver)

    # add noise
    # xmin = X.min()
    # if args.noise_level is not None:
    #     noise = args.noise_level
    #     X += noise * torch.randn_like(X, device=dev)
    #     # make any negative entries small to avoid numerical errors (unsure of what the exact bug is
    #     # happens during backward pass
    #     X = torch.clip(X, min=1e-4)
    #     X /= X.sum(0)

    n_atoms = D.shape[1]

    #### iterate over localities
    localities = [0.0, 0.001, 0.1, 10]
    # localities = [0.0, 0.1]

    ######## Plot Setup #####

    # fig_l, ax_l = plt.subplots(len(localities) + 1, n_atoms,
    #                            figsize=(n_atoms * 4 , (len(localities)) * 4))
    # for i in range(n_atoms):
    #     im = ax_l[0, i].imshow(vec2grid(D[:, i], height=height, width=width))
    #     ax_l[0, i].set_title(f"True Atom {i}")
    #     fig_l.colorbar(im, ax=ax_l[0, i])
    #
    # # plot first 50 coefficients
    # fig_c, ax_c = plt.subplots(len(localities) + 1, 1, figsize=(6, (len(localities) + 1)))
    #
    # n_coef = 50
    # tmp = torch.zeros((n_atoms, n_coef * n_digits))
    # for l in range(n_digits):
    #     tmp[:, l * n_coef:(l + 1) * n_coef] = Lambda[:,
    #                                           l * n_samples:l * n_samples + n_coef]
    # ax_c[0].imshow(tmp)
    # ax_c[0].set_title(f"True Coefficients")

    # plot a few example data points from each set
    n_examples_per_set = 2
    n_examples = n_examples_per_set * n_digits
    fig_r, ax_r = plt.subplots(len(localities) + 1, n_examples,
                               figsize=(4 * (n_examples), 4 * (len(localities) + 1)))
    for set in range(n_digits):
        for example in range(n_examples_per_set):
            im = ax_r[0, set * n_examples_per_set + example].imshow(
                vec2grid(X[:, set * n_samples + example].to("cpu"), height, width), cmap="gray_r")
            ax_r[0, set * n_examples_per_set + example].axis("off")

            # ax_r[0, set * n_examples_per_set + example].set_title(f"True datapoint {i}")
            # fig_r.colorbar(im, ax=ax_r[0, set * n_examples_per_set + example])

    # evolution of atoms plots
    fig_e, ax_e, = plt.subplots(5 + 1, n_atoms, figsize=(4 * n_atoms, 24))

    # loss plots
    fig_l, ax_l = plt.subplots(1, len(localities))
    fig_l.set_size_inches(6.75133, 6.75133 / 4)

    # animations of atoms being learned
    # fig_anim, ax_anim = plt.subplots(len(localities), n_atoms)
    # w_in_inches = 2 * n_atoms + 1
    # h_in_inches = 2 * len(localities) + 2
    # fig_anim.set_size_inches(w_in_inches, h_in_inches, True)

    ###############

    # WDL Params
    dtstr = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if args.noise_level is not None:
        test_name = f"all_digits_{dtstr}_noise{args.noise_level}"
    else:
        test_name = f"all_digits_{dtstr}"
    test_name += "vectorized"
    init_method = "kmeans++-init"
    max_iters = 250
    # max_iters = 50
    lr = 0.25
    n_restarts = 1

    # solver for matching
    OTsolver = OT(gridCost(height, width), method='lp')

    # save assignment costs of atoms and weights
    atom_assign_costs = torch.zeros((len(localities)))
    weights_assign_costs = torch.zeros((len(localities)))

    # # dict to save data for use after running script
    # big_dict = {}
    # for locality in localities:
    #     big_dict[locality] = {}
    #
    # big_dict["True D"] = D.to("cpu")
    # big_dict["True weights"] = Lambda.to("cpu")
    # big_dict["max iter"] = max_iters
    # big_dict["max sink iter"] = maxsinkiters
    # big_dict["restarts"] = n_restarts
    # big_dict["lr"] = lr

    # # video of atoms
    # frames = []
    # # labels for plots in video
    # labels = []
    for row, j in enumerate(range(len(localities))):
        # labels.append([])

        mu = localities[j]
        #### WDL
        print(f"--------- Locality: {mu} ---------")

        # clear cuda cache
        torch.cuda.empty_cache()

        wdl = WDL(n_atoms=len(digits) * n_atoms_per_digit, dev=dev)
        # debugging:
        # wdl = WDL(n_atoms=2, dev=dev)

        weights, log = wdl.fit(X=X,
                               C=None,
                               init_method=init_method,
                               loss_method="conv",
                               bary_method="conv",
                               height=height,
                               width=width,
                               reg=reg,
                               mu=mu,
                               max_iters=max_iters,
                               max_sinkhorn_iters=maxsinkiters,
                               weightUpdate="joint",
                               dictionaryUpdate="joint",
                               jointOptimizer=torch.optim.Adam,
                               jointOptimKWargs={"lr": lr},
                               verbose=False,
                               n_restarts=n_restarts,
                               log_iters=1,
                               log=True,
                               )
        weights = weights.to("cpu")

        DLearned, weights, cost, assignments, old_assignments = matchAtoms(wdl.D.detach().clone(), D, weights,
                                                                           OTsolver=OTsolver,
                                                                           return_assignments=True)
        atom_assign_costs[j] = cost

        weights = weights.to("cpu")
        DLearned = DLearned.to("cpu")
        Lambda = Lambda.to("cpu")

        #### display and save results
        weights_assign_costs[j] = torch.linalg.norm(weights - Lambda) / weights.shape[1]

        for set in range(n_digits):
            for example in range(n_examples_per_set):
                b = barySolver(DLearned, weights[:, set * n_samples + example])
                im = ax_r[row + 1, set * n_examples_per_set + example].imshow(
                    vec2grid(b.to("cpu"), height, width), cmap="gray_r")
                # ax_r[0, set * n_examples_per_set + example].set_title(f"True datapoint {i}")
                # fig_r.colorbar(im, ax=ax_r[0, set * n_examples_per_set + example])
                ax_r[row + 1, set * n_examples_per_set + example].axis("off")

        ax_l[row].plot(range(1, max_iters + 1), log["loss"].cpu())

        if mu == 0.1:
            # true atoms
            for atom_idx in range(n_atoms):
                ax_e[0, atom_idx].imshow(vec2grid(D[:, atom_idx].to("cpu"), height=height, width=width), cmap="gray_r")
                ax_e[0, atom_idx].axis("off")

            # several iterations of atoms
            for e_row, iter in enumerate([10, 20, 30, 40, 50]):
                tmpD = log["atoms"][iter - 1]
                tmpD[:, assignments] = tmpD[:, old_assignments]
                for atom_idx in range(n_atoms):
                    ax_e[e_row + 1, atom_idx].imshow(vec2grid(tmpD[:, atom_idx].to("cpu"), height=height, width=width),
                                                     cmap="gray_r")
                    ax_e[e_row + 1, atom_idx].axis("off")

        """
        frames.append(log["atoms"])
        for atom in range(n_atoms):
            labels[j].append(f"atom {atom} mu={mu}")
        """

        """
        tmp = torch.zeros((n_atoms, n_coef * n_digits))
        for l in range(n_digits):
            tmp[:, l * n_coef:(l + 1) * n_coef] = weights[:,
                                                  l * n_samples:l * n_samples + n_coef]

        ax_c[j + 1].imshow(tmp.to("cpu"))
        ax_c[j + 1].set_title(f"Learned Coefficients, locality={mu:.3f}")

        # plot atoms
        for i in range(n_atoms):
            im = ax_l[j + 1, i].imshow(vec2grid(DLearned[:, i].to("cpu"), height=height, width=width))
            ax_l[j + 1, i].set_title(f"Learned Atom {i}, locality={mu:.3f}")
            fig_l.colorbar(im, ax=ax_l[j + 1, i])

        for set in range(n_digits):
            for example in range(n_examples_per_set):
                b = barySolver(DLearned, weights[:, n_samples * set + example])
                im = ax_r[j + 1, set * n_examples_per_set + example].imshow(vec2grid(b.to("cpu"), height, width))
                ax_r[j + 1, set * n_examples_per_set + example].set_title(f"Est. datapoint {i}, locality={mu:.3f}")
                fig_r.colorbar(im, ax=ax_r[j + 1, set * n_examples_per_set + example])
        """

    #     big_dict[mu]["Learned D"] = DLearned.to("cpu")
    #     big_dict[mu]["Learned weights"] = weights.to("cpu")
    #
    # with open(f"NeurReps2022Data/{test_name}.pkl", "wb") as f:
    #     pickle.dump(big_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # anim = animateGrayImages(fig_anim, ax_anim, labels=labels, frames=frames, height=height, width=width)
    # fig_anim.suptitle(f"(Atom init: {init_method}, max iters: {max_iters}, restarts: {n_restarts})")
    # fig_anim.tight_layout()
    # anim.save(
    #     f"videos/{test_name}_(Atom_init={init_method}-max_iters={max_iters}-restarts={n_restarts}_lr={lr}).mp4", \
    #     dpi=150)
    #
    # fig_c.suptitle(
    #     f"Learned vs True Coefficients (Atom init: {init_method}, max iters: {max_iters}, restarts: {n_restarts} lr={lr})")
    # fig_c.tight_layout(rect=[0, 0.03, 1, 0.95])
    # fig_c.savefig(
    #     f"images/{test_name}_manual-coefficients(Atom_init={init_method},max_iters={max_iters},restarts={n_restarts}lr={lr}).png",
    #     dpi=150)

    # fig_r.suptitle(
    #     f"Sample of reconstructed data (Atom init: {init_method}, max iters: {max_iters}, restarts: {n_restarts} lr={lr})")
    # fig_r.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_r.tight_layout()
    fig_r.savefig(
        f"images/{test_name}_TAGS_supp_recon.pdf")

    fig_e.tight_layout()
    fig_e.savefig(f"images/{test_name}_TAGS_supp_atom_evolution.pdf")

    fig_l.tight_layout()
    fig_l.savefig(f"images/{test_name}_TAGS_loss.pdf")

    # plt.show()

    # fig_l.suptitle(
    #     f"True vs Learned Atoms (Atom init: {init_method}, max iters: {max_iters}, restarts: {n_restarts})")
    # fig_l.tight_layout(rect=[0, 0.03, 1, 0.95])
    # fig_l.savefig(
    #     f"images/{test_name}_manual-atoms_(Atom_init={init_method},max_iters={max_iters},restarts={n_restarts}lr={lr}).png",
    #     dpi=150)
    #
    # # plot assignment costs
    # fig_assign, ax_assign = plt.subplots(2)
    # w_in_inches = 6
    # h_in_inches = 6
    # fig_assign.set_size_inches(w_in_inches, h_in_inches, True)
    #
    # ax_assign[0].semilogx(localities, atom_assign_costs)
    # ax_assign[0].set_title("Assignment costs (True OT between atoms)")
    # ax_assign[0].set_xlabel("Locality")
    #
    # ax_assign[1].semilogx(localities, weights_assign_costs)
    # ax_assign[1].set_title("Frobenius norm of weights difference")
    # ax_assign[1].set_xlabel("Locality")
    #
    # fig_assign.suptitle(f"(Atom init: {init_method}, max iters: {max_iters}, restarts: {n_restarts}_lr={lr})")
    # fig_assign.tight_layout(rect=[0, 0.03, 1, 0.95])
    # fig_assign.savefig(
    #     f"images/{test_name}_assignmentcosts(Atom_init={init_method}-max_iters={max_iters}-restarts={n_restarts}_lr={lr}).png",
    #     dpi=150)
