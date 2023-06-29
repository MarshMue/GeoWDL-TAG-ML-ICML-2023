import torch
import matplotlib.pyplot as plt
import pickle
from utilities.simpleDistributions import vec2grid
import matplotlib
import tikzplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

if __name__ == "__main__":
    plt.rcParams["font.family"] = "Times New Roman"
    fname = "results/data.pkl"

    with open(fname, "rb") as f:
        df = pickle.load(f)

    for key in df:
        print(key, df[key])

    true_D = df["True D"]
    true_w = df["True weights"]

    max_iter = df["max iter"]
    max_sink_iter = df["max sink iter"]
    restarts = df["restarts"]
    lr = df["lr"]

    locality_Ds_and_ws = {}
    max_val = 0
    for key in df:
        if type(key) is float or type(key) is int:
            locality_Ds_and_ws[key] = {}
            locality_Ds_and_ws[key]["D"] = df[key]["Learned D"]
            locality_Ds_and_ws[key]["w"] = df[key]["Learned weights"]

    # confusion matrix type plot of weights plot
    n_plots = len(locality_Ds_and_ws)
    fig, ax = plt.subplots(1, n_plots, figsize=(4 * n_plots + 2, 6), constrained_layout=True)
    weight_sums = torch.zeros((n_plots, 10, 10))
    for k, locality in enumerate(sorted(locality_Ds_and_ws)):

        for i in range(10):
            for j in range(i, 10):
                weight_sums[k, i, j] = weight_sums[k, j, i] = locality_Ds_and_ws[locality]["w"][i * 3:(i + 1) * 3,
                                                              50 * j:(j + 1) * 50].sum() / 50
        if weight_sums[k].max() > max_val:
            max_val = weight_sums[k].max()

    for k, locality in enumerate(sorted(locality_Ds_and_ws)):
        im = ax[k].imshow(weight_sums[k], vmin=0, vmax=1, cmap="gray_r")
        diag = torch.trace(weight_sums[k])
        total = weight_sums[k].sum()
        ax[k].set_title(r"$\rho=$" + f"{locality}, trace / sum: {diag / total:0.3f}")

    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.65)

    # fig.set_size_inches(w=3.25063, h=1.5)
    fig.savefig("images/" + fname.split("/")[1][:-4] + ".pdf", bbox_inches="tight")
    # tikzplotlib.save("NeurReps2022Data/images/" + fname.split("/")[1][:-4] + ".tex", figure=fig)

    # plot the True and Learned atoms

    fig_a, ax_a = plt.subplots((n_plots + 1), 30, figsize=(30, (n_plots + 1) * 1), constrained_layout=True,
                               gridspec_kw={'wspace': 0, 'hspace': 0})

    for i in range(30):
        ax_a[0, i].imshow(vec2grid(true_D[:, i], height=28, width=28), cmap="gray_r")
        ax_a[0, i].axis('off')
        # ax_a[0, i].tick_params(which="both", bottom=False, top=False, left=False, right=False,
        # labelbottom=False, labelleft=False)

    for k, locality in enumerate(sorted(locality_Ds_and_ws)):
        learned_D = locality_Ds_and_ws[locality]["D"]
        for i in range(30):
            ax_a[k + 1, i].imshow(vec2grid(learned_D[:, i], height=28, width=28), cmap="gray_r")
            ax_a[k + 1, i].axis('off')
            # ax_a[k + 1, i].tick_params(which="both", bottom=False, top=False, left=False, right=False,
            #                           labelbottom=False, labelleft=False)

    fig_a.tight_layout()
    # fig_a.set_size_inches(w=3.25063, h=1.5)
    fig_a.savefig("images/" + fname.split("/")[1][:-4] + "atoms" + ".pdf", bbox_inches="tight")
    # tikzplotlib.save("NeurReps2022Data/images/" + fname.split("/")[1][:-4] + "atoms.tex", figure=fig_a)

    # histograms of counts among all data at each locality level with number of bins that have x% of the mass

    fig_c, ax_c = plt.subplots(1, n_plots, figsize=(4 * n_plots + 2, 4), constrained_layout=True)
    x = 0.95
    for k, locality in enumerate(sorted(locality_Ds_and_ws)):
        learned_w = locality_Ds_and_ws[locality]["w"]
        counts = []
        for i in range(learned_w.shape[1]):
            sorted, _ = learned_w[:, i].sort(descending=True)
            count = 1
            while sorted[:count].sum() <= x:
                count += 1
            counts.append(count)

        ax_c[k].hist(counts, bins=list(range(1, 31)), color="black")
        ax_c[k].set_title(r"$\rho=$" + f"{locality}")

    # fig_c.set_size_inches(w=3.25063, h=1.5)
    fig_c.savefig("images/" + fname.split("/")[1][:-4] + f"hists_counts_>{x:0.2f}.pdf",
                  bbox_inches="tight")
    # tikzplotlib.save("NeurReps2022Data/images/" + fname.split("/")[1][:-4] + f"hists_counts_>{x:0.2f}.tex",
    # figure=fig_c)

    # plt.show()
