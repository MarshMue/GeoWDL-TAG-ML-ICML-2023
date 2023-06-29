import warnings

import torch
import pickle
import glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import tikzplotlib
import numpy as np

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import argparse
from datetime import datetime
import re

if __name__ == "__main__":
    plt.rcParams["font.family"] = "Times New Roman"
    parser = argparse.ArgumentParser()
    parser.add_argument("--kmin", type=int)
    parser.add_argument("--kmax", type=int)
    parser.add_argument("--prefix", type=str)

    args = parser.parse_args()

    assert args.kmax >= args.kmin, "kmax should be larger than kmin"

    n_ks = args.kmax - args.kmin + 1

    locality_data_dicts = {}
    # all_data_dict = {}

    # count data per each locality
    num_data = {}

    # whether to make each plot an individual file or not
    individual = True

    for file in glob.glob("data_outputs/" + args.prefix + "*.pkl"):
        if not '""' in file and (args.prefix == "" and "_" in file.split("/")[1]):
            # skip the news data
            continue

        used = True
        locality = float(file.split("mu=")[1].split(".pkl")[0])
        with open(file, 'rb') as handle:
            data_dict = pickle.load(handle)
            # each key is what classification method was used
            for key in data_dict:
                # ensure only tests with valid k range are used
                if data_dict[key].shape[1] != n_ks:
                    used = False
                    break

                # add the results to a list
                # check if locality in the dictionary
                if locality in locality_data_dicts:
                    # add method data
                    if key in locality_data_dicts[locality]:
                        locality_data_dicts[locality][key].append(data_dict[key])
                    else:
                        locality_data_dicts[locality][key] = [data_dict[key]]
                else:
                    # no locality added, make sub dictionary for each method
                    locality_data_dicts[locality] = {}
                    # add method data
                    locality_data_dicts[locality][key] = [data_dict[key]]

        if used:
            if locality in num_data:
                num_data[locality] += 1
            else:
                num_data[locality] = 1

    # check that all localities have an equal number of trials
    placeholder = num_data[next(iter(num_data))]
    for locality in num_data:
        print(f"Number of samples for locality {locality}: {num_data[locality]}")
        if num_data[locality] != placeholder:
            warnings.warn("Unequal numbers of samples between localities")

    # setup figures

    if individual:
        templist = [plt.subplots() for i in range(5)]
        fig_exp, ax_exp = [templist[i][0] for i in range(len(templist))], [templist[i][1] for i in range(len(templist))]
        del templist
    else:
        # plots for separating methods
        fig_exp, ax_exp = plt.subplots(1, 5, figsize=(16, 3), dpi=300)

    # string matching patterns
    NNe = re.compile("1NN.*e")

    MADe = re.compile("Mean.*e")

    MINe = re.compile("Min.*e")

    MINl = re.compile("Min.*l")

    MAX = re.compile("Max.*")

    # setup colorbar
    # need number of localities and then to sort them to fit the colorbar
    n_colors = len(num_data)
    localities = torch.tensor(list(num_data.keys()))

    localities, _ = torch.sort(localities)

    # plots for separating locality
    if individual:
        templist = [plt.subplots() for i in range(localities.shape[0])]
        fig_loc, ax_loc = [templist[i][0] for i in range(len(templist))], [templist[i][1] for i in range(len(templist))]
        del templist
    else:
        fig_loc, ax_loc = plt.subplots(1, localities.shape[0], figsize=(16, 3), dpi=300)

    viridis = cm.get_cmap('viridis', n_colors)

    min_y = 1
    max_y = 0

    # make dict for individual plots for a given locality level and method
    ind_plots = {}

    for locality in locality_data_dicts:

        if locality not in ind_plots:
            ind_plots[locality] = {}

        curr_ax_loc = ax_loc[int(torch.where(localities == locality)[0])]

        for key in locality_data_dicts[locality]:
            # combine the results of each trial and take the average over k
            data = torch.vstack(locality_data_dicts[locality][key])
            min = data.mean(0).min()
            max = data.mean(0).max()

            if min < min_y:
                min_y = min
            if max > max_y:
                max_y = max

            color_index = torch.where(localities == locality)
            color = viridis(color_index)

            if key[-1] == "w":
                continue

            if "Data" in key:
                curr_ax = 0
                style = "--"
                label = "Data"
                ind_color = "tomato"
            else:
                curr_ax = 1
                style = "-"
                label = "Dict"
                ind_color = "turquoise"

            if NNe.match(key):
                curr_ax_sep = ax_exp[0]
                loc_color = "tomato"
            elif MADe.match(key):
                curr_ax_sep = ax_exp[1]
                loc_color = "orange"
            elif MINe.match(key):
                curr_ax_sep = ax_exp[2]
                loc_color = "blue"
            elif MINl.match(key):
                curr_ax_sep = ax_exp[3]
                loc_color = "purple"
            elif MAX.match(key):
                curr_ax_sep = ax_exp[4]
                loc_color = "turquoise"

            # many redundant settings, but it's fine
            if MINl.match(key):
                method = key.split(" -")[0] + " (QP)"
            else:
                method = key.split(" -")[0]

            if not individual:
                curr_ax_sep.set_title(method)

            if method not in ind_plots[locality]:
                fig, ax = plt.subplots()
                ind_plots[locality][method] = (fig, ax)
            else:
                (fig, ax) = ind_plots[locality][method]

            # plot data vs dict
            curr_ax_sep.plot(range(args.kmin, args.kmax + 1), data.mean(0), color=color,
                             label=r"$\rho=$ " + f"{locality:.2f}",
                             linestyle=style)
            # curr_ax_sep.fill_between(range(args.kmin, args.kmax + 1),
            #                         data.mean(0) - data.std(0), data.mean(0) + data.std(0),
            #                         color=color, alpha=0.2)

            # plot in locality plot
            curr_ax_loc.plot(range(args.kmin, args.kmax + 1), data.mean(0),
                             label=method, color=loc_color,
                             linestyle=style)

            ax.plot(range(args.kmin, args.kmax + 1), data.mean(0),
                    label=method, color=ind_color,
                    linestyle=style)
            ax.fill_between(range(args.kmin, args.kmax + 1),
                            data.mean(0) - data.std(0), data.mean(0) + data.std(0),
                            color=ind_color, alpha=0.2)

    for i in range(len(ax_exp)):
        # ax_exp[i].legend()
        ax_exp[i].set_ylim([min_y, 1])
        ax_exp[i].set_yticks(np.linspace(0.5, 1, 6))
        ax_exp[i].set_xticks(list(range(2, 13, 2)))

    for i, locality in enumerate(localities):
        ax_loc[i].set_ylim([min_y, 1])
        if not individual:
            ax_loc[i].set_title(r"$\rho=$" + f"{locality:.2f}")
        ax_loc[i].set_xticks(list(range(2, 13, 2)))
        ax_loc[i].set_yticks(np.linspace(0.5, 1, 6))

    for key in num_data:
        print(key, num_data[key])

    handles, labels = ax_exp[-1].get_legend_handles_labels()

    # only label the solid lines
    handles = handles[1::2]
    labels = labels[1::2]

    labels, handles = zip(*sorted(zip(labels, handles)))
    legend_x = 0.88

    if not individual:

        fig_exp.legend(handles, labels, loc=(legend_x, 0.5))

        fig_exp.tight_layout(rect=[0, 0, legend_x, 1])

        fig_exp.savefig("bcm_comparison/" + args.prefix + "experiments_separated.png")
    else:
        experiments = ["1NN", "MAD", "MBL", "MBL-QP", "MC"]

        # only add legend to last plot
        # fig_exp[-1].tight_layout(rect=[0, 0, legend_x, 1])
        fig_legend, ax_legend = plt.subplots()
        fig_legend.set_size_inches(w=(3.25063 + 0.2) / 2, h=3.25063 / 2)
        ax_legend.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size": 10},
                         frameon=False)
        ax_legend.axis("off")
        fig_legend.savefig(f"bcm_comparison/individual/experiment_legend.pgf")
        # fig_exp[-1].legend(handles, labels, loc="center right")
        for i, exp in enumerate(experiments):
            fig_exp[i].set_size_inches(w=3.25063 / 2 + 0.1, h=3.25063 / 2)
            fig_exp[i].tight_layout()
            fig_exp[i].savefig(f"bcm_comparison/individual/{exp}.pgf")
            tikzplotlib.save(f"bcm_comparison/individual/{exp}.tex", figure=fig_exp[i])

    for locality in ind_plots:
        for method in ind_plots[locality]:
            fig, ax = ind_plots[locality][method]
            ax.set_ylim([min_y, 1])
            fig.set_size_inches(w=(3.25063 + 0.2) / 2, h=3.25063 / 2)
            tikzplotlib.save(f"bcm_comparison/supplement/loc={locality:0.2f}_{method}.tex", figure=fig)

    # handles and labels for locality plots
    handles, labels = ax_loc[-1].get_legend_handles_labels()

    # only select the solid lines
    handles = handles[5:]
    labels = labels[5:]

    if individual:
        # fig_loc[-1].tight_layout(rect=[0, 0, legend_x, 1])
        fig_legend, ax_legend = plt.subplots()
        fig_legend.set_size_inches(w=(3.25063 + 0.2) / 2, h=3.25063 / 2)
        ax_legend.legend(handles, experiments, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size": 10},
                         frameon=False)
        ax_legend.axis("off")
        fig_legend.savefig(f"bcm_comparison/individual/locality_legend.pgf")
        # fig_loc[-1].legend(handles, labels, loc="center right")
        for i, loc in enumerate(localities):
            fig_loc[i].set_size_inches(w=3.25063 / 2 + 0.1, h=3.25063 / 2)
            fig_loc[i].tight_layout()
            fig_loc[i].savefig(f"bcm_comparison/individual/locality{loc:.2f}.pgf")
            tikzplotlib.save(f"bcm_comparison/individual/locality{loc:.2f}.tex", figure=fig_loc[i])



    else:
        fig_loc.tight_layout(rect=[0, 0, legend_x, 1])
        fig_loc.legend(handles, labels, loc=(legend_x, 0.5))

        fig_loc.savefig("bcm_comparison/" + args.prefix + "locality_separated.png")

    if not individual:
        plt.show()
