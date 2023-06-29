import torch
import pickle
import glob
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kmin", type=int)
    parser.add_argument("--kmax", type=int)
    parser.add_argument("--locality", type=float)

    args = parser.parse_args()

    assert args.kmax >= args.kmin, "kmax should be larger than kmin"

    n_ks = args.kmax - args.kmin + 1

    all_data_dict = {}

    num_data = 0

    for file in glob.glob("data_outputs/*.pkl"):
        if float(file.split("mu=")[1].split(".pkl")[0]) == args.locality:
            # check whether this file is used or not
            used = True
            with open(f'{file}', 'rb') as handle:
                data_dict = pickle.load(handle)
                # each key is what classification method was used
                for key in data_dict:
                    # ensure only tests with valid k range are used
                    if data_dict[key].shape[1] != n_ks:
                        used = False
                        break

                    # add the results to a list
                    if key in all_data_dict:
                        all_data_dict[key].append(data_dict[key])
                    else:
                        all_data_dict[key] = [data_dict[key]]

            # only count files that are used
            if used:
                num_data += 1

    fig, ax = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    fig_sep, ax_sep = plt.subplots(2, 4, figsize=(24, 12))

    # string matching patterns
    NNe = re.compile("1NN.*e")
    NNw = re.compile("1NN.*w")

    MADe = re.compile("Mean.*e")
    MADw = re.compile("Mean.*w")

    MINe = re.compile("Min.*e")
    MINw = re.compile("Min.*w")
    MINl = re.compile("Min.*l")

    MAX = re.compile("Max.*")

    ax_sep[0, 0].set_title("Nearest Neighbor - e")
    ax_sep[0, 1].set_title("Nearest Neighbor - w")
    ax_sep[0, 2].set_title("Mean Average Distance - e")
    ax_sep[0, 3].set_title("Mean Average Distance - w")

    ax_sep[1, 0].set_title("Min BC Loss - e")
    ax_sep[1, 1].set_title("Min BC Loss - w")
    ax_sep[1, 2].set_title("Min BC Loss - l")
    ax_sep[1, 3].set_title("Max Coordinate")

    for key in all_data_dict:
        # combine the results of each trial and take the average over k
        data = torch.vstack(all_data_dict[key])

        linestyle = '-'
        if key[-1] == "e":
            linestyle = "--"
        elif key[-1] == "l":
            linestyle = ":"

        if "Data" in key:
            curr_ax = 0
            color = 'red'
            label = "Data"
        else:
            curr_ax = 1
            color = 'blue'
            label = "Dict"

        if NNe.match(key):
            curr_ax_sep = ax_sep[0, 0]
        elif NNw.match(key):
            curr_ax_sep = ax_sep[0, 1]
        elif MADe.match(key):
            curr_ax_sep = ax_sep[0, 2]
        elif MADw.match(key):
            curr_ax_sep = ax_sep[0, 3]
        elif MINe.match(key):
            curr_ax_sep = ax_sep[1, 0]
        elif MINw.match(key):
            curr_ax_sep = ax_sep[1, 1]
        elif MINl.match(key):
            curr_ax_sep = ax_sep[1, 2]
        elif MAX.match(key):
            curr_ax_sep = ax_sep[1, 3]

        # plot data vs dict
        curr_ax_sep.plot(range(args.kmin, args.kmax + 1), data.mean(0), color=color, label=label)
        curr_ax_sep.fill_between(range(args.kmin, args.kmax + 1),
                                 data.mean(0) - data.std(0), data.mean(0) + data.std(0),
                                 color=color, alpha=0.2)

        # plot on all together plot
        p = ax[curr_ax].plot(range(args.kmin, args.kmax + 1), data.mean(0), label=key, linestyle=linestyle)

        # add std dev details
        c = p[0].get_color()

        # ax[curr_ax].fill_between(range(args.kmin, args.kmax + 1),
        #                         data.mean(0) - data.std(0), data.mean(0) + data.std(0),
        #                         color=c, alpha=0.2)

    ax[0].set_ylim([0, 1])
    ax[1].set_ylim([0, 1])

    ax[0].set_title("Data as reference")
    ax[1].set_title("Dict as reference")
    ax[0].legend()
    ax[1].legend()

    for i in range(2):
        for j in range(4):
            ax_sep[i, j].set_xlabel("k")
            ax_sep[i, j].set_ylabel("Accuracy")
            ax_sep[i, j].legend()
            ax_sep[i, j].set_ylim([0, 1])

    fig.suptitle(f"Averaged over {num_data} trials, locality={args.locality:.3f}")
    fig_sep.suptitle(f"Averaged over {num_data} trials, locality={args.locality:.3f}")

    fig.tight_layout()
    fig.savefig(
        f"bcm_comparison/{args.kmin}to{args.kmax}_loc={args.locality}_results-{datetime.now().strftime('%Y%m%d-%H:%M')}.png")

    fig_sep.tight_layout()
    fig_sep.savefig(
        f"bcm_comparison/{args.kmin}to{args.kmax}_loc={args.locality}_results_sep-{datetime.now().strftime('%Y%m%d-%H:%M')}.png")

    plt.show()
