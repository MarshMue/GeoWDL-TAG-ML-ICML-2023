import torch
from utilities.MNISTutils import MNISTtoHistograms
from utilities.cost import gridCost
from wdl.bregman import barycenter, OT
from wdl.WDL import WDL
from utilities.simpleDistributions import sampleBaryFromDict, matchAtoms

import pickle
import datetime

import scipy.io as sio

if __name__ == "__main__":
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    #### pick mnist samples
    height, width = 28, 28
    n_atoms_per_digit = 3
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    n_digits = len(digits)
    D, _ = MNISTtoHistograms('../../data', digits=digits, samples_per_digit=n_atoms_per_digit,
                             height=height,
                             width=width)
    
    # uncomment to load original true dictionary used in experiment in paper
    # with open(
    #         "MNISTdata.pkl",
    #         'rb') as f:
    #     data = pickle.load(f)

    # D = data["True D"]

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

    n_atoms = D.shape[1]

    #### iterate over localities
    localities = [0.0, 0.001, 0.1, 10]


    # WDL Params
    dtstr = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    test_name = f"all_digits_{dtstr}"
    init_method = "kmeans++-init"
    max_iters = 250
    lr = 0.25
    n_restarts = 1

    # solver for matching
    OTsolver = OT(gridCost(height, width), method='lp')

    # save assignment costs of atoms and weights
    atom_assign_costs = torch.zeros((len(localities)))
    weights_assign_costs = torch.zeros((len(localities)))

    # dict to save data for use after running script
    big_dict = {}
    for locality in localities:
        big_dict[locality] = {}

    big_dict["True D"] = D.to("cpu")
    big_dict["True weights"] = Lambda.to("cpu")
    big_dict["max iter"] = max_iters
    big_dict["max sink iter"] = maxsinkiters
    big_dict["restarts"] = n_restarts
    big_dict["lr"] = lr

    # video of atoms
    frames = []
    # labels for plots in video
    labels = []
    for j in range(len(localities)):
        labels.append([])
        mu = localities[j]
        #### WDL
        print(f"--------- Locality: {mu} ---------")

        wdl = WDL(n_atoms=D.shape[1])

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
                               jointOptimizer=torch.optim.Adam,
                               jointOptimKWargs={"lr": lr},
                               verbose=True,
                               n_restarts=n_restarts,
                               log_iters=5,
                               log=True,
                               )
        weights = weights.to("cpu")

        DLearned, weights, cost = matchAtoms(wdl.D, D, weights, OTsolver=OTsolver)
        atom_assign_costs[j] = cost

        weights = weights.to("cpu")
        DLearned = DLearned.to("cpu")
        Lambda = Lambda.to("cpu")

        #### display and save results
        weights_assign_costs[j] = torch.linalg.norm(weights - Lambda) / weights.shape[1]


        big_dict[mu]["Learned D"] = DLearned.to("cpu")
        big_dict[mu]["Learned weights"] = weights.to("cpu")

    with open(f"results/data.pkl", "wb") as f:
        pickle.dump(big_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

