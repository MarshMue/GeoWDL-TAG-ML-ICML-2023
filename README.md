## How to replicate paper results

It is recommended that you set up a conda environment and install pytorch, numpy, pot, scipy, and tqdm. Specific versions should generally not matter, but we used python 3.9.7

Relevant data is included in the top level data folder.

The main WDL code is in `wdl/WDL.py`

We use code from the following two papers:
- https://proceedings.mlr.press/v162/werenski22a.html
- https://arxiv.org/abs/2012.02134

We thank the authors of these works for providing their code.

*Note:* our code will install MNIST to the data folder, but you will need to install the NLP data obtained [here](https://www.dropbox.com/sh/nf532hddgdt68ix/AABGLUiPRyXv6UL2YAcHmAFqa?dl=0) (only `bbcsport-emd-tr-te-split.mat`).

### Figure 2

(a) The jupyter notebook `k-deep-simplex-main/src/KDS on MNIST.ipynb` contains code to generate this figure

(b) runnning `new_wdl_tests/MNIST Tests/simpleSetsOfDigits.py` will generate data to be plotted by `new_wdl_tests/MNIST Tests/MNIST_results_viz.py`

(c), (d) and (e) by runnning `new_wdl_tests/MNIST Tests/simpleSetsOfDigitsNoisy.py` will generate data to be plotted by `new_wdl_tests/MNIST Tests/MNIST_results_viz.py`

The `fname` parameter in the second file will need to be changed to `results/noisy_data.pkl`


### Figures 3, 6, and 7

We used an HPC to generate these results. Included are a snippet of the slurm batch file as well as the relevant scripts to generate the data and plots

The main workhorse to generate the data is `new_wdl_tests/NLPTests/BCM_extension.py`

You may wish to try running a smaller version `new_wdl_tests/NLPTests/BCM_extension_split.py` which allows one to specify only 1 dictionary size. 

`new_wdl_tests/NLPTests/trial_result_plotter.py` and `new_wdl_tests/NLPTests/combinedTrialResultPlotter.py` were used to generate the figures


Example HPC script: `new_wdl_tests/NLPTests/example_HPC_script.sh`

### Figure 4
The notebook `new_wdl_tests/MNIST Tests/noisyMnist.ipynb` provides code to recreate this figure. 

### Figure 5

The script `new_wdl_tests/MNIST Tests/simpleSetsOfDigitsTAGSSupplement.py` will generate the pdf plots for each of these figures. 