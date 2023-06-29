[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# K-Deep Simplex (KDS)

This repository contains:
* a PyTorch implementation of KDS in `src/model.py`
* a script `src/clustering_experiment.py` that uses KDS to cluster various real-world data sets
* a script `src/synthetic_experiment.py` that evaluates the performance of KDS as a function of dictionary size and cluster separation

### Dependencies

* Sacred-0.8.1
* PyTorch-1.5.0
* TensorFlow-2.2.0
* TensorBoard-2.3.0
* Keras-2.4.3

### Usage

To run either experiment, navigate to the `src` directory.

**Example usage (clustering experiment):**

`python clustering_experiment.py -F ../results/moons with moons_default`

Other provided parameter settings include `salinas_default`, `yale2_default`, `yale3_default`, and `mnist_default`.

**Example usage (synthetic experiment):**

`python synthetic_experiment.py -F ../results/synthetic`
