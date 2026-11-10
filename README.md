# CSCI576Group13
CSCI 576 Deep Learning Fall 2026 Project

# NEAT: NeuroEvolution of Augmenting Topologies

A custom implementation of **NeuroEvolution of Augmenting Topologies (NEAT)** using PyTorch.

The project aims to evolve neural network architectures and weights using genetic algorithms. Networks will support feedforward (FFNN), recurrent (RNN), and long short-term memory (LSTM) cells. The implementation will initially be tested on maze-solving and pole-balancing simulations.

## Getting Started

### 1. Install Conda

If you do not already have Conda installed, install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or [Anaconda](https://www.anaconda.com/download).

### 2. Clone the Repository

```bash
git clone https://github.com/Landon-Gehr/CSCI576Group13.git
cd https://github.com/Landon-Gehr/CSCI576Group13.git
```

### 3. Create the Conda Environment

The project includes an `environment.yml` file containing the dependencies required to run the project.

From the project root directory, create the environment with:

```bash
conda env create -f environment.yml
```

This creates a Conda environment using the dependencies and environment name specified in `environment.yml`.

### 4. Activate the Environment

After creating the environment, activate it using the environment name specified by the `name` field in `environment.yml`:

```bash
conda activate neat-env
```

## Updating the Environment

If any new packages are added to the conda environment, update the environment.yaml file with:

```bash
conda env export --name neat-env --no-builds > environment.yaml
```

to pin the exact package version.