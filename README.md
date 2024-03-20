# Detecting critical treatment effect bias in observational studies

[![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2312.03871-B31B1B.svg)](https://arxiv.org/abs/2312.03871)
[![Python 3.9.18](https://img.shields.io/badge/python-3.9.18-blue.svg)](https://python.org/downloads/release/python-3918/)
[![JAX 0.4.23](https://img.shields.io/badge/jax-0.4.23-green.svg)](https://jax.readthedocs.io/en/latest/changelog.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the Python implementation of [Hidden among subgroups: Detecting critical treatment effect bias in observational studies](https://arxiv.org/abs/2312.03871).

* [Overview](#overview)
* [Getting Started](#getting-started)
* [Contents](#contents)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Citation](#citation)

## Overview

This repository presents the methods from the paper "Hidden among subgroups: Detecting critical treatment effect bias in observational studies."

Randomized trials are the gold standard for informed decision-making in medicine, yet they may not always capture the full scope of the population in clinical practice. Observational studies are usually more representative of the patient population but are susceptible to variuos biases such as those arising from unmeasured confounding.

Our tools are designed to enhance the reliability of observational studies. We provide a statistical test that allows researchers to validate if the treatment effects observed in specific patient subgroups fall within an acceptable tolerance when **benchmarked** against randomized trials. Additionally, we propose a method to estimate a lower bound for potential biases within these subgroup effects. 

## Getting Started

### Dependencies

- Python 3.9.18
- Numpy 1.26.2
- Scipy 1.11.4
- Scikit-learn 1.1.3
- Pandas 2.1.4
- Scikit-uplift 0.5.1
- Optax 0.1.7
- JAX 0.4.23
- Flax 0.7.5


### Installation

To set up your environment and install the package, follow these steps:

#### Create and Activate a Conda Environment

Start by creating a Conda environment with Python 3.9.18. This step ensures your package runs in an environment with the correct Python version. 
```bash
conda create -n myenv python=3.9.18
conda activate myenv
```
#### Install the Package

There are two ways to install the package:

1. **Local Installation:**
   Start by cloning the repository from GitHub. Then, upgrade `pip` to its latest version and use the local setup files to install the package. This method is ideal for development or when you have the source code.
   ```bash
   git clone https://github.com/jaabmar/kernel-test-bias.git
   cd kernelized-test-code
   pip install --upgrade pip
   pip install -e .
   ```
2. **Direct Installation from GitHub (Recommended):**
   You can also install the package directly from GitHub. This method is straightforward and ensures you have the latest version.
   ```bash
   pip install git+https://github.com/jaabmar/kernel-test-bias.git
   ```

## Contents

The `src` folder contains the core code of the package, organized as follows::

- `datasets`: This directory includes modules for loading and preprocessing datasets.
  - `bias_models.py`: Defines the bias models used in the paper.
  - `hillstrom.py`: Contains functions specific to the Hillstrom dataset processing.
  
- `tests`: Includes testing procedures for bias discussed in the paper.
  - `ate_test.py`: An implementation of the average treatment effect (ATE) test that allows for tolerance, inspired by [De Bartolomeis et al.](https://arxiv.org/abs/2312.03871).
  - `kernel_test.py`: Our proposed kernel-based test that offers both granularity and tolerance.
  - `utils_test.py`: Utility functions to support testing procedures.

- `experiment_utils.py`: Utility functions that facilitate the execution of experiments.

- `experiment.py`: Executes example experiments as per the paper, with parameters that can be customized by the user.

### Running Experiments

To run experiments using `experiment.py`, follow these instructions:

1. **Activate Your Environment**: Ensure you have activated the Conda environment or virtual environment where the package is installed.

2. **Run the Script**: From the terminal, navigate to the `src` directory where `experiment.py` is located, and run the following command:
   ```bash
   python experiment.py --test_type [TYPE] --bias_model [MODEL] --user_shift [SHIFT] ...
   ```
Replace [TYPE], [MODEL], [SHIFT], etc., with your desired values.

Example:

```bash
python experiment.py --test_type kernel_test --bias_model scenario_1 --user_shift 60.0 --epochs 2000 --lr 0.1
```
For a complete list of configurable parameters and their descriptions, consult the argument parser setup in `experiment.py`.

## Contributing

We welcome contributions to improve this project. Here's how you can contribute:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Contact

For any inquiries, please reach out:

- Javier Abad Martinez - [javier.abadmartinez@ai.ethz.ch](mailto:javier.abadmartinez@ai.ethz.ch)
- Piersilvio de Bartolomeis - [pdebartol@ethz.ch](mailto:pdebartol@ethz.ch)
- Konstantin Donhauser - [konstantin.donhauser@ai.ethz.ch](mailto:konstantin.donhauser@ai.ethz.ch)

## Citation

If you find this code useful, please consider citing our paper:
 ```
@article{debartolomeis2024hidden,
  title={Hidden among subgroups: Detecting critical treatment effect bias in observational studies},
  author={Piersilvio De Bartolomeis and Javier Abad and Konstantin Donhauser and Fanny Yang},
  year={2024},
  journal={arXiv preprint arXiv:2312.03871},
  eprint={2312.03871},
  archivePrefix={arXiv},
  primaryClass={stat.ML}
}
```