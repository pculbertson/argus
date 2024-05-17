# Argus
A vision-based pose estimator for the in-hand manipulation hardware setup in the AMBER Lab.

## Installation

1. Clone this repository and `cd` into the repo root.
2. Manage the `cudatoolkit` version and `python` dependencies using `conda` by running the following
```
conda env create --name <your_env_name> --file=environment.yml
pip install -e .[dev]  # use [dev] for dev tooling and testing - else, don't need it
```
3. If you installed the dev dependencies, activate the pre-commit checks with
```
pre-commit install
```
Now, when you commit files, the checks will be run first.