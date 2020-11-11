# Differentiable Optimization for the Prediction of Ground State Structures (DOGSS)
Implemens the Differentiable Optimization for the Prediction of Ground State Structures (DOGSS) that takes arbitrary chemical structures to predict their ground-state structures.
The following paper describes the details of the DOGSS framework:
[Differentiable Optimization for the Prediction of Ground State Structures (DOGSS)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.173001)

## Installation
Create conda envionrment with require packages:
```bash
conda env create -f env.yml
```
Activate the conda environment with 
```bash
conda activate dogss
```
Install the package with `pip install -e .`.

## Usage
We provide scripts to train/load DOGSS for predicting ground state structures of only `H adsorption` dataset. Other datasets mentioned in the paper (`Bare surfaces` and `CO adsorption`) can be used in the same way but with different hyperparameters.