# Will it be born? Predicting birth feasibility in Dynamic Energy Budget models

Code, datasets, and trained surrogate models for predicting whether a Dynamic Energy Budget (DEB) model parameterization reaches birth.

The repository contains two classification approaches:

- **Genetic programming (GP):** a symbolic classifier built with `gplearn`, including custom mathematical functions and named constants.
- **Neural network (DEBBirthNet):** a feedforward classifier implemented in PyTorch.

Both archived models use four dimensionless inputs, in this order:

| Input | Description |
|---|---|
| `g` | Energy investment ratio |
| `k` | Maintenance ratio |
| `v_Hb` | Scaled maturity at birth |
| `f` | Scaled functional response |

The target is `reached_birth`: `True` / `1` indicates that birth is reached; `False` / `0` indicates that it is not reached according to the dataset labels.

## Repository structure

```text
.
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── data/
│   ├── raw/                  # Simulation datasets: LHS, grids, and AmP-based samples
│   └── processed/
│       ├── data_description.md
│       ├── deb_reach_birth.csv
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── notebooks/
│   ├── Data Preprocessing.ipynb
│   ├── Test Set Results.ipynb
│   └── Decision Boundary Visualization.ipynb
├── src/debbirth/
│   ├── data/                 # MATLAB generators, schema, loading, splitting, scaling
│   ├── models/
│   │   ├── gp/               # Symbolic classifier, training, calibration, export
│   │   └── nn/               # Neural-network architecture and training
│   ├── evaluate/             # Classification metrics and model comparison
│   ├── plot/                 # Decision-boundary visualization
│   └── utils/
├── results/models/
│   ├── DEBBirthGP/            # Saved GP model, expression, configuration, metrics
│   └── DEBBirthNet/           # Saved NN weights, scaler, configuration, metrics
└── tests/                    # Symbolic-expression tests
```

## Data

The repository includes raw simulation outputs and prepared training, validation, and test splits.

The preprocessing notebook currently selects:

```text
data/raw/sample4D_lhs_N_200000_g_1em3_1e2_k_m4_1_f_0p1_1_ddec_1_getlb2.csv
```

It removes rows with nonpositive execution times and missing or nonfinite input parameters, then creates stratified **80% training, 10% validation, and 10% test** splits using `random_state=42`.

The raw directory also contains parameter grids for decision-boundary analysis, outputs associated with `get_lb` and `get_lb2`, and an AmP-based perturbed-parameter dataset.

See [data_description.md](data/processed/data_description.md) for column descriptions. Different dataset families contain different subsets of the documented columns. Simulation diagnostics such as `success`, `execution_time`, and `error_type` are separate from the classification target.

**Running the preprocessing notebook writes the processed CSV files, including the supplied splits.**

## Python setup

Clone the repository and enter its root directory:

```bash
git clone https://github.com/diogo-f-oliveira/deb-birth-prediction.git
cd deb-birth-prediction
```

Use a dedicated Python environment. 
Install the dependencies from `requirements.txt` and the additional packages used in the notebooks:

```bash
python -m pip install -r requirements.txt
```

## Training

Run the included training examples from the repository root:

```bash
python -m src.debbirth.models.gp.train
python -m src.debbirth.models.nn.train
```

Each module defines its example configuration in its `__main__` block. These examples load the prepared splits, train a model, evaluate it, and save artifacts under `results/runs/`.

Training settings can be customized through:

- `GPConfig` and `TrainGPConfig` for genetic programming.
- `DEBBirthNetConfig` and `TrainDEBBirthNetConfig` for the neural network.

The example settings are not necessarily identical to the archived model settings. In particular, the GP example uses a different parsimony coefficient and includes an additional zero constant.

## Saved models and analysis

The repository includes trained artifacts under `results/models/`.

**DEBBirthGP** contains:

- `model/gp_model.joblib`
- `model/best_program.txt`
- `model/best_program_matlab.m`
- Training configuration, history, and validation/test metrics

**DEBBirthNet** contains:

- `model/model_state_dict.pth`
- `model/scaler.pth`
- Training configuration, history, and validation/test metrics

Use `load_gp_run` and `load_trained_nn` from the corresponding training modules to load these artifacts.

For inference, preserve the feature order `g`, `k`, `v_Hb`, `f`. The GP model receives unscaled inputs. The neural network requires its saved scaler, which applies the configured logarithmic transformation and standardization.

The notebooks provide workflows for:

- **Data Preprocessing:** cleaning, exploration, and split creation.
- **Test Set Results:** loading saved models, comparing metrics, timing inference, and plotting evaluation curves.
- **Decision Boundary Visualization:** comparing simulation outcomes and surrogate decision boundaries.

Before running the notebooks, adapt the local data and figure-output paths. Their relative paths generally assume a working directory of `notebooks/`, while imports require the repository root to be on Python’s module search path.

The saved configuration files also contain absolute paths from the original training machine. Adapt these before using the configurations to load data. The saved GP configuration includes string representations of custom function objects, so it is not a standalone configuration that can directly reconstruct training.

## Regenerating simulation data

MATLAB generation scripts are provided in `src/debbirth/data/` for parameter grids, Latin hypercube sampling, and perturbations of AmP species parameters.

Regeneration requires the external DEB routines used by the scripts and appropriate MATLAB toolboxes for functions such as `parfeval` and `lhsdesign`. AmP-based generation also requires the species data. Configure these dependencies and the input/output paths before running the scripts.

The supplied CSV files allow Python training and analysis without regenerating the simulations.

## Citation

If you use this code or data, please cite the software using [CITATION.cff](CITATION.cff), which records the title **“Will it be born? Code and data”** and the authors:

Diogo F. Oliveira, Miguel S. E. Martins, Gonçalo M. Marques, Tiago Domingos, Susana M. Vieira, and João M. C. Sousa.

For reproducibility, identify the release tag or commit used in your work.

## License

This repository is distributed under the [MIT License](LICENSE).