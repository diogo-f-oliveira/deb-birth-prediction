# Environment and archived baseline check

Checked on 2026-09-06 for task T01. This was a setup and inference check, not training, a dataset audit, or a reproduction of the paper's metrics.

## Execution environment

The existing conda environment `debbirth` is available:

| Component | Observed value |
| --- | --- |
| Python | 3.14.7, Anaconda build |
| NumPy | 2.5.2 |
| PyTorch | 2.14.0 |
| scikit-learn | 1.9.0 |
| gplearn | 0.4.3 |
| `torch.cuda.is_available()` | `False` |
| Inference device used | CPU |

Conda resolves to `C:/Users/diogo/miniconda3/Library/bin/conda.bat`. The verified environment interpreter is `C:/Users/diogo/miniconda3/envs/debbirth/python.exe`. CUDA is unavailable to this environment; this does not establish whether the machine has GPU hardware.

Working environment check from the repository root:

```powershell
conda run -n debbirth python -c "import sys, torch; print(sys.executable); print(sys.version); print(torch.cuda.is_available())"
```

Passing a multiline Python string through `conda.bat run ... -c` returned no output in this session and was not accepted as verification. The baseline probe was rerun successfully with the verified `debbirth` interpreter directly. For longer work, use a script through conda or the verified environment interpreter, rather than relying on multiline batch-file argument forwarding.

No dependencies were installed or changed. Optional tuning dependencies were not checked. The current environment differs from the Python 3.10 environment described in the paper; successful inference is not evidence of identical retraining behavior.

## Data access

Normal sandbox access to `data/` and `tests/` failed. An approved read outside the sandbox confirmed that the directories and files are present. No files were restored or replaced, and permissions were not changed.

| File | Size in bytes |
| --- | ---: |
| `data/processed/deb_reach_birth.csv` | 27,049,864 |
| `data/processed/train.csv` | 21,639,314 |
| `data/processed/val.csv` | 2,706,292 |
| `data/processed/test.csv` | 2,704,444 |
| `data/raw/sample4D_lhs_N_200000_g_1em3_1e2_k_m4_1_f_0p1_1_ddec_1_getlb2.csv` | 26,150,652 |

The training CSV header and one row were read successfully. Its columns are:

```text
g,k,v_Hb,f,lb,tb,lb<f,k*vHb<c,reached_birth,success,execution_time,error_type,error_message
```

The processed description file and the test directory were also listed. No tests were executed. Future data reads may still require execution outside this sandbox; do not interpret sandbox Git deletion reports as actual data loss. Full CSV parsing, counts, and normalization coverage remain T02 work.

## Archived model inference

Both repository loaders worked on the archived artifacts:

- GP: `load_gp_run('results/models/DEBBirthGP')`, including a non-null config object. This does not mean the serialized primitive strings can reconstruct training.
- NN: `load_trained_nn('results/models/DEBBirthNet', device='cpu')`, using its saved scaler. The loaded network was in training mode; `model.eval()` was called explicitly before inference to disable dropout. T05 tracks the loader correction.

The same four synthetic positive parameter vectors were used in the historical feature order `(g, k, v_Hb, f)`. These are diagnostic inputs, not labeled evaluation observations.

| g | k | v_Hb | f | GP birth probability | NN birth probability | Both predictions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3 | 0.1 | 1 | 0.99289916 | 0.99953759 | 1 |
| 1 | 1 | 0.5 | 1 | 0.71930577 | 0.99282008 | 1 |
| 1 | 1 | 2 | 1 | 3.58492e-7 | 1.53183e-8 | 0 |
| 0.1 | 3 | 0.01 | 0.8 | 0.86167091 | 0.99719739 | 1 |

Reproduce the inference check from the repository root in PowerShell:

```powershell
$baselineProbe = @'
import numpy as np
import torch
from src.debbirth.models.gp.train import load_gp_run
from src.debbirth.models.nn.train import load_trained_nn

X = np.array([[1., .3, .1, 1.], [1., 1., .5, 1.],
              [1., 1., 2., 1.], [.1, 3., .01, .8]])
gp = load_gp_run('results/models/DEBBirthGP')
print('GP:', gp['model'].predict_proba(X)[:, 1])
nn = load_trained_nn('results/models/DEBBirthNet', device='cpu')
nn['model'].eval()
with torch.inference_mode():
    X_nn = nn['scaler'].transform(torch.tensor(X, dtype=torch.float32))
    print('NN:', nn['model'].predict_proba(X_nn))
'@
& 'C:/Users/diogo/miniconda3/envs/debbirth/python.exe' -c $baselineProbe
```

All reported probabilities were finite and in range. No model weights, scalers, saved configurations, or source code were modified.

## Existing exploratory notebooks

Inspected source cells without running them:

- `notebooks/Data Exploration and Experiments.ipynb`: parameter/species exploration and preliminary classifiers/regression, with positional column selection and its own split logic.
- `notebooks/ML Experiments.ipynb`: existing split-loader usage with random-forest and gplearn experiments.

These contain relevant exploratory work but neither inspection established a normalized/boundary implementation. Preserve the notebooks; choose where to put T02 analysis after considering their existing scope.

## Outcome

T01 is complete: the environment runs, the data are intact and readable outside the sandbox, and both archived models produce predictions. Proceed to T02. The environment version difference, sandbox data access, and required explicit NN evaluation mode remain documented limitations.
