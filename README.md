# stGLM

**stGLM** is a Python-based CLI tool and workflow engine designed to run single-trial GLMs (General Linear Models) on fMRI data using the BIDS standard. It supports LSA (Least Squares All) and LSS (Least Squares Separate) approaches to model trial-specific brain responses.

# Features
- Run-wise or Concatenated Analysis: Choose between per-run estimation or merged multi-run modeling.

- LSA and LSS Estimation: Compute trial-specific betas using flexible GLM estimation strategies.

- BIDS-Compatible: Fully supports input/output in BIDS format.

- Confound Selection: Extract nuisance regressors based on regex filtering of confound files (e.g., motion, aCompCor, ICA-AROMA).

- Design Matrix Visualization: Automatically generates design matrix plots for QA.

- Metadata Generation: Saves trial structure and modeling parameters as JSON.

# Installation

```bash
pip install git+https://github.com/gjheij/stglm
```

# Usage

```bash
job="stglm"

# or slurm:
job="sbatch --partition=main --cpus-per-task=4 --job-name=sub-015_desc-stglm --output=/mnt/d/fMRI/logs/stglm/HRA --export=ALL /home/gjheij/miniconda3/envs/fmriproc/bin/stglm"

${job} \
    --project-dir /path/to/bids_dataset \
    --subject 01 \
    --evs CS- CS+_noUCS CS+_withUCS \
    --confound-subset "Motion parameters" \
    --hrf dgamma_with_derivs \
    --lss-mode cond \
    --task HRA \
    --desc masked \
    --run-mode loop \
    --space T1w \
    --n-procs 4
```

## Required Arguments

| Argument        | Description                                                           |
| --------------- | --------------------------------------------------------------------- |
| `--project-dir` | Path to your BIDS dataset.                                            |
| `--subject`     | BIDS subject label (e.g., `"01"`).                                    |
| `--evs`         | Event/trial names to model (e.g., `CS-`, `CS+_noUCS`, `CS+_withUCS`). |

## Optional arguments
| Argument            | Default                           | Description                                                                                        |
| ------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------- |
| `--deriv-dir`       | `<project-dir>/derivatives/stglm` | Output directory.                                                                                  |
| `--confound-subset` | `"Motion parameters"`             | Nuisance regressors to include.                                                                    |
| `--hrf`             | `dgamma`                          | HRF basis: `dgamma`, `dgamma_with_derivs`, or `flobs`.                                             |
| `--lss-mode`        | `cond`                            | Mode for LSS estimation: `cond` or `nuis`.                                                         |
| `--task`            | `HRA`                             | Task label (e.g., `rest`, `HRA`).                                                                  |
| `--desc`            | `masked`                          | Description label for BOLD files: `preproc` (from fMRIprep), or `masked` (skull-stripped version). |
| `--space`           | `T1w`                             | Spatial normalization space (e.g., `T1w`, `MNI152NLin2009cAsym`, or `func` for native BOLD space). |
| `--run-mode`        | `loop`                            | Choose `loop` (per-run estimation) or `concat` (merged across runs).                               |
| `--workdir`         | `/tmp`                            | Working directory for intermediate files.                                                          |
| `--n-procs`         | `1`                               | Number of CPU processes to use.                                                                    |

## Outputs

Outputs are saved in:

```bash
<deriv-dir>/sub-<ID>/[ses-<ID>/]func/
```

Like so:

```bash
/mnt/d/fMRI/HRA/derivatives/stglm/
└── sub-015
    └── ses-1
        └── func
            ├── sub-015_ses-1_task-HRA_run-1_space-T1w_desc-lsa_betas.nii.gz
            ├── sub-015_ses-1_task-HRA_run-1_space-T1w_desc-lss_betas.nii.gz
            ├── sub-015_ses-1_task-HRA_run-1_space-T1w_desc-stglm.json
            ├── sub-015_ses-1_task-HRA_run-2_space-T1w_desc-lsa_betas.nii.gz
            ├── sub-015_ses-1_task-HRA_run-2_space-T1w_desc-lss_betas.nii.gz
            ├── sub-015_ses-1_task-HRA_run-2_space-T1w_desc-stglm.json
            ├── sub-015_ses-1_task-HRA_run-3_space-T1w_desc-lsa_betas.nii.gz
            ├── sub-015_ses-1_task-HRA_run-3_space-T1w_desc-lss_betas.nii.gz
            ├── sub-015_ses-1_task-HRA_run-3_space-T1w_desc-stglm.json
            ├── sub-015_ses-1_task-HRA_run-4_space-T1w_desc-lsa_betas.nii.gz
            ├── sub-015_ses-1_task-HRA_run-4_space-T1w_desc-lss_betas.nii.gz
            └── sub-015_ses-1_task-HRA_run-4_space-T1w_desc-stglm.json
```

For each run or subject, you'll get:

- `*_desc-lsa_betas.nii.gz`: Beta images from LSA.

- `*_desc-lss_betas.nii.gz`: Beta images from LSS.

- `*.json`: Metadata about model configuration and trial definitions.

## Confound Subset Options
Choose from the following predefined options (regex matched following `HALFpipe`'s convention:

- "`ICA-AROMA`"

- "`Motion parameters`"

- "`Derivatives of motion parameters`"

- "`Motion parameters squared`"

- "`Derivatives of motion parameters squared`"

- "`Motion scrubbing`"

- "`aCompCor (top five components)`"

- "`White matter signal`"

- "`CSF signal`"

- "`Global signal`"