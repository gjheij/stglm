from pathlib import Path
from bids import BIDSLayout
from types import SimpleNamespace
from stglm.cli import build_inputs, run_stglm_from_inputs
from stglm.logger import get_logger
import os

opj = os.path.join
logger = get_logger(__name__)

def test_concat():

    logger.info("Starting single-trial GLM workflow")

    project_dir = "/mnt/d/fMRI/HRA"
    workdir = "/tmp/stglm_concat"
    subject = "015"
    run_mode = "concat"
    n_procs = 1
    task = os.path.basename(project_dir)
    desc = "masked"
    confound_subset = "Motion parameters"
    deriv_dir = os.path.join(workdir, "derivatives", "stglm")

    space = "T1w"
    space_kwargs = {} if space == "func" else {"space": space}

    layout = BIDSLayout(project_dir, derivatives=True)

    if deriv_dir is None:
        derivatives_dir = Path(project_dir) / "derivatives" / "stglm"
    else:
        derivatives_dir = deriv_dir

    logger.info(layout)

    runs = build_inputs(
        layout,
        subject=subject,
        run_mode=run_mode,
        desc=desc,
        task=task,
        **space_kwargs
    )

    confound_options = {
        "ICA-AROMA": "ICA-AROMA",
        "Motion parameters": "(trans|rot)_[xyz]",
        "Derivatives of motion parameters": "(trans|rot)_[xyz]_derivative1",
        "Motion parameters squared": "(trans|rot)_[xyz]_power2",
        "Derivatives of motion parameters squared": "(trans|rot)_[xyz]_derivative1_power2",
        "Motion scrubbing": "motion_outlier[0-9]+",
        "aCompCor (top five components)": "c_comp_cor_0[0-4]",
        "White matter signal": "white_matter",
        "CSF signal": "csf",
        "Global signal": "global_signal",
    }

    confound_regexes = [confound_options[confound_subset]]
    logger.info(f"Using '{confound_subset}' ({confound_regexes}) from confound files")

    ddict = {
        "name": "SingleTrials",
        "setting": "SingleTrialsSetting",
        "type": "single_trials",
        "conditions": [
            "CS-",
            "CS+_noUCS",
            "CS+_withUCS"
        ],
        "hrf": "dgamma",
        "mode": "cond"
    }

    feature = SimpleNamespace(**ddict)

    from nipype import config
    config.update_config({
        'execution': {
            'remove_unnecessary_outputs': False,
            'keep_inputs': True,  # Optional: keep input copies too
        }
    })

    run_stglm_from_inputs(
        run_data_list=runs,
        feature=feature,
        workdir=workdir,
        confound_regexes=confound_regexes,
        n_procs=int(n_procs),
        derivatives=str(derivatives_dir)
    )