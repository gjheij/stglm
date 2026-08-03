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
    workdir = "/tmp/stglm_pybest"
    evs = [
        "CS-",
        "CS+_noUCS",
        "CS+_withUCS"
    ]
    hrf = "dgamma"
    subject = "015"
    run_mode = "loop"
    lss_mode = "cond"
    n_procs = 4
    task = os.path.basename(project_dir)
    desc = "pybest"
    confound_subset = "Motion parameters"
    deriv_dir = os.path.join(workdir, "derivatives", "stglm")
    
    # no confounds
    no_confounds = True

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
        no_confounds=True,
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

    confound_regexes = None
    if not no_confounds:
        confound_regexes = [confound_options[confound_subset]]
        logger.info(f"Using '{confound_subset}' ({confound_regexes}) from confound files")
    else:
        logger.info("Not including any confounds; assuming data is already treated properly..")

    logger.info(f"Including conditions: {evs}")
    ddict = {
        "hrf": hrf,
        "mode": lss_mode,
        "confounds": confound_subset,
        "confounds_regex": confound_regexes,
        "conditions": evs
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