# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import shutil
import argparse
import numpy as np
import pandas as pd
import nibabel as nb
from stglm import (
    utils,
    conditions
)
from lazyfmri.utils import get_file_from_substring
from pathlib import Path
from bids import BIDSLayout
from types import SimpleNamespace
from stglm.confounds import init_confounds_select_wf
from stglm.single_trials import init_singletrials_wf
from stglm.logger import get_logger
from stglm.events import get_event_names

logger = get_logger(__name__)
opj = os.path.join

def build_inputs(
    layout,
    subject,
    run_mode="loop",
    glm_single=False,
    task=None,
    desc="masked",
    confounds_suffix="timeseries",
    no_confounds=False,
    evs=None,
    glm_kws={},
    **kwargs
):

    if glm_single:
        no_confounds = True

    bold_files = layout.get(
        subject=subject,
        task=task,
        desc=desc,
        suffix='bold',
        extension='nii.gz',
        return_type='object',
        **kwargs
    )

    if not bold_files:
        raise FileNotFoundError(f"No bold files found for sub-{subject}")
    
    mask_files = layout.get(
        subject=subject,
        task=task,
        suffix='mask',
        extension='nii.gz',
        return_type='file',
        **kwargs
    )

    if not no_confounds:
        confounds = layout.get(
            subject=subject,
            task=task,
            suffix=confounds_suffix,
            extension='tsv',
            return_type='file'
        )

        if not confounds:
            raise FileNotFoundError(f"No confounds found for sub-{subject}")
        else:
            logger.info(f"Found {len(confounds)} files with suffix='{confounds_suffix}'")
    else:
        confounds = None

    events = layout.get(
        subject=subject,
        task=task,
        suffix='events',
        extension='tsv',
        return_type='file'
    )

    if not events:
        raise FileNotFoundError(f"No events found for sub-{subject}")

    entities = bold_files[0].entities
    tr = entities["RepetitionTime"]

    if not glm_single:
        if run_mode == "concat":
            logger.info(f"Estimation mode = {run_mode}; concatenating all runs")
            merged_bold = utils.merge_bold(
                [i.path for i in bold_files]
            )

            if not no_confounds:
                merged_confounds = utils.merge_confounds(
                    confounds,
                    out_file="/tmp/merged_confounds.tsv"
                )
            else:
                merged_confounds = None

            merged_events = utils.merge_events(
                events,
                bold_files,
                tr,
                out_file="/tmp/merged_events.tsv"
            )

            if evs is None:
                evs = get_event_names(merged_events)

            logger.info(f"Including events = {evs}")
            try:
                entities.pop("run")
            except:
                pass
            
            return [{
                "bold": merged_bold,
                "mask": mask_files[0],
                "confounds": merged_confounds,
                "events": merged_events,
                "tags": entities,
                "tr": tr,
                "conditions": evs
            }]
        
        elif run_mode == "loop":
            logger.info(f"Estimation mode = {run_mode}; performing run-wise trial estimation")
            all_runs = []
            for i in range(len(bold_files)):

                if evs is None:
                    use_evs = get_event_names(events[i])
                else:
                    use_evs = evs
            
                logger.info(f"Including events = {use_evs} for run {i+1}/{len(bold_files)}")
                all_runs.append({
                    "bold": bold_files[i].path,
                    "mask": mask_files[i] if len(mask_files) > 1 else mask_files[0],
                    "confounds": confounds[i] if confounds is not None else None,
                    "events": events[i],
                    "tags": bold_files[i].entities,
                    "tr": tr,
                    "conditions": use_evs
                })

            return all_runs
        else:
            raise ValueError(f"Unknown run_mode: {run_mode}")
    else:
        return {
            "bold_files": [i.path for i in bold_files],
            "event_files": events,
            "tr": tr,
            "params": glm_kws,
            "tags": bold_files[0].entities
        }

def run_stglm_from_inputs(
        run_data_list,
        feature,
        workdir,
        confound_regexes=None,
        n_procs=1,
        derivatives=None,
        clean_run=False
    ):

    results = []
    for run in run_data_list:
        
        tags = run["tags"]

        sub_id = f"sub_{tags['subject']}"
        ses_id = f"ses_{tags['session']}" if "session" in tags else None
        run_id = f"run_{tags['run']}" if "run" in tags else None
        task_id = f"task_{tags['task']}" if "task" in tags else None


        # Build BIDS-style workflow path
        bold_prefix = f"bold"
        if ses_id:
            bold_prefix += f"_{ses_id}"
        if task_id:
            bold_prefix += f"_{task_id}"
        if run_id:
            bold_prefix += f"_{run_id}"

        bold_wf_name = f"{bold_prefix}_wf"
        subject_wf_name = f"{sub_id}_wf"

        workdir_i = (
            Path(workdir)
            / "stglm_wf"
            / subject_wf_name
            / bold_wf_name
        )

        if clean_run:
            logger.warning(f"Removing all directories under '{str(workdir_i)}'")
            shutil.rmtree(workdir_i, ignore_errors=True)

        if run["confounds"] is not None:
            if confound_regexes is not None:
                logger.info("Running confound workflow")
                conf_wf = init_confounds_select_wf()
                conf_wf.base_dir = workdir_i

                conf_input = conf_wf.get_node("inputnode").inputs
                conf_input.bold = run["bold"]
                conf_input.tags = run["tags"]
                conf_input.mask = run["mask"]
                conf_input.confounds = run["confounds"]
                conf_input.confound_names = confound_regexes

                conf_result = conf_wf.run()
                conf_tsv = utils.get_node_by_name(conf_result, "selectcolumns").result.outputs.out_with_header
            else:
                conf_tsv = run["confounds"]
                logger.info(f"Confound file: '{conf_tsv}'")
        else:
            conf_tsv = None

        # add conditions to feature
        setattr(feature, "conditions", run["conditions"])

        # Run single-trials
        stglm_wf = init_singletrials_wf(
            condition_files=run["events"],
            condition_units="secs",
            feature=feature,
            workdir=workdir_i,
            derivatives=derivatives,
            confound_file=conf_tsv
        )
        stglm_wf.base_dir = workdir_i
        
        st_input = stglm_wf.get_node("inputnode").inputs
        st_input.bold = run["bold"]
        st_input.mask = run["mask"]
        st_input.repetition_time = run["tr"]
        st_input.tags = run["tags"]
        st_input.vals = {"scan_start": 0.0}
        
        kwargs = {}
        if n_procs > 1:
            kwargs = {
                "plugin": "MultiProc",
                "plugin_args": {
                    "n_procs": n_procs
                }
            }

        result = stglm_wf.run(**kwargs)
        results.append(result)

    return results


def run_glmsingle_from_inputs(
    bold_files: list,
    event_files: list,
    tr: float,
    include_trials: list[str] | None=None,
    output_dir=None,
    stim_duration=0,
    **kwargs
):
    from glmsingle.glmsingle import GLM_single

    logger.info(f"Loading BOLD files")
    funcs = [
        nb.load(i).get_fdata(dtype=float) for i in bold_files
    ]

    # derive chuncklen depending on free memory

    # load the designs
    logger.info(f"Loading event-files")
    dms = []
    event_dfs = []
    for ev_file in event_files:
        df = pd.read_csv(ev_file, sep="\t")
        dm = conditions.events_to_glmsingle_dm(
            df,
            funcs[0].shape[-1],
            tr=tr,
            include_trials=include_trials
        )
        dms.append(dm)
        event_dfs.append(df)

    tags = kwargs.pop("tags", None)
    base_path = None
    if tags is not None:
        base_path = f"sub-{tags['subject']}"
        ses_id = f"ses-{tags['session']}" if "session" in tags else None

        if ses_id is not None:
            base_path = opj(base_path, ses_id)

    # store image of design matrices
    final_output = opj(
        output_dir,
        base_path
    )

    logger.info(f"Final output directory: {final_output}")
    if not os.path.exists(opj(final_output, "figures")):
        os.makedirs(opj(final_output, "figures"), exist_ok=True)

    # save trial list:
    trial_list = []
    for df in event_dfs:
        df = df[df['trial_type'].isin(include_trials)].copy().sort_values("onset")
        trial_list += df["trial_type"].values.tolist()
    
    logger.info(f"Saving trial list with {len(trial_list)} trial IDs")
    np.savetxt(
        opj(final_output,'trial_list.txt'),
        trial_list,
        fmt="%s"
    )

    # save design matrices
    _ = utils.plot_design_matrices(
        dms,
        output_dir=opj(final_output, "figures")
    )

    # run fitter
    logger.info(f"Start fitting..")
    gs = GLM_single(**kwargs)
    gs.fit(
        dms,
        funcs,
        stim_duration,
        tr,
        outputdir=final_output,
        figuredir=opj(final_output, "figures")
    )

    # save output as nifti-files
    logger.info(f"Saving outputs to '{final_output}'")
    models = {
        "typeb": opj(final_output, "TYPEB_FITHRF.npy"),
        "typed": opj(final_output, "TYPED_FITHRF_GLMDENOISE_RR.npy"),
    }

    results = {}
    for key, val in models.items():
        if os.path.exists(val):
            utils.save_glmsingle_outputs(
                np.load(val, allow_pickle=True).item(),
                bold_files[0],
                final_output,
                prefix=f"model-{key}"
            )

            results[key] = get_file_from_substring(f"model-{key}_beta-", final_output)

    return results


def valid_file_path(path):
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"File '{path}' does not exist.")
    return path


def args_parser():

    parser = argparse.ArgumentParser(
        prog="stglm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
stGLM is a Python-based CLI tool and workflow engine designed to run single-trial GLMs (General Linear Models) on fMRI data using the BIDS standard.
It supports LSA (Least Squares All), LSS (Least Squares Separate), and GLMsingle approaches to model trial-specific brain responses.

By default, we run LSA and LSS using FEAT wrapper in nipype-workflows.
These workflows are pretty much a copy of the single-trial implementation in the upcoming HALFpipe release to ensure compatibility.

To allow more flexibility, stGLM also supports GLMsingle with the '--glm-single' flag.

If you want more custom options, generate a JSON file with relevant arguments, such as:

```
{
    "wantlibrary": 1,           # use HRF library
    "wantglmdenoise": 1,        # use GLMdenoise
    "wantfracridge": 1,         # use fractional ridge regression
    "wantlss": 1,               # use LSS method for model B
    "chunklen": 50000,          # should not be so large that you run out of RAM
    "wantpercentbold": 1,       # convert to percent change
    "n_pcs": 10,                # maximum number of PCs to enter into the model
    "brainthresh": [99, 0.1],   # [A, B] where A is a percentile for voxel
         intensity values and B is a fraction to apply to the percentile
    "wantautoscale": 1          # automatically scale and offset
         the model estimates
}
```

After saving this as a .json-file, you can run:
```
stglm [] --config /path/to/config.json
```

For more custom implementation, this package might not be for you.
    """)
    parser.add_argument(
        '--project-dir',
        required=True,
        help='Path to BIDS project'
    )

    parser.add_argument(
        '--deriv-dir',
        help='Path to output folder project (default = project_dir/stglm or project_dir/glmsingle if --glm-single is specified)'
    )
        
    parser.add_argument(
        '--subject',
        required=True,
        help="Subject ID (e.g., '01')"
    )

    parser.add_argument(
        '--confounds-subset',
        choices=[
            "ICA-AROMA",
            "Motion parameters",
            "Derivatives of motion parameters",
            "Motion parameters squared",
            "Derivatives of motion parameters squared",
            "Motion scrubbing",
            "aCompCor (top five components)",
            "White matter signal",
            "CSF signal",
            "Global signal",
            "Motion acompcor",
            "ENIGMA"
        ],
        default="ENIGMA",
        help="Select subset of nuisance regressors based on HALFpipe's terminology (default = 'ENIGMA', which represents an ensemble of 'Motion parameter', 'Derivatives of motion parameters', 'Motion parameters squared', 'Derivatives of motion parameters squared', and 'aCompCor (top five components)')"
    )

    parser.add_argument(
        '--confounds-suffix',
        choices=['timeseries', 'physio', 'custom'],
        default='timeseries',
        help="'timeseries' refers to the default output from fMRIprep, 'physio' a set of physiological regressors (e.g., output from TAPAS), and 'custom' is something you specified yourself"
    )

    parser.add_argument(
        '--evs',
        nargs='+',
        required=False,
        default=None,
        help="Trial types to include (e.g., --evs CS- CS+_noUCS CS+_withUCS). Default is None, which means all trial types will be included"
    )    

    parser.add_argument(
        '--workdir',
        default='/tmp',
        help="Working directory"
    )

    parser.add_argument(
        '--run-mode', 
        choices=['loop', 'concat'],
        default='loop',
        help="If multiple run are present, we can estimate them in a single go (which comes with risks regarding noise and scaling) or perform run-wise trial-estimation"
    )

    parser.add_argument(
        '--hrf',
        choices=['dgamma', 'dgamma_with_derivs', 'flobs'],
        default='dgamma',
        help="Type of HRF to use following HALFpipe convention"
    )

    parser.add_argument(
        '--lss-mode',
        choices=['cond', 'nuis'],
        default='cond',
        help="Mode of LSS-estimation: in 'nuis', the design matrix consists of [trial-of-interest, other] regressors. In 'cond', the design contains the trial-of-interest + trial type-specific regressors containing trials-of-no-interest [trial-of-interest, trial_type1, trial_type2, trial_type3]"
    )

    parser.add_argument(
        '--task',
        default='HRA',
        help='Task label (e.g., rest, HRA, etc.)'
    )

    parser.add_argument(
        '--desc',
        choices=['preproc', 'masked', 'pybest'],
        default='masked',
        help="'preproc' refers to the default output from fMRIprep, 'masked' is the skullstripped version using 'call_fprepmask', 'pybest' is the full output from Pybest"
    )
    
    parser.add_argument(
        '--space',
        default='T1w',
        help='Spatial normalization space (e.g., T1w, MNI152NLin2009cAsym, or "func" for original BOLD space)'
    )

    parser.add_argument(
        '--n-procs',
        default=1,
        help='Number of processes (CPUs) to use (default = 1)'
    )    
    parser.add_argument(
        '--config',
        default=None,
        type=valid_file_path,
        help='Specify a json-file with options for GLMsingle'
    )      

    parser.add_argument(
        '--no-confounds',
        action='store_true',
        help="Do not use any confounds (assuming input data is already denoised or '--glm-single' is specified [utilizes GLMdenoise internally])"
    ) 

    parser.add_argument(
        '--clean',
        action='store_true',
        help="Remove full subject-specific working directory containing intermediate files (clean run)"
    ) 

    parser.add_argument(
        '--glm-single',
        action='store_true',
        help="Run GLMsingle, instead of 'standard' LSA/LSS. Implements GLMdenoise, HRF optimization, and beta-regularization. Requires multiple runs and repeats!"
    )

    parser.add_argument(
        '--stim-duration',
        type=float,
        default=0.0,
        help='Stimulus duration (default = 0) for GLMsingle'
    )

    return parser.parse_args()

def main():

    logger.info("Starting single-trial GLM workflow")
    args = args_parser()

    space = args.space
    space_kwargs = {} if space == "func" else {"space": space}

    logger.info(f"Fetching BIDSLayout of {args.project_dir}")
    layout = BIDSLayout(
        args.project_dir,
        derivatives=True,
        validate=False
    )

    run_glmsingle = args.glm_single
    if args.deriv_dir is None:
        if run_glmsingle:
            derivatives_base_dir = "glmsingle"
        else:
            derivatives_base_dir = "stglm"

        derivatives_dir = Path(args.project_dir) / "derivatives" / derivatives_base_dir
    else:
        derivatives_dir = args.deriv_dir

    logger.info(layout)

    glm_kws = {}
    if isinstance(args.config, str):
        logger.info(f"Using config file: '{args.config}'")
        import json
        with open(args.config, "rb") as f:
            glm_kws = json.load(f)

    runs = build_inputs(
        layout,
        subject=args.subject,
        run_mode=args.run_mode,
        desc=args.desc,
        task=args.task,
        glm_single=run_glmsingle,
        no_confounds=args.no_confounds,
        confounds_suffix=args.confounds_suffix,
        glm_kws=glm_kws,
        evs=args.evs,
        **space_kwargs
    )

    confound_subtype = args.confounds_subset
    if not run_glmsingle:
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
        if not args.no_confounds:
            if args.confounds_suffix == "timeseries":

                # add default that i use for ENIGMA
                if confound_subtype != "ENIGMA":
                    # check if input is list
                    if "," in confound_subtype:
                        search_list = confound_subtype.split(",")
                    else:
                        search_list = [confound_subtype]
                else:
                    search_list = [
                        "Motion parameters",
                        "Derivatives of motion parameters",
                        "Motion parameters squared",
                        "Derivatives of motion parameters squared",
                        "aCompCor (top five components)"
                    ]

                confound_regexes = [confound_options[cs] for cs in search_list]
                
                logger.info(f"Using '{confound_subtype}' ({confound_regexes}) from confound file(s)")
            else:
                if args.confounds_suffix in ["physio", "custom"]:
                    confound_subtype = args.confounds_suffix
                    logger.info(f"Not subselecting confound files; using the data as is (type={confound_subtype})")
                else:
                    raise ValueError(f"Confound suffix must be one of 'timeseries', 'physio', or 'custom', not '{args.confound_suffix}'")
        else:
            logger.info("Not including any confounds; assuming data is already treated properly..")

        ddict = {
            "hrf": args.hrf,
            "mode": args.lss_mode,
            "confounds": confound_subtype,
            "confounds_regex": confound_regexes,
        }

        feature = SimpleNamespace(**ddict)
        _ = run_stglm_from_inputs(
            run_data_list=runs,
            feature=feature,
            workdir=args.workdir,
            confound_regexes=confound_regexes,
            n_procs=int(args.n_procs),
            derivatives=str(derivatives_dir),
            clean_run=args.clean
        )
    else:

        try:
            from glmsingle.glmsingle import GLM_single
        except ImportError:
            logger.error(
                "GLMsingle is required for this step. "
                "Please install it: 'pip install git+https://github.com/cvnlab/GLMsingle'"
            )
            raise

        logger.info("Estimating single-trial beta-weights with GLMsingle")
        # set stim duration        
        if not "stim_duration" in list(runs.keys()):
            runs["stim_duration"] = float(args.stim_duration)

        if not "output_dir" in list(runs.keys()):
            runs["output_dir"] = derivatives_dir

        if args.evs is not None:
            runs["include_trials"] = args.evs

        # should contain dictionary with 'bold_files' and 'events', 'tr', 'params', 
        # 'stim_duration'
        _ = run_glmsingle_from_inputs(
            **runs
        )

if __name__ == "__main__":
    try:
        main()
        logger.info("stGLM finished without errors")
    except Exception as e:
        logger.error(f"stGLM failed with errors {e}")

