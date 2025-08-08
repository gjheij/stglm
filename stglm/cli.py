import argparse
from stglm import utils
from pathlib import Path
from bids import BIDSLayout
from types import SimpleNamespace
from stglm.confounds import init_confounds_select_wf
from stglm.single_trials import init_singletrials_wf
from stglm.logger import get_logger

logger = get_logger(__name__)

def build_inputs(
    layout,
    subject,
    run_mode="loop",
    task=None,
    desc="masked",
    no_confounds=False,
    **kwargs
    ):

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
        raise FileNotFoundError(f"No bold files found for subject {subject}")

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
            suffix='timeseries',
            extension='tsv',
            return_type='file'
        )

        if not confounds:
            raise FileNotFoundError(f"No confounds found for subject {subject}")
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
        raise FileNotFoundError(f"No events found for subject {subject}")

    entities = bold_files[0].entities
    tr = entities["RepetitionTime"]

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
        }]
    
    elif run_mode == "loop":
        logger.info(f"Estimation mode = {run_mode}; performing run-wise trial estimation")
        all_runs = []
        for i in range(len(bold_files)):
            all_runs.append({
                "bold": bold_files[i].path,
                "mask": mask_files[i] if len(mask_files) > 1 else mask_files[0],
                "confounds": confounds[i] if confounds is not None else None,
                "events": events[i],
                "tags": bold_files[i].entities,
                "tr": tr
            })
        return all_runs
    else:
        raise ValueError(f"Unknown run_mode: {run_mode}")

def run_stglm_from_inputs(
    run_data_list,
    feature,
    workdir,
    confound_regexes,
    n_procs=1,
    derivatives=None,
    ):

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

        if run["confounds"] is not None:
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
            conf_tsv = None

        # Run single-trials
        stglm_wf = init_singletrials_wf(
            condition_files=run["events"],
            condition_units="secs",
            feature=feature,
            workdir=workdir_i,
            derivatives=derivatives
        )
        stglm_wf.base_dir = workdir_i
        
        st_input = stglm_wf.get_node("inputnode").inputs
        st_input.bold = run["bold"]
        st_input.mask = run["mask"]
        st_input.repetition_time = run["tr"]
        st_input.tags = run["tags"]
        st_input.vals = {"scan_start": 0.0}
        st_input.confounds_selected = conf_tsv
        
        kwargs = {}
        if n_procs > 1:
            kwargs = {
                "plugin": "MultiProc",
                "plugin_args": {
                    "n_procs": n_procs
                }
            }

        try:
            stglm_wf.run(**kwargs)
            logger.info("stGLM completed without errors")
        except Exception as e:
            raise RuntimeError(f"stGLM failed: {e}")
        

def args_parser():

    parser = argparse.ArgumentParser(description="Run single-trial GLM using LSA and LSS estimation")
    parser.add_argument(
        '--project-dir',
        required=True,
        help='Path to BIDS project'
    )

    parser.add_argument(
        '--deriv-dir',
        help='Path to output folder project (default = project_dir/stglm)'
    )
        
    parser.add_argument(
        '--subject',
        required=True,
        help="Subject ID (e.g., '01')"
    )

    parser.add_argument(
        '--confound-subset',
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
        ],
        default="Motion parameters",
        help="Select subset of nuisance regressors based on HALFpipe's terminology (default = 'Motion parameters')"
    )

    parser.add_argument(
        '--evs',
        nargs='+',
        required=True,
        default=None,
        help="Trial types to include (e.g., --evs CS- CS+_noUCS CS+_withUCS)"
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
        choices=['preproc', 'masked'],
        default='masked',
        help="'preproc' refers to the default output from fMRIprep, 'masked' is the skullstripped version using 'call_fprepmask'"
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
        '--no-confounds',
        action='store_true',
        help="Do not use any confounds (assuming input data is already denoised)"
    ) 

    return parser.parse_args()

def main():

    logger.info("Starting single-trial GLM workflow")
    args = args_parser()

    space = args.space
    space_kwargs = {} if space == "func" else {"space": space}

    layout = BIDSLayout(args.project_dir, derivatives=True)

    if args.deriv_dir is None:
        derivatives_dir = Path(args.project_dir) / "derivatives" / "stglm"
    else:
        derivatives_dir = args.deriv_dir

    logger.info(layout)

    runs = build_inputs(
        layout,
        subject=args.subject,
        run_mode=args.run_mode,
        desc=args.desc,
        task=args.task,
        no_confounds=args.no_confounds,
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
    if not args.no_confounds:
        confound_regexes = [confound_options[args.confound_subset]]
        logger.info(f"Using '{args.confound_subset}' ({confound_regexes}) from confound files")
    else:
        logger.info("Not including any confounds; assuming data is already treated properly..")

    logger.info(f"Including conditions: {args.evs}")
    ddict = {
        "hrf": args.hrf,
        "mode": args.lss_mode,
        "confounds": args.confound_subset,
        "confounds_regex": confound_regexes,
        "conditions": args.evs
    }

    feature = SimpleNamespace(**ddict)
    run_stglm_from_inputs(
        run_data_list=runs,
        feature=feature,
        workdir=args.workdir,
        confound_regexes=confound_regexes,
        n_procs=int(args.n_procs),
        derivatives=str(derivatives_dir)
    )

if __name__ == "__main__":
    main()
