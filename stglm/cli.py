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

def _entity_match(reference, candidate, keys=(
    "subject", "session", "task", "acquisition", "run", "echo", "space", "resolution"
)):
    """Return True when all entities shared by two BIDS files agree."""
    ref_entities = reference.entities
    cand_entities = candidate.entities
    for key in keys:
        ref_value = ref_entities.get(key)
        cand_value = cand_entities.get(key)
        if ref_value is not None and cand_value is not None and str(ref_value) != str(cand_value):
            return False
    return True


def _find_bids_match(reference, candidates, label, required=True):
    matches = [candidate for candidate in candidates if _entity_match(reference, candidate)]
    if not matches:
        if required:
            raise FileNotFoundError(f"No matching {label} for {reference.path}")
        return None
    if len(matches) > 1:
        # Prefer the candidate sharing the largest number of concrete entities.
        def score(candidate):
            return sum(
                reference.entities.get(key) is not None
                and candidate.entities.get(key) is not None
                and str(reference.entities[key]) == str(candidate.entities[key])
                for key in ("session", "task", "acquisition", "run", "echo", "space", "resolution")
            )
        matches.sort(key=score, reverse=True)
        if len(matches) > 1 and score(matches[0]) == score(matches[1]):
            raise ValueError(
                f"Ambiguous {label} match for {reference.path}: "
                f"{[candidate.path for candidate in matches if score(candidate) == score(matches[0])]}"
            )
    return matches[0]


def _run_number(value):
    """Normalize a BIDS run index without accepting zero or nonnumeric labels."""
    text = str(value)
    if text.startswith('run-'):
        text = text[4:]
    if not text.isascii() or not text.isdigit() or int(text) < 1:
        raise ValueError('Run must be a positive BIDS index, e.g. 01 or run-01')
    return int(text)


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
    glm_kws=None,
    gm_probseg_threshold=0.5,
    runs=None,
    **kwargs,
):
    """Build run inputs and attach fMRIPrep GM probability maps.

    tSNR is intentionally not queried from the derivatives because fMRIPrep does
    not normally emit it. It is computed later from each run's BOLD image.
    """
    glm_kws = dict(glm_kws or {})
    selected_runs = None
    if runs is not None:
        selected_runs = sorted({_run_number(value) for value in runs})
        if not selected_runs:
            raise ValueError('Run selection must not be empty')
        kwargs = dict(kwargs, run=selected_runs)
    if glm_single:
        no_confounds = True

    bold_files = layout.get(
        subject=subject, task=task, desc=desc, suffix="bold",
        extension="nii.gz", return_type="object", **kwargs
    )
    if not bold_files:
        raise FileNotFoundError(f"No bold files found for sub-{subject}")
    if selected_runs is not None:
        found_runs = {_run_number(item.entities['run']) for item in bold_files if item.entities.get('run') is not None}
        missing_runs = sorted(set(selected_runs) - found_runs)
        if missing_runs:
            raise FileNotFoundError(f'Requested BOLD runs missing for sub-{subject}: {missing_runs}')
        logger.info('Selected BOLD runs for sub-%s: %s', subject, selected_runs)

    # Exclude unmatched runs before reading TR, building QC inputs or fitting.
    # Ambiguous matches still raise: they cannot be resolved by dropping data.
    events = layout.get(
        subject=subject, task=task, suffix="events", extension="tsv",
        return_type="object"
    )
    paired_bold = []
    paired_events = []
    for bold in bold_files:
        event = _find_bids_match(bold, events, 'events', required=False)
        if event is None:
            logger.warning('Skipping BOLD run without matching events: %s', bold.path)
        else:
            paired_bold.append(bold)
            paired_events.append(event)
    if not paired_bold:
        raise FileNotFoundError(f'No BOLD runs with matching events remain for sub-{subject}')
    if len(paired_bold) != len(bold_files):
        logger.warning('sub-%s: retained %d/%d BOLD runs after events matching',
                       subject, len(paired_bold), len(bold_files))
    bold_files = paired_bold
    events = paired_events

    mask_files = layout.get(
        subject=subject, task=task, suffix="mask", extension="nii.gz",
        return_type="object", **kwargs
    )
    if not mask_files:
        raise FileNotFoundError(f"No masks found for sub-{subject}")

    confounds = []
    if not no_confounds:
        confounds = layout.get(
            subject=subject, task=task, suffix=confounds_suffix,
            extension="tsv", return_type="object"
        )
        if not confounds:
            raise FileNotFoundError(f"No confounds found for sub-{subject}")
        confounds = [_find_bids_match(bold, confounds, 'confounds') for bold in bold_files]
        logger.info("Found %d files with suffix='%s'", len(confounds), confounds_suffix)

    # fMRIPrep/sMRIPrep normally provides label-GM_probseg in anatomical or
    # standard space. The QC helper resamples it to each BOLD/tSNR grid.
    gm_probsegs = layout.get(
        subject=subject, label="GM", suffix="probseg", extension="nii.gz",
        return_type="object", **kwargs
    )
    if not gm_probsegs:
        # Avoid task/desc restrictions inherited through kwargs for anatomical maps.
        gm_probsegs = layout.get(
            subject=subject, label="GM", suffix="probseg", extension="nii.gz",
            return_type="object"
        )
    if not gm_probsegs and not glm_single:
        raise FileNotFoundError(
            f"No label-GM_probseg found for sub-{subject}; required for MeanGMTSNR"
        )

    first_metadata = layout.get_metadata(bold_files[0].path)
    tr = bold_files[0].entities.get("RepetitionTime") or first_metadata.get("RepetitionTime")
    if tr is None:
        raise ValueError(f"RepetitionTime is unavailable for {bold_files[0].path}")

    if glm_single:
        matched_events = events
        for bold in bold_files:
            run_tr = layout.get_metadata(bold.path).get("RepetitionTime", tr)
            if not np.isclose(float(run_tr), float(tr), rtol=0, atol=1e-5):
                raise ValueError("GLMsingle requires a common TR across runs")
        # Read fMRIPrep confounds for motion QC independently of which nuisance
        # regressors GLMsingle actually fits.
        qc_confounds = layout.get(subject=subject,task=task,suffix="timeseries",
                                 desc="confounds",extension="tsv",return_type="object")
        qc_runs = []
        for bold in bold_files:
            fd = _find_bids_match(bold,qc_confounds,"FD confounds for QC",required=False)
            gm = _find_bids_match(bold,gm_probsegs,"GM probability map for QC",required=False)
            mask = _find_bids_match(bold,mask_files,"brain mask for QC",required=False)
            qc_runs.append(dict(bold=bold.path,tr=float(tr),metadata=layout.get_metadata(bold.path),
                                fd_confounds=fd.path if fd is not None else None,
                                gm_probseg=gm.path if gm is not None else None,
                                mask=mask.path if mask is not None else None))
        return {
            "qc_runs": qc_runs,
            "bold_files": [item.path for item in bold_files],
            "event_files": [item.path for item in matched_events],
            "tr": tr,
            "params": glm_kws,
            "tags": dict(bold_files[0].entities),
        }

    if run_mode == "concat":
        logger.info("Estimation mode = concat; concatenating all runs")
        merged_bold = utils.merge_bold([item.path for item in bold_files])
        merged_confounds = None if no_confounds else utils.merge_confounds(
            [item.path for item in confounds], out_file="/tmp/merged_confounds.tsv"
        )
        merged_events = utils.merge_events(
            [item.path for item in events], bold_files, tr,
            out_file="/tmp/merged_events.tsv"
        )
        use_evs = get_event_names(merged_events) if evs is None else evs
        tags = dict(bold_files[0].entities)
        tags.pop("run", None)
        gm_probseg = _find_bids_match(bold_files[0], gm_probsegs, "GM probability map")
        return [{
            "bold": merged_bold,
            "mask": _find_bids_match(bold_files[0], mask_files, "mask").path,
            "confounds": merged_confounds,
            "events": merged_events,
            "tags": tags,
            "tr": tr,
            "conditions": use_evs,
            "gm_probseg": gm_probseg.path,
            "gm_threshold": float(gm_probseg_threshold),
            "metadata": first_metadata,
            "vals": {"scan_start": 0.0},
        }]

    if run_mode != "loop":
        raise ValueError(f"Unknown run_mode: {run_mode}")

    logger.info("Estimation mode = loop; performing run-wise trial estimation")
    all_runs = []
    for index, bold_file in enumerate(bold_files, start=1):
        event_file = events[index-1]
        mask_file = _find_bids_match(bold_file, mask_files, "mask")
        confound_file = None if no_confounds else confounds[index-1]
        gm_probseg = _find_bids_match(bold_file, gm_probsegs, "GM probability map")
        metadata = layout.get_metadata(bold_file.path)
        run_tr = bold_file.entities.get("RepetitionTime") or metadata.get("RepetitionTime") or tr
        use_evs = get_event_names(event_file.path) if evs is None else evs
        logger.info("Including events = %s for run %d/%d", use_evs, index, len(bold_files))
        all_runs.append({
            "bold": bold_file.path,
            "mask": mask_file.path,
            "confounds": confound_file.path if confound_file is not None else None,
            "events": event_file.path,
            "tags": dict(bold_file.entities),
            "tr": run_tr,
            "conditions": use_evs,
            "gm_probseg": gm_probseg.path,
            "gm_threshold": float(gm_probseg_threshold),
            "metadata": metadata,
            "vals": {"scan_start": 0.0},
        })
    return all_runs

def run_stglm_from_inputs(
    run_data_list,
    feature,
    workdir,
    confound_regexes=None,
    n_procs=1,
    derivatives=None,
    clean_run=False,
    fd_threshold=0.2,
    tsnr_regress_poly=2,
):
    """Run the single-trial GLM workflow for each run.

    tSNR is computed inside the Nipype workflow from each run BOLD image.
    """
    results = []

    for run in run_data_list:
        tags = run["tags"]

        sub_id = f"sub_{tags['subject']}"
        ses_id = f"ses_{tags['session']}" if tags.get("session") else None
        run_id = f"run_{tags['run']}" if tags.get("run") else None
        task_id = f"task_{tags['task']}" if tags.get("task") else None

        # Build the workflow working-directory hierarchy.
        bold_prefix = "bold"
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
            logger.warning(
                "Removing all directories under '%s'",
                workdir_i,
            )
            shutil.rmtree(workdir_i, ignore_errors=True)

        full_confounds_tsv = run.get("confounds")
        selected_confounds_tsv = None

        if full_confounds_tsv is not None:
            if confound_regexes is not None:
                logger.info("Running confound selection workflow")

                conf_wf = init_confounds_select_wf()
                conf_wf.base_dir = str(workdir_i)

                conf_input = conf_wf.get_node("inputnode").inputs
                conf_input.bold = run["bold"]
                conf_input.tags = tags
                conf_input.mask = run["mask"]
                conf_input.confounds = full_confounds_tsv
                conf_input.confound_names = confound_regexes

                conf_result = conf_wf.run()
                selectcolumns = utils.get_node_by_name(
                    conf_result,
                    "selectcolumns",
                )
                selected_confounds_tsv = (
                    selectcolumns.result.outputs.out_with_header
                )

                logger.info(
                    "Selected confound file: '%s'",
                    selected_confounds_tsv,
                )
            else:
                # Preserve the existing behavior: use the full confounds
                # table as the model nuisance-regressor input.
                selected_confounds_tsv = full_confounds_tsv
                logger.info(
                    "Confound file: '%s'",
                    selected_confounds_tsv,
                )

        # Conditions may differ by run.
        setattr(feature, "conditions", run["conditions"])

        stglm_wf = init_singletrials_wf(
            condition_files=run["events"],
            condition_units="secs",
            feature=feature,
            workdir=workdir_i,
            derivatives=derivatives,

            # Selected regressors used by SpecifyModel.
            confound_file=selected_confounds_tsv,

            # Full confounds table used for framewise-displacement metrics.
            confounds_file=full_confounds_tsv,

            # tSNR is computed from BOLD; GM is defined by fMRIPrep probseg.
            gm_probseg=run.get("gm_probseg"),
            gm_threshold=float(run.get("gm_threshold", 0.5)),
            fd_threshold=float(fd_threshold),
            tsnr_regress_poly=int(tsnr_regress_poly),
        )
        stglm_wf.base_dir = str(workdir_i)

        st_input = stglm_wf.get_node("inputnode").inputs
        st_input.bold = run["bold"]
        st_input.mask = run["mask"]
        st_input.repetition_time = run["tr"]
        st_input.tags = tags

        vals = dict(run.get("vals") or {})
        vals.setdefault("scan_start", 0.0)

        # Preserve preprocessing provenance separately from the Model block.
        # Only report settings that this CLI can establish from the actual run.
        setting = dict(vals.get("setting") or {})
        setting["Space"] = tags.get("space") or "native"
        setting["ConfoundsRemoval"] = list(confound_regexes or [])
        setting["ICAAROMA"] = bool(
            getattr(feature, "confounds", None) == "ICA-AROMA"
            or any(regex == "ICA-AROMA" for regex in (confound_regexes or []))
        )

        high_pass_cutoff = getattr(feature, "high_pass_filter_cutoff", None)
        if high_pass_cutoff is not None and np.isfinite(high_pass_cutoff):
            setting["BandpassFilter"] = {
                "HighPassWidth": float(high_pass_cutoff),
                "LowPassWidth": None,
                "Type": "gaussian",
            }

        vals["setting"] = setting
        st_input.vals = vals

        metadata = run.get("metadata")
        if metadata is not None:
            st_input.metadata = metadata

        run_kwargs = {}
        if n_procs > 1:
            run_kwargs = {
                "plugin": "MultiProc",
                "plugin_args": {
                    "n_procs": n_procs,
                },
            }

        result = stglm_wf.run(**run_kwargs)
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
    from stglm.glmsingle_io import fit_glmsingle
    return fit_glmsingle(bold_files, event_files, tr, include_trials=include_trials,
                         output_dir=output_dir, stim_duration=stim_duration, **kwargs)


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
        '--run', '--runs', dest='runs', nargs='+', type=_run_number,
        help='Select BIDS run indices before events matching, e.g. --run 01 03 04. Default: all runs'
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

    parser.add_argument('--export-only', action='store_true',
        help='Export existing GLMsingle models without fitting; verify DESIGNINFO and any saved input plan')
    parser.add_argument('--split-trial-types', action='store_true',
        help='Also export per-run/condition 4D trial beta images with JSON and TSV metadata')
    parser.add_argument('--export-residuals', action='store_true',
        help='Reconstruct TYPE-D taskpoly/fullnuisance residuals per run; compatible with --export-only')
    parser.add_argument('--residual-units', choices=['beta', 'input'], default='beta',
        help='Residual voxel scaling: saved TYPE-D beta-compatible (default), or original BOLD units')
    parser.add_argument('--residual-chunk-voxels', type=int, default=2048,
        help='Voxel block size for disk-backed residual reconstruction (default: 2048)')
    parser.add_argument('--drop-invalid', action='store_true',
        help='Exclude out-of-grid events and events with less than 8 s after the rounded onset; save excluded_events.tsv without shifting timing')
    parser.add_argument('--event-duration', action='append', default=[], metavar='PATTERN=SECONDS',
        help='Validate a condition duration rule for GLMsingle; all modeled durations must equal --stim-duration')
    parser.add_argument('--additional-confounds', action='append', default=[], metavar='TSV_TEMPLATE',
        help='Additional GLMsingle regressors; repeat for files, with {subject}, {session}, {task}, {run} placeholders')
    parser.add_argument(
        '--stim-duration',
        type=float,
        default=0.0,
        help='Stimulus duration (default = 0) for GLMsingle'
    )

    parser.add_argument(
        '--fd-threshold',
        type=float,
        default=0.2,
        help='Framewise-displacement threshold in millimetres used for FDPerc (default = 0.2)'
    )

    parser.add_argument(
        '--gm-threshold',
        type=float,
        default=0.5,
        help='Threshold applied to the fMRIPrep GM probability map for MeanGMTSNR (default = 0.5)'
    )

    parser.add_argument(
        '--tsnr-regress-poly',
        type=int,
        default=2,
        help='Polynomial detrending order used when computing tSNR (default = 2)'
    )

    parser.add_argument(
        '--high-pass-filter-cutoff',
        type=float,
        default=None,
        help='Optional FSL high-pass cutoff in seconds. Omit to retain the workflow default.'
    )

    return parser.parse_args()

def main():

    logger.info("Starting single-trial GLM workflow")
    args = args_parser()

    if args.fd_threshold < 0:
        raise ValueError("--fd-threshold must be non-negative")
    if not 0.0 <= args.gm_threshold <= 1.0:
        raise ValueError("--gm-threshold must be between 0 and 1")
    if args.tsnr_regress_poly < 0:
        raise ValueError("--tsnr-regress-poly must be non-negative")
    if (
        args.high_pass_filter_cutoff is not None
        and args.high_pass_filter_cutoff <= 0
    ):
        raise ValueError("--high-pass-filter-cutoff must be positive")

    space = args.space
    space_kwargs = {} if space == "func" else {"space": space}

    logger.info(f"Fetching BIDSLayout of {args.project_dir}")
    layout = BIDSLayout(
        args.project_dir,
        derivatives=True,
        validate=False
    )

    run_glmsingle = args.glm_single
    if args.residual_chunk_voxels < 1:
        raise ValueError('--residual-chunk-voxels must be positive')
    if not run_glmsingle and (args.event_duration or args.additional_confounds or args.drop_invalid or args.export_only or args.split_trial_types or args.export_residuals):
        raise ValueError("GLMsingle event/confound/export options require --glm-single")
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
        gm_probseg_threshold=args.gm_threshold,
        runs=args.runs,
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
            "name": "SingleTrials",
            "setting": "SingleTrialsSetting",
            "type": "single_trials",
            "hrf": args.hrf,
            "mode": args.lss_mode,
            "confounds": confound_subtype,
            "confounds_regex": confound_regexes,
        }
        if args.high_pass_filter_cutoff is not None:
            ddict["high_pass_filter_cutoff"] = float(
                args.high_pass_filter_cutoff
            )

        feature = SimpleNamespace(**ddict)
        _ = run_stglm_from_inputs(
            run_data_list=runs,
            feature=feature,
            workdir=args.workdir,
            confound_regexes=confound_regexes,
            n_procs=int(args.n_procs),
            derivatives=str(derivatives_dir),
            clean_run=args.clean,
            fd_threshold=args.fd_threshold,
            tsnr_regress_poly=args.tsnr_regress_poly,
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
            **runs, event_durations=args.event_duration,
            additional_confounds=args.additional_confounds, drop_invalid=args.drop_invalid,
            export_only=args.export_only, split_trial_types=args.split_trial_types,
            export_residuals=args.export_residuals, residual_units=args.residual_units,
            residual_chunk_voxels=args.residual_chunk_voxels,
            fd_threshold=args.fd_threshold, gm_threshold=args.gm_threshold,
            tsnr_regress_poly=args.tsnr_regress_poly,
        )

if __name__ == "__main__":
    try:
        main()
        logger.info("stGLM finished without errors")
    except Exception as e:
        logger.error(f"stGLM failed with errors {e}")
        raise

