# -*- coding: utf-8 -*-
from pathlib import Path

import nipype.interfaces.utility as niu
from nipype.pipeline import engine as pe
from nipype.interfaces.base import isdefined
from nipype.interfaces.fsl import model as fslm
from nipype.interfaces.fsl import FEATModel, FILMGLS, Merge
import nipype.interfaces.utility as niu
import nipype.algorithms.modelgen as model
import os
import numpy as np
from math import isfinite
from .tsv import FillNA
from .utils import generate_sanitized_metadata
from .conditions import (
    ApplyConditionOffset,
    ParseConditionFile,
)
from nipype.interfaces.io import DataSink

def cond_bunch_to_lsa(subject_info):
    from nipype.interfaces.base import Bunch
    cb = subject_info[0] if isinstance(subject_info, list) else subject_info
    conditions, onsets, durations, amps = [], [], [], []
    for ci, cname in enumerate(cb.conditions):
        for ti, onset in enumerate(cb.onsets[ci]):
            tname = f"{cname}__{ti+1:03d}"
            conditions.append(tname)
            onsets.append([onset])
            durations.append([cb.durations[ci][ti]])
            if getattr(cb, "amplitudes", None):
                amps.append([cb.amplitudes[ci][ti]])
            else:
                amps.append([1.0])

    return [
        Bunch(
            conditions=conditions,
            onsets=onsets,
            durations=durations,
            amplitudes=amps,
            tmod=None,
            pmod=None,
            regressor_names=None,
            regressors=None
        )
    ]

def cond_bunch_to_lss(subject_info, mode='cond'):
    from nipype.interfaces.base import Bunch
    cb = subject_info[0] if isinstance(subject_info, list) else subject_info
    flat = []
    has_amp = getattr(cb, "amplitudes", None) is not None
    for ci, cname in enumerate(cb.conditions):
        for ti, onset in enumerate(cb.onsets[ci]):
            flat.append(dict(
                cname=cname,
                onset=onset,
                duration=cb.durations[ci][ti],
                amp=(cb.amplitudes[ci][ti] if has_amp else 1.0),
                trial_id=ti+1
            ))
    cond_names = cb.conditions
    out = []
    for k, tr in enumerate(flat):
        roi = f"{tr['cname']}__{tr['trial_id']:03d}"
        conditions = [roi]
        onsets = [[tr['onset']]]
        durations = [[tr['duration']]]
        amplitudes = [[tr['amp']]]
        others = [i for i in range(len(flat)) if i != k]
        if mode == 'nuis':
            conditions.append('others')
            onsets.append([flat[i]['onset'] for i in others])
            durations.append([flat[i]['duration'] for i in others])
            amplitudes.append([flat[i]['amp'] for i in others])
        elif mode == 'cond':
            for cname in cond_names:
                mask = [i for i in others if flat[i]['cname'] == cname]
                if mask:
                    conditions.append(f'{cname}_others')
                    onsets.append([flat[i]['onset'] for i in mask])
                    durations.append([flat[i]['duration'] for i in mask])
                    amplitudes.append([flat[i]['amp'] for i in mask])
        else:
            raise ValueError(mode)
        out.append(
            Bunch(
                conditions=conditions,
                onsets=onsets, 
                durations=durations,
                amplitudes=amplitudes,
                tmod=None,
                pmod=None,
                regressor_names=None,
                regressors=None
            )
        )
    return out


def pick_roi_betas(param_lists, idx=0):
    # param_lists: list of lists
    return [plist[idx] for plist in param_lists]


def _get_scan_start(vals) -> float:
    return float(vals["scan_start"])


def extract_conditions(subject_info):
    return {"TrialList": subject_info[0].conditions}


def init_singletrials_wf(
    workdir: Path | str,
    feature,
    condition_files,
    condition_units,
    derivatives=None
):
    """
    Create a workflow that runs GLMsingle then dynamically builds and sinks
    resultdicts for each model in the nifti_outputs.
    """
    name = "singletrials_wf"
    workflow = pe.Workflow(name=name)

    # Input & Output nodes
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'tags',
                'vals',
                'metadata',
                'bold',
                'mask',
                'repetition_time',
                'confounds_selected',
                'condition_names',
                'condition_files',
                'condition_units',
            ]
        ),
        name='inputnode'
    )

    inputnode.inputs.condition_names = feature.conditions
    inputnode.inputs.condition_files = condition_files
    inputnode.inputs.condition_units = condition_units

    #------------------------------------------------------------------
    # general
    parse_condition_file = pe.Node(
        ParseConditionFile(),
        name='parse_condition_file'
    )

    get_scan_start = pe.Node(
        niu.Function(
            input_names=['vals'],
            output_names=['scan_start'],
            function=_get_scan_start
        ),
        name='get_scan_start'
    )

    apply_condition_offset = pe.Node(
        ApplyConditionOffset(),
        name='apply_condition_offset'
    )

    # generate design from first level specification
    if feature.hrf == "dgamma":
        bases: dict[str, dict] = dict(dgamma=dict())
    elif feature.hrf == "dgamma_with_derivs":
        bases = dict(dgamma=dict(derivs=True))
    elif feature.hrf == "flobs":
        bfcustompath = Path(os.environ["FSLDIR"]) / "etc" / "default_flobs.flobs" / "hrfbasisfns.txt"
        assert bfcustompath.is_file()
        bases = dict(
            custom=dict(
                bfcustompath=str(bfcustompath),
                basisfnum=3,
            )
        )
    else:
        raise ValueError(f'HRF "{feature.hrf}" is not yet implemented')

    #
    fillna = pe.Node(
        FillNA(),
        name="fillna"
    )

    extract_conditions_node = pe.Node(
        niu.Function(
            input_names=["subject_info"],
            output_names=["trial_list"],
            function=extract_conditions,
        ),
        name="extract_conditions"
    )

    #------------------------------------------------------------------
    # LSA
    cond2lsa = pe.Node(
        niu.Function(
            input_names=['subject_info'],
            output_names=['lsa_subject_info'],
            function=cond_bunch_to_lsa
        ),
        name='cond2lsa'
    )

    spec_lsa = pe.Node(
        model.SpecifyModel(),
        name='specify_lsa'
    )

    spec_lsa.inputs.high_pass_filter_cutoff = np.inf  # disable if missing
    if hasattr(feature, "high_pass_filter_cutoff"):
        hpfc = feature.high_pass_filter_cutoff
        if isinstance(hpfc, float) and isfinite(hpfc):
            spec_lsa.inputs.high_pass_filter_cutoff = hpfc

    l1_lsa = pe.Node(
        fslm.Level1Design(
            model_serial_correlations=False,
            bases=bases,
        ),
        name='l1design_lsa'
    )

    feat_lsa = pe.Node(
        FEATModel(),
        name='featmodel_lsa',
    )

    film_lsa = pe.Node(
        FILMGLS(
            autocorr_noestimate=True
        ),
        name='filmgls_lsa',
    )

    def select_trial_betas(in_files, trial_list):
        # Number of trial regressors
        n_trials = len(trial_list["TrialList"]) if isinstance(trial_list, dict) else len(trial_list)
        return in_files[:n_trials]

    # limit to nr of trials
    pick_lsa = pe.Node(
        niu.Function(
            input_names=["in_files", "trial_list"],
            output_names=["out_files"],
            function=select_trial_betas
        ),
        name="pick_lsa"
    )

    merge_lsa = pe.Node(
        Merge(
            dimension='t',
            output_type='NIFTI_GZ',
            merged_file="merged_betas.nii.gz"
        ),
        name='merge_betas_lsa'
    )
    
    #------------------------------------------------------------------
    # LSS
    cond2lss = pe.Node(
        niu.Function(
            input_names=['subject_info','mode'],
            output_names=['lss_subject_infos'],
            function=cond_bunch_to_lss
        ),
        name='cond2lss'
    )

    spec_lss = pe.MapNode(
        model.SpecifyModel(),
        iterfield=['subject_info'],
        name='specify_lss'
    )

    spec_lss.inputs.high_pass_filter_cutoff = np.inf  # disable if missing
    if hasattr(feature, "high_pass_filter_cutoff"):
        hpfc = feature.high_pass_filter_cutoff
        if isinstance(hpfc, float) and isfinite(hpfc):
            spec_lss.inputs.high_pass_filter_cutoff = hpfc
            
    l1_lss = pe.MapNode(
        fslm.Level1Design(
            model_serial_correlations=False,
            bases=bases,
        ),
        iterfield=['session_info'],
        name='l1design_lss'
    )

    feat_lss = pe.MapNode(
        FEATModel(),
        iterfield=['fsf_file','ev_files'],
        name='featmodel_lss'
    )

    film_lss = pe.MapNode(
        FILMGLS(
            autocorr_noestimate=True
        ),
        iterfield=['design_file'],
        name='filmgls_lss'
    )

    pick_lss = pe.Node(
        niu.Function(
            input_names=['param_lists','idx'],
            output_names=['roi_betas'],
            function=pick_roi_betas
        ),
        name='pick_lss_betas'
    )
    
    pick_lss.inputs.idx = 0

    merge_lss = pe.Node(
        Merge(
            dimension='t',
            output_type='NIFTI_GZ',
            merged_file="merged_betas.nii.gz"
        ),
        name='merge_betas_lss'
    )

    # maybe just once in merge_resultdicts?
    if feature is not None:
        cond2lss.inputs.mode = feature.mode

    def format_bids_filenames(tags: dict):
        """
        Generate BIDS-style filenames for LSA and LSS outputs.
        """
        parts = []
        for key in ["subject", "session", "task", "run", "space"]:
            if key in tags:
                if key in ["subject", "session"]:
                    kk = key[:3]
                else:
                    kk = key
                    
                parts.append(f"{kk}-{tags[key]}")

        base = "_".join(parts)
        return [
            f"{base}_desc-lsa_betas.nii.gz",
            f"{base}_desc-lss_betas.nii.gz",
            f"{base}.json",
            f"{base}_desc-lss_betas.png",
            f"{base}_desc-lss_betas.mat",
            f"{base}_desc-lsa_betas.png",
            f"{base}_desc-lsa_betas.mat",            
        ]

    # format output names
    format_bids_names = pe.Node(
        niu.Function(
            input_names=["tags"],
            output_names=["out_names"],
            function=format_bids_filenames  # use the full function, not lambda
        ),
        name="format_bids_names"
    )

    rename_outputs  = pe.MapNode(
        niu.Rename(),
        name="rename_outputs",
        iterfield=["in_file", "format_string"]
    )

    # datasink
    if derivatives is None:
        derivatives = os.path.join(workdir, "derivatives", "stglm")

    derivatives = str(derivatives)

    # metadata
    sanitize_metadata = pe.Node(
        niu.Function(
            input_names=[
                "tags",
                "vals",
                "trial_list",
                "Conditions",
                "ConditionFile",
                "ConditionUnits",
                "BasedOn",
                "BetasLSA",
                "BetasLSS"
            ],
            output_names=["metadata"],
            function=generate_sanitized_metadata
        ),
        name="sanitize_metadata"
    )

    def write_metadata_json(in_dict, out_name="metadata.json"):
        import json
        from pathlib import Path

        # Get the current working directory of the node (runtime dir)
        cwd = Path.cwd()

        # Define the output path in the node’s directory
        out_path = cwd / out_name

        # Write JSON with pretty formatting
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(in_dict, f, indent=4, ensure_ascii=False)

        return str(out_path)

    write_json_node = pe.Node(
        niu.Function(
            input_names=["in_dict", "out_name"],
            output_names=["out_file"],
            function=write_metadata_json
        ),
        name="write_json_node"
    )

    def collect_generated_outputs(lsa_file, lss_file, lsa_mat, lsa_png, lss_mat, lss_png, json_file):
        return [lsa_file, lss_file, json_file, lss_png, lss_mat, lsa_png, lsa_mat]

    generate_outputs = pe.Node(
        niu.Function(
            input_names=[
                "lsa_file",
                "lss_file",
                "lsa_mat",
                "lsa_png",
                "lss_mat",
                "lss_png",
                "json_file"
            ],
            output_names=["file_list"],
            function=collect_generated_outputs
        ),
        name="generate_outputs"
    )

    def split_renamed_outputs(file_list):
        return tuple(file_list)

    split_outputs = pe.Node(
        niu.Function(
            input_names=["file_list"],
            output_names=[
                "out_lsa",
                "out_lss",
                "out_json",
                "out_lss_png",
                "out_lss_mat",
                "out_lsa_png",
                "out_lsa_mat"
            ],
            function=split_renamed_outputs
        ),
        name="split_outputs"
    )

    datasink = pe.Node(
        DataSink(
            parameterization=False,
            base_directory=str(derivatives),
        ),
        name="stglm_datasink"
    )

    def generate_sink_paths(tags):
        sub_id = tags.get("subject")
        ses_id = tags.get("session")

        if not sub_id:
            raise ValueError("Missing subject ID in tags")

        container = f"sub-{sub_id}"
        if ses_id:
            container += f"/ses-{ses_id}"

        func_path = "func"
        return container, func_path

    get_sink_paths = pe.Node(
        niu.Function(
            input_names=["tags"],
            output_names=["container", "func_path"],
            function=generate_sink_paths
        ),
        name="get_sink_paths"
    )

    select_first_png = pe.Node(
        niu.Function(
            input_names=["file_list"],
            output_names=["first_file"],
            function=lambda file_list: file_list[0] if file_list else None
        ),
        name="select_first_png"
    )

    select_first_mat = pe.Node(
        niu.Function(
            input_names=["file_list"],
            output_names=["first_file"],
            function=lambda file_list: file_list[0] if file_list else None
        ),
        name="select_first_mat"
    )    

    # if confound file is specified
    if isdefined(inputnode.inputs.confounds_selected):
        workflow.connect([
            (inputnode, fillna, [('confounds_selected', 'in_tsv')]),
            (fillna, spec_lsa,  [('out_no_header', 'realignment_parameters')]),
            (fillna, spec_lss,  [('out_no_header', 'realignment_parameters')]),
        ])

    # Connect the core graph
    workflow.connect([
        # scan start offset
        (inputnode, get_scan_start, [('vals','vals')]),
        (inputnode, format_bids_names, [("tags", "tags")]),
        (inputnode, get_sink_paths, [("tags", "tags")]),
        (inputnode, sanitize_metadata, [
            ("tags", "tags"),
            ("vals", "vals"),
            ("condition_names", "Conditions"),
            ("condition_files", "ConditionFile"),
            ("condition_units", "ConditionUnits"),
            ("bold", "BasedOn"),
        ]),

        # parse conditions
        (inputnode, parse_condition_file, [
            ('condition_names','condition_names'),
            ('condition_files','in_any')
        ]),
        (get_scan_start, apply_condition_offset, [('scan_start','scan_start')]),
        (parse_condition_file, apply_condition_offset, [('subject_info','subject_info')]),

        # LSA
        (inputnode, spec_lsa, [
            ('bold','functional_runs'),
            ('repetition_time','time_repetition'),
            ('condition_units', 'input_units')
        ]),
        (apply_condition_offset, cond2lsa, [('subject_info', 'subject_info')]),
        (cond2lsa, extract_conditions_node, [("lsa_subject_info", "subject_info")]),
        (cond2lsa,  spec_lsa, [('lsa_subject_info','subject_info')]),
        (spec_lsa,  l1_lsa,   [('session_info','session_info')]),
        (inputnode, l1_lsa,   [('repetition_time', 'interscan_interval')]),
        (l1_lsa,    feat_lsa, [
            ('fsf_files','fsf_file'),
            ('ev_files','ev_files')
        ]),
        (feat_lsa,  film_lsa, [('design_file','design_file')]),
        (inputnode, film_lsa, [('bold','in_file')]),
        (film_lsa,  pick_lsa, [("param_estimates", "in_files")]),
        (extract_conditions_node, pick_lsa, [("trial_list", "trial_list")]),
        (pick_lsa,  merge_lsa,[('out_files','in_files')]),

        # LSS
        (inputnode, spec_lss, [
            ('bold','functional_runs'),
            ('repetition_time','time_repetition'),
            ('condition_units', 'input_units')
        ]),
        (apply_condition_offset, cond2lss, [('subject_info', 'subject_info')]),
        (cond2lss,  spec_lss, [('lss_subject_infos','subject_info')]),
        (spec_lss,  l1_lss,   [('session_info','session_info')]),
        (inputnode, l1_lss,   [('repetition_time', 'interscan_interval')]),
        (l1_lss,    feat_lss, [
            ('fsf_files','fsf_file'),
            ('ev_files','ev_files')
        ]),
        (feat_lss,  film_lss, [('design_file','design_file')]),
        (inputnode, film_lss, [('bold','in_file')]),
        (film_lss,  pick_lss, [('param_estimates','param_lists')]),
        (pick_lss,  merge_lss,[('roi_betas','in_files')]),
        (feat_lss, select_first_png, [("design_image", "file_list")]),
        (feat_lss, select_first_mat, [("design_file", "file_list")]),

        # metadata
        (extract_conditions_node, sanitize_metadata, [("trial_list", "trial_list")]),
        (sanitize_metadata, write_json_node, [("metadata", "in_dict")]),

        # generate outputs
        (merge_lsa, generate_outputs, [("merged_file", "lsa_file")]),
        (merge_lss, generate_outputs, [("merged_file", "lss_file")]),
        (write_json_node, generate_outputs, [("out_file", "json_file")]),
        (feat_lsa, generate_outputs, [("design_file", "lsa_mat")]),
        (feat_lsa, generate_outputs, [("design_image", "lsa_png")]),
        (select_first_mat, generate_outputs, [("first_file", "lss_mat")]),
        (select_first_png, generate_outputs, [("first_file", "lss_png")]),

        # feed into renaming for later DataSink
        (generate_outputs, rename_outputs, [("file_list", "in_file")]),
        (format_bids_names, rename_outputs, [("out_names", "format_string")]),

        # split for proper parsing to DataSink
        (rename_outputs, split_outputs, [("out_file", "file_list")]),
        (get_sink_paths, datasink, [("container", "container")]),
    ])

    sink_map = [
        ("out_lsa", "func.@lsa"),
        ("out_lss", "func.@lss"),
        ("out_json", "func.@json"),
        ("out_lss_png", "func.@lss_png"),
        ("out_lss_mat", "func.@lss_mat"),
        ("out_lsa_png", "func.@lsa_png"),
        ("out_lsa_mat", "func.@lsa_mat"),
    ]

    for out_field, sink_path in sink_map:
        workflow.connect(split_outputs, out_field, datasink, sink_path)

    return workflow