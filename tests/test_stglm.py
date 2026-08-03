from types import SimpleNamespace
from stglm import single_trials, utils, confounds
from bids import BIDSLayout
import os
opj = os.path.join

def test_stglm():
    proj_dir = "/mnt/d/fMRI/HRA"
    sub_id = "015"
    layout = BIDSLayout(proj_dir, derivatives=opj(proj_dir, "derivatives", "fmriprep"))
    tsv_files = layout.get(
        subject=sub_id,
        suffix='events',
        extension='tsv',
        return_type='file'
    )

    confound_files = layout.get(
        subject=sub_id,
        suffix='timeseries',
        extension='tsv',
        return_type='file'
    )

    bold_files = layout.get(
        subject=sub_id,
        suffix='bold',
        space="T1w",
        extension='nii.gz',
        return_type='object',
        desc="masked"
    )

    bold_masks = layout.get(
        subject=sub_id,
        suffix='mask',
        space="T1w",
        extension='nii.gz',
        return_type='file'
    )

    entities = bold_files[0].entities
    source_file = bold_files[0].path            

    confound_options = {
        "ICAs-AROMA": "ICA-AROMA",
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

    search_list = [
        "Motion parameters",
        "Derivatives of motion parameters",
        "Motion parameters squared",
        "Derivatives of motion parameters squared",
        "aCompCor (top five components)"
    ]

    confound_regexes = [confound_options[cs] for cs in search_list]
    
    confounds_wf = confounds.init_confounds_select_wf()
    confounds_wf.base_dir = "/tmp/test"

    inputnode = confounds_wf.get_node("inputnode").inputs
    inputnode.bold = source_file
    inputnode.tags = entities
    inputnode.mask = bold_masks[0]
    inputnode.confounds = confound_files[0]
    inputnode.confound_names = confound_regexes

    res = confounds_wf.run()
    
    # fetch new tsv file
    in_tsv = utils.get_node_by_name(res, "selectcolumns").result.outputs.out_with_header

    ddict = {
        "name": "SingleTrials",
        "setting": "SingleTrialsSetting",
        "type": "single_trials",
        "conditions": [
            # "CSpu",
            # "CSpr",
            # "CSm",
            # "USo",
            # "USm",
            "USp"
        ],
        "hrf": "dgamma",
        "mode": "cond"
    }

    feature = SimpleNamespace(**ddict)
    single_trials_wf = single_trials.init_singletrials_wf(
        condition_files=tsv_files[0],
        condition_units="secs",
        feature=feature,
        workdir="/tmp/test"
    )

    single_trials_wf.base_dir = "/tmp/test"

    # single_trials_wf.base_dir = workdir  # ensure outputs stay in main workdir
    inputnode = single_trials_wf.get_node("inputnode").inputs
    inputnode.bold = source_file
    inputnode.repetition_time = entities["RepetitionTime"]
    inputnode.tags = entities
    inputnode.mask = bold_masks[0]
    inputnode.confounds_selected = in_tsv
    inputnode.vals = {"scan_start": 0.0}

    from nipype import config
    config.update_config({
        'execution': {
            'remove_unnecessary_outputs': False,
            'keep_inputs': True,  # Optional: keep input copies too
        }
    })

    single_trials_wf.write_graph(
        graph2use="colored",
        dotfilename=opj(single_trials_wf.base_dir, "graph.dot")
    )
    
    graph = single_trials_wf.run(plugin='MultiProc', plugin_args={'n_procs': 4})
