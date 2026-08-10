from types import SimpleNamespace
from pathlib import Path
import os
import numpy as np

from bids import BIDSLayout
from nipype import config
from nipype.interfaces.base import isdefined

from stglm import confounds, single_trials, utils

opj = os.path.join


def _entity_value(bids_file, key):
    value = bids_file.entities.get(key)
    return None if value is None else str(value)


def _find_matching_file(reference, candidates, label):
    """Find one BIDS file matching the run-level entities of reference."""
    match_keys = (
        "subject",
        "session",
        "task",
        "acquisition",
        "run",
        "echo",
        "space",
        "resolution",
    )

    scored = []
    for candidate in candidates:
        score = 0
        valid = True

        for key in match_keys:
            reference_value = _entity_value(reference, key)
            candidate_value = _entity_value(candidate, key)

            if reference_value is None:
                continue

            if candidate_value == reference_value:
                score += 1
            elif candidate_value is not None:
                valid = False
                break

        if valid:
            scored.append((score, candidate))

    if not scored:
        raise FileNotFoundError(
            f"No matching {label} found for {reference.path}"
        )

    best_score = max(score for score, _ in scored)
    matches = [
        candidate
        for score, candidate in scored
        if score == best_score
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Ambiguous {label} match for {reference.path}: "
            f"{[candidate.path for candidate in matches]}"
        )

    return matches[0]


def test_stglm():
    proj_dir = "/mnt/d/fMRI/SC2F"
    sub_id = "22"
    workdir = Path("/tmp/test")
    workdir.mkdir(parents=True, exist_ok=True)

    layout = BIDSLayout(
        proj_dir,
        derivatives=opj(proj_dir, "derivatives", "fmriprep"),
        validate=False,
    )

    event_files = layout.get(
        subject=sub_id,
        suffix="events",
        extension="tsv",
        return_type="object",
    )

    confound_files = layout.get(
        subject=sub_id,
        suffix="timeseries",
        extension="tsv",
        return_type="object",
    )

    bold_files = layout.get(
        subject=sub_id,
        suffix="bold",
        space="T1w",
        extension="nii.gz",
        return_type="object",
        desc="masked",
    )

    bold_masks = layout.get(
        subject=sub_id,
        suffix="mask",
        space="T1w",
        extension="nii.gz",
        return_type="object",
    )

    # fMRIPrep normally provides a GM probability segmentation, not a tSNR map.
    # The single-trial workflow computes tSNR from the BOLD series itself.
    gm_probsegs = layout.get(
        subject=sub_id,
        suffix="probseg",
        label="GM",
        extension="nii.gz",
        return_type="object",
    )

    native_gm_probsegs = [
        gm_file
        for gm_file in gm_probsegs
        if gm_file.entities.get("space") is None
    ]

    if native_gm_probsegs:
        gm_probsegs = native_gm_probsegs

    assert bold_files, f"No BOLD files found for sub-{sub_id}"
    assert bold_masks, f"No BOLD masks found for sub-{sub_id}"
    assert confound_files, f"No confounds found for sub-{sub_id}"
    assert event_files, f"No events found for sub-{sub_id}"
    assert gm_probsegs, f"No label-GM_probseg found for sub-{sub_id}"

    # Test one run while matching all auxiliary files by BIDS entities.
    bold_file = bold_files[0]
    mask_file = _find_matching_file(bold_file, bold_masks, "BOLD mask")
    confounds_file = _find_matching_file(
        bold_file,
        confound_files,
        "confounds TSV",
    )
    events_file = _find_matching_file(
        bold_file,
        event_files,
        "events TSV",
    )
    gm_probseg_file = _find_matching_file(
        bold_file,
        gm_probsegs,
        "GM probability segmentation",
    )

    entities = dict(bold_file.entities)
    source_file = bold_file.path

    repetition_time = entities.get("RepetitionTime")
    metadata = layout.get_metadata(source_file)
    if repetition_time is None:
        repetition_time = metadata.get("RepetitionTime")

    assert repetition_time is not None, (
        f"No RepetitionTime found for {source_file}"
    )

    confound_options = {
        "ICA-AROMA": "ICA-AROMA",
        "Motion parameters": "(trans|rot)_[xyz]",
        "Derivatives of motion parameters":
            "(trans|rot)_[xyz]_derivative1",
        "Motion parameters squared": "(trans|rot)_[xyz]_power2",
        "Derivatives of motion parameters squared":
            "(trans|rot)_[xyz]_derivative1_power2",
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
        "aCompCor (top five components)",
    ]
    confound_regexes = [
        confound_options[name]
        for name in search_list
    ]

    # Select the nuisance-regressor subset used by the GLM.
    confounds_wf = confounds.init_confounds_select_wf()
    confounds_wf.base_dir = str(workdir)

    conf_input = confounds_wf.get_node("inputnode").inputs
    conf_input.bold = source_file
    conf_input.tags = entities
    conf_input.mask = mask_file.path
    conf_input.confounds = confounds_file.path
    conf_input.confound_names = confound_regexes

    confound_result = confounds_wf.run()

    selected_confounds_tsv = utils.get_node_by_name(
        confound_result,
        "selectcolumns",
    ).result.outputs.out_with_header

    feature = SimpleNamespace(
        name="SingleTrials",
        setting="SingleTrialsSetting",
        type="single_trials",
        conditions=["USp"],
        hrf="dgamma",
        mode="cond",
    )

    single_trials_wf = single_trials.init_singletrials_wf(
        condition_files=events_file.path,
        condition_units="secs",
        feature=feature,
        workdir=workdir,

        # Selected columns are used as model nuisance regressors.
        confound_file=selected_confounds_tsv,

        # The full fMRIPrep TSV is used to calculate FDMean, FDMax, FDPerc.
        confounds_file=confounds_file.path,

        # tSNR is computed internally from BOLD and averaged in this GM map.
        gm_probseg=gm_probseg_file.path,

        fd_threshold=0.2,
        tsnr_regress_poly=2,
        gm_threshold=0.5,
    )
    single_trials_wf.base_dir = str(workdir)

    st_input = single_trials_wf.get_node("inputnode").inputs
    st_input.bold = source_file
    st_input.repetition_time = float(repetition_time)
    st_input.tags = entities
    st_input.mask = mask_file.path
    st_input.vals = {"scan_start": 0.0}
    st_input.metadata = metadata

    config.update_config({
        "execution": {
            "remove_unnecessary_outputs": False,
            "keep_inputs": True,
        }
    })

    single_trials_wf.write_graph(
        graph2use="colored",
        dotfilename=opj(single_trials_wf.base_dir, "graph.dot"),
    )

    graph = single_trials_wf.run(
        plugin="MultiProc",
        plugin_args={"n_procs": 4},
    )

    # Verify that the newly added QC nodes executed and produced values.
    fd_result = utils.get_node_by_name(
        graph,
        "update_vals_with_fd",
    ).result.outputs.vals

    gm_tsnr_result = utils.get_node_by_name(
        graph,
        "update_vals_with_mean_gm_tsnr",
    ).result.outputs.vals

    assert "fd_mean" in fd_result
    assert "fd_max" in fd_result
    assert "fd_perc" in fd_result

    assert fd_result["fd_mean"] >= 0
    assert fd_result["fd_max"] >= fd_result["fd_mean"]
    assert 0 <= fd_result["fd_perc"] <= 100

    # Confirm the output sidecars retain HALFpipe-style Setting metadata and
    # add a separate statistical Model provenance block.
    lsa_metadata_files = utils.get_node_by_name(
        graph,
        "rename_lsa_metadata",
    ).result.outputs.out_file

    if isinstance(lsa_metadata_files, str):
        lsa_metadata_files = [lsa_metadata_files]

    import json
    with open(lsa_metadata_files[0], "r", encoding="utf-8") as f:
        sidecar = json.load(f)

    assert "Model" in sidecar
    assert sidecar["Model"]["Estimator"] == "LSA"
    assert sidecar["Model"]["Type"] == "single_trials"
    assert sidecar["Model"]["HRF"] == "dgamma"
    assert sidecar["Model"]["Mode"] == "cond"
    assert sidecar["Model"]["Conditions"] == ["USp"]

    if "Setting" in metadata:
        assert sidecar["Setting"] == metadata["Setting"]

    tsnr_result = utils.get_node_by_name(
        graph,
        "tsnr",
    ).result.outputs

    assert isdefined(tsnr_result.tsnr_file)
    assert os.path.isfile(tsnr_result.tsnr_file)

    assert "mean_gm_tsnr" in gm_tsnr_result
    assert np.isfinite(gm_tsnr_result["mean_gm_tsnr"])
    assert gm_tsnr_result["mean_gm_tsnr"] > 0
