# -*- coding: utf-8 -*-
"""Single-trial task workflow with HALFpipe-style condition-wise beta time series outputs."""

from __future__ import annotations

import json
import os
import re
from math import isfinite
from pathlib import Path

import nipype.algorithms.modelgen as model
from nipype.algorithms.confounds import TSNR
import nipype.interfaces.utility as niu
import numpy as np
from nipype.interfaces.base import isdefined
from nipype.interfaces.fsl import FEATModel, FILMGLS, Merge
from nipype.interfaces.fsl import model as fslm
from nipype.interfaces.io import DataSink
from nipype.pipeline import engine as pe

from .conditions import ApplyConditionOffset, ParseConditionFile
from .tsv import FillNA
from .qc import update_vals_with_fd, update_vals_with_mean_gm_tsnr, collect_base_metadata


def _get_scan_start(vals) -> float:
    return float(vals["scan_start"])


def update_vals_with_number_of_volumes(vals, bold_file):
    """Populate NumberOfVolumes from the BOLD NIfTI time dimension."""
    import nibabel as nib

    vals = dict(vals or {})
    img = nib.load(bold_file)

    if len(img.shape) < 4:
        raise ValueError(
            f"Expected a 4D BOLD image, got shape {img.shape}: {bold_file}"
        )

    vals["number_of_volumes"] = int(img.shape[3])
    return vals


def _as_bunch(subject_info):
    """Accept either a Bunch or a singleton list/tuple containing a Bunch."""
    return subject_info[0] if isinstance(subject_info, (list, tuple)) else subject_info


def _trial_digits(n_trials: int) -> int:
    return max(3, len(str(n_trials)))


def _split_trial_condition(condition: str) -> str:
    """Return the original condition name from a trial-wise regressor name."""
    if "__" in condition:
        condition, _ = condition.rsplit("__", 1)
        return condition
    condition, _ = condition.rsplit("_", 1)
    return condition


def _safe_entity(value) -> str:
    """Make a value safe enough for a BIDS-like filename entity."""
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9+.-]+", "", value)
    return value or "unknown"


def _base_bids_name(tags: dict) -> str:
    parts: list[str] = []
    for key in ["subject", "session", "task", "run", "space"]:
        if key in tags and tags[key] not in (None, ""):
            entity = key[:3] if key in {"subject", "session"} else key
            parts.append(f"{entity}-{_safe_entity(tags[key])}")
    return "_".join(parts)


def _events_from_bunch(subject_info):
    cb = _as_bunch(subject_info)
    events: dict[str, list[dict[str, float]]] = {}
    has_amp = getattr(cb, "amplitudes", None) is not None

    for ci, condition in enumerate(cb.conditions):
        condition_events = []
        for ti, onset in enumerate(cb.onsets[ci]):
            event = {
                "onset": float(onset),
                "duration": float(cb.durations[ci][ti]),
            }
            if has_amp:
                event["amplitude"] = float(cb.amplitudes[ci][ti])
            condition_events.append(event)
        events[condition] = condition_events
    return events


def cond_bunch_to_lsa(subject_info):
    """Convert a normal condition Bunch to least-squares-all trial regressors.

    Nipype's Function interface serializes only this function body, so helper
    logic must live inside the function rather than referring to module globals.
    """
    from nipype.interfaces.base import Bunch

    cb = subject_info[0] if isinstance(subject_info, (list, tuple)) else subject_info
    conditions, onsets, durations, amps = [], [], [], []
    has_amp = getattr(cb, "amplitudes", None) is not None

    for ci, cname in enumerate(cb.conditions):
        digits = max(3, len(str(len(cb.onsets[ci]))))
        for ti, onset in enumerate(cb.onsets[ci]):
            conditions.append(f"{cname}__{ti + 1:0{digits}d}")
            onsets.append([onset])
            durations.append([cb.durations[ci][ti]])
            amps.append([cb.amplitudes[ci][ti] if has_amp else 1.0])

    return [
        Bunch(
            conditions=conditions,
            onsets=onsets,
            durations=durations,
            amplitudes=amps,
            tmod=None,
            pmod=None,
            regressor_names=None,
            regressors=None,
        )
    ]


def cond_bunch_to_lss(subject_info):
    """Convert a normal condition Bunch to HALFpipe-style LSS models.

    Each generated model has:
      [target_trial, same_condition_others, *other_original_conditions]

    Nipype's Function interface serializes only this function body, so helper
    logic must live inside the function rather than referring to module globals.
    """
    from nipype.interfaces.base import Bunch

    cb = subject_info[0] if isinstance(subject_info, (list, tuple)) else subject_info
    has_amp = getattr(cb, "amplitudes", None) is not None
    subject_infos = []

    for condition_index, (condition, condition_onsets, condition_durations) in enumerate(
        zip(cb.conditions, cb.onsets, cb.durations, strict=True)
    ):
        digits = max(3, len(str(len(condition_onsets))))

        other_conditions = list(cb.conditions)
        other_conditions.pop(condition_index)
        other_onsets_lists = list(cb.onsets)
        other_onsets_lists.pop(condition_index)
        other_duration_lists = list(cb.durations)
        other_duration_lists.pop(condition_index)
        if has_amp:
            other_amplitude_lists = list(cb.amplitudes)
            other_amplitude_lists.pop(condition_index)

        for trial_index, (onset, duration) in enumerate(
            zip(condition_onsets, condition_durations, strict=True)
        ):
            other_onsets = list(condition_onsets)
            other_onsets.pop(trial_index)
            other_durations = list(condition_durations)
            other_durations.pop(trial_index)

            conditions = [f"{condition}__{trial_index + 1:0{digits}d}", f"{condition}_others", *other_conditions]
            onsets = [[onset], other_onsets, *other_onsets_lists]
            durations = [[duration], other_durations, *other_duration_lists]

            kwargs = {}
            if has_amp:
                other_amplitudes = list(cb.amplitudes[condition_index])
                target_amplitude = other_amplitudes.pop(trial_index)
                kwargs["amplitudes"] = [[target_amplitude], other_amplitudes, *other_amplitude_lists]

            subject_infos.append(
                Bunch(
                    conditions=conditions,
                    onsets=onsets,
                    durations=durations,
                    tmod=None,
                    pmod=None,
                    regressor_names=None,
                    regressors=None,
                    **kwargs,
                )
            )

    return subject_infos


def group_lsa_parameter_estimates(subject_infos, param_estimates):
    """Group LSA param-estimate files into one list per original condition."""
    from collections import defaultdict

    subject_info = subject_infos[0] if isinstance(subject_infos, (list, tuple)) else subject_infos
    if isinstance(param_estimates, (list, tuple)) and len(param_estimates) == 1 and isinstance(param_estimates[0], (list, tuple)):
        param_estimates = param_estimates[0]

    grouped = defaultdict(list)
    events = defaultdict(list)

    for condition, onset, duration, pe_file in zip(
        subject_info.conditions,
        subject_info.onsets,
        subject_info.durations,
        param_estimates,
        strict=False,
    ):
        original_condition = condition.rsplit("__", 1)[0] if "__" in condition else condition.rsplit("_", 1)[0]
        grouped[original_condition].append(pe_file)
        events[original_condition].append(
            {"onset": float(onset[0]), "duration": float(duration[0])}
        )

    if grouped.keys() != events.keys():
        raise ValueError("Mismatch between LSA parameter estimates and events")

    return list(grouped.keys()), list(grouped.values()), list(events.values())


def group_lss_parameter_estimates(subject_infos, param_estimates):
    """Group the first beta from each LSS model into one list per original condition."""
    from collections import defaultdict

    grouped = defaultdict(list)
    events = defaultdict(list)

    for subject_info, pe_files in zip(subject_infos, param_estimates, strict=True):
        condition = subject_info.conditions[0]
        original_condition = condition.rsplit("__", 1)[0] if "__" in condition else condition.rsplit("_", 1)[0]
        grouped[original_condition].append(pe_files[0])
        events[original_condition].append(
            {
                "onset": float(subject_info.onsets[0][0]),
                "duration": float(subject_info.durations[0][0]),
            }
        )

    if grouped.keys() != events.keys():
        raise ValueError("Mismatch between LSS parameter estimates and events")

    return list(grouped.keys()), list(grouped.values()), list(events.values())


def format_condition_bids_filenames(
    tags: dict,
    conditions: list[str],
    estimator: str,
):
    import re

    def safe_entity(value):
        value = str(value)
        value = re.sub(r"[^A-Za-z0-9+.-]+", "", value)
        return value or "unknown"

    parts = []
    for key in ["subject", "session", "task", "run"]:
        if key in tags and tags[key] not in (None, ""):
            entity = key[:3] if key in {"subject", "session"} else key
            parts.append(f"{entity}-{safe_entity(tags[key])}")

    base = "_".join(parts)

    statmaps = []
    metadata = []
    masks = []

    for condition in conditions:
        cond = safe_entity(condition)
        stem = f"{base}_feature-{estimator}_condition-{cond}"
        statmaps.append(f"{stem}_stat-effect_statmap.nii.gz")
        metadata.append(f"{stem}_stat-effect_statmap.json")
        masks.append(f"{stem}_mask.nii.gz")

    return statmaps, metadata, masks


def repeat_file_to_match(in_file, items):
    return [in_file for _ in items]


def format_tsnr_filename(tags: dict):
    """Return a BIDS-like filename for the run-level tSNR map."""
    import re

    def safe_entity(value):
        value = str(value)
        value = re.sub(r"[^A-Za-z0-9+.-]+", "", value)
        return value or "unknown"

    parts = []
    for key in ["subject", "session", "task", "run"]:
        if key in tags and tags[key] not in (None, ""):
            entity = key[:3] if key in {"subject", "session"} else key
            parts.append(f"{entity}-{safe_entity(tags[key])}")

    base = "_".join(parts)
    return f"{base}_stat-tsnr_boldmap.nii.gz"

def format_design_filenames(tags: dict):
    import re

    def safe_entity(value):
        value = str(value)
        value = re.sub(r"[^A-Za-z0-9+.-]+", "", value)
        return value or "unknown"

    parts = []
    for key in ["subject", "session", "task", "run"]:
        if key in tags and tags[key] not in (None, ""):
            entity = key[:3] if key in {"subject", "session"} else key
            parts.append(f"{entity}-{safe_entity(tags[key])}")
    base = "_".join(parts)

    return [
        f"{base}_feature-LSA_desc-design_matrix.png",
        f"{base}_feature-LSA_desc-design_matrix.mat",
        f"{base}_feature-LSS_desc-design_matrix.png",
        f"{base}_feature-LSS_desc-design_matrix.mat",
    ]


def write_condition_metadata_jsons(
    metadata: dict,
    conditions: list[str],
    events: list[list[dict]],
    estimator: str,
):
    """Write condition sidecars in the flat HALFpipe statmap layout.

    The condition and estimator are already represented in the BIDS-like filename,
    so they are intentionally not duplicated as non-reference JSON fields.
    """
    import json
    import re
    from pathlib import Path

    def safe_entity(value):
        value = str(value)
        value = re.sub(r"[^A-Za-z0-9+.-]+", "", value)
        return value or "unknown"

    out_files = []
    cwd = Path.cwd()

    for condition, condition_events in zip(conditions, events, strict=True):
        out_metadata = dict(metadata or {})

        # Keep analysis provenance separate from HALFpipe-style preprocessing
        # settings. The branch-specific estimator belongs in each sidecar.
        model_metadata = dict(out_metadata.get("Model") or {})
        model_metadata["Estimator"] = estimator
        model_metadata.setdefault("Condition", condition)
        out_metadata["Model"] = model_metadata

        # Match the reference sidecar's capitalization and event representation.
        out_metadata["Events"] = [
            {
                "duration": float(event["duration"]),
                "onset": float(event["onset"]),
                **(
                    {"amplitude": float(event["amplitude"])}
                    if "amplitude" in event
                    else {}
                ),
            }
            for event in condition_events
        ]

        out_file = cwd / f"condition-{safe_entity(condition)}_desc-{estimator}_betas.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out_metadata, f, indent=4, ensure_ascii=False)
        out_files.append(str(out_file))

    return out_files


def format_condition_metadata_filenames(
        tags: dict,
        conditions: list[str],
        estimator: str
    ):
    import re

    def safe_entity(value):
        value = str(value)
        value = re.sub(r"[^A-Za-z0-9+.-]+", "", value)
        return value or "unknown"

    parts = []
    for key in ["subject", "session", "task", "run"]:
        if key in tags and tags[key] not in (None, ""):
            entity = key[:3] if key in {"subject", "session"} else key
            parts.append(f"{entity}-{safe_entity(tags[key])}")
    base = "_".join(parts)

    return [
        f"{base}_condition-{safe_entity(condition)}_desc-{estimator}_betas.json"
        for condition in conditions
    ]



def feature_to_model_metadata(feature):
    """Create a JSON-safe statistical-model provenance block.

    ``Setting`` is reserved for HALFpipe-style preprocessing settings. This
    block records the single-trial model definition independently.
    """
    from pathlib import Path

    def json_safe(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {
                str(key): json_safe(item)
                for key, item in value.items()
                if not str(key).startswith("_")
            }
        if isinstance(value, (list, tuple, set)):
            return [json_safe(item) for item in value]
        return str(value)

    source = vars(feature) if hasattr(feature, "__dict__") else dict(feature or {})
    fields = {
        key: json_safe(value)
        for key, value in source.items()
        if not key.startswith("_") and not callable(value)
    }

    preferred_names = {
        "name": "Name",
        "setting": "SettingName",
        "type": "Type",
        "conditions": "Conditions",
        "hrf": "HRF",
        "mode": "Mode",
        "confounds": "Confounds",
        "confounds_regex": "ConfoundRegex",
        "high_pass_filter_cutoff": "HighPassFilterCutoff",
    }
    return {
        preferred_names.get(key, key): value
        for key, value in fields.items()
    }




def select_first(in_files):
    return in_files[0] if in_files else None


def generate_sink_paths(tags):
    sub_id = tags.get("subject")
    ses_id = tags.get("session")
    task_id = tags.get("task")

    if not sub_id:
        raise ValueError("Missing subject ID in tags")

    container = f"sub-{sub_id}"
    if ses_id:
        container += f"/ses-{ses_id}"

    return container, task_id

def generate_datasink_substitutions(task_id):
    if task_id:
        return [
            ("func/task", f"func/task-{task_id}"),
        ]

    # remove empty task directory
    return [
        ("func/task", "func"),
    ]


def init_singletrials_wf(
    workdir: Path | str,
    feature,
    condition_files,
    condition_units,
    confound_file=None,
    confounds_file=None,
    gm_probseg=None,
    fd_threshold=0.2,
    tsnr_regress_poly=2,
    gm_threshold=0.5,
    derivatives=None,
):
    """
    Create a single-trial workflow with HALFpipe-style condition-wise beta time series outputs.

    Outputs are harmonized with HALFpipe's task_based.py single-trial logic:
    parameter estimates are grouped by original condition, merged along the
    fourth dimension per condition, and paired with per-condition event metadata.
    """
    workflow = pe.Workflow(name="singletrials_wf")

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "tags",
                "vals",
                "metadata",
                "bold",
                "mask",
                "repetition_time",
                "confounds_selected",
                "confounds_file",
                "gm_probseg",
                "condition_names",
                "condition_files",
                "condition_units",
            ]
        ),
        name="inputnode",
    )
    inputnode.inputs.condition_names = feature.conditions
    inputnode.inputs.condition_files = condition_files
    inputnode.inputs.condition_units = condition_units
    inputnode.inputs.confounds_selected = confound_file
    inputnode.inputs.confounds_file = confounds_file
    inputnode.inputs.gm_probseg = gm_probseg

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "lsa_betas",
                "lss_betas",
                "lsa_conditions",
                "lss_conditions",
                "lsa_events",
                "lss_events",
                "lsa_metadata",
                "lss_metadata",
                "lsa_masks",
                "lss_masks",
                "tsnr_map",
            ]
        ),
        name="outputnode",
    )

    parse_condition_file = pe.Node(ParseConditionFile(), name="parse_condition_file")
    get_scan_start = pe.Node(
        niu.Function(input_names=["vals"], output_names=["scan_start"], function=_get_scan_start),
        name="get_scan_start",
    )
    apply_condition_offset = pe.Node(ApplyConditionOffset(), name="apply_condition_offset")

    if feature.hrf == "dgamma":
        bases: dict[str, dict] = dict(dgamma=dict())
    elif feature.hrf == "dgamma_with_derivs":
        bases = dict(dgamma=dict(derivs=True))
    elif feature.hrf == "flobs":
        flobs_path = Path(os.environ["FSLDIR"]) / "etc" / "default_flobs.flobs" / "hrfbasisfns.txt"
        if not flobs_path.is_file():
            raise FileNotFoundError(f"FLOBS basis function file {flobs_path} does not exist")
        bases = dict(custom=dict(bfcustompath=str(flobs_path), basisfnum=3))
    else:
        raise ValueError(f'HRF "{feature.hrf}" is not yet implemented')

    fillna = pe.Node(FillNA(), name="fillna")

    update_fd_vals = pe.Node(
        niu.Function(
            input_names=["vals", "confounds_file", "fd_thres"],
            output_names=["vals"],
            function=update_vals_with_fd,
        ),
        name="update_vals_with_fd",
    )
    update_fd_vals.inputs.fd_thres = float(fd_threshold)

    tsnr = pe.Node(
        TSNR(regress_poly=int(tsnr_regress_poly)),
        name="tsnr",
    )
    tsnr.inputs.tsnr_file = "tsnr.nii.gz"
    tsnr.inputs.mean_file = "mean.nii.gz"
    tsnr.inputs.stddev_file = "stdev.nii.gz"
    tsnr.inputs.detrended_file = "detrended.nii.gz"

    format_tsnr_name = pe.Node(
        niu.Function(
            input_names=["tags"],
            output_names=["out_name"],
            function=format_tsnr_filename,
        ),
        name="format_tsnr_name",
    )
    rename_tsnr = pe.Node(
        niu.Rename(keep_ext=False),
        name="rename_tsnr",
    )

    update_gm_tsnr_vals = pe.Node(
        niu.Function(
            input_names=[
                "vals",
                "tsnr_file",
                "gm_probseg_file",
                "mask_file",
                "gm_threshold",
            ],
            output_names=["vals"],
            function=update_vals_with_mean_gm_tsnr,
        ),
        name="update_vals_with_mean_gm_tsnr",
    )
    update_gm_tsnr_vals.inputs.gm_threshold = float(gm_threshold)

    base_metadata = pe.Node(
        niu.Function(
            input_names=[
                "tags",
                "vals",
                "metadata",
                "Conditions",
                "ConditionFile",
                "ConditionUnits",
                "BasedOn",
                "Model",
            ],
            output_names=["metadata"],
            function=collect_base_metadata,
        ),
        name="base_metadata",
    )
    base_metadata.inputs.Model = feature_to_model_metadata(feature)

    update_nvols_vals = pe.Node(
        niu.Function(
            input_names=["vals", "bold_file"],
            output_names=["vals"],
            function=update_vals_with_number_of_volumes,
        ),
        name="update_vals_with_number_of_volumes",
    )

    # ------------------------------------------------------------------
    # LSA branch
    cond2lsa = pe.Node(
        niu.Function(
            input_names=["subject_info"],
            output_names=["lsa_subject_info"],
            function=cond_bunch_to_lsa,
        ),
        name="cond2lsa",
    )

    spec_lsa = pe.Node(model.SpecifyModel(), name="specify_lsa")
    spec_lsa.inputs.high_pass_filter_cutoff = np.inf
    if hasattr(feature, "high_pass_filter_cutoff"):
        hpfc = feature.high_pass_filter_cutoff
        if isinstance(hpfc, float) and isfinite(hpfc):
            spec_lsa.inputs.high_pass_filter_cutoff = hpfc

    l1_lsa = pe.Node(
        fslm.Level1Design(model_serial_correlations=False, bases=bases),
        name="l1design_lsa",
    )
    feat_lsa = pe.Node(FEATModel(), name="featmodel_lsa")
    film_lsa = pe.Node(FILMGLS(autocorr_noestimate=True), name="filmgls_lsa")

    group_lsa = pe.Node(
        niu.Function(
            input_names=["subject_infos", "param_estimates"],
            output_names=["conditions", "param_estimates", "events"],
            function=group_lsa_parameter_estimates,
        ),
        name="group_lsa_parameter_estimates",
    )
    merge_lsa = pe.MapNode(
        Merge(dimension="t", output_type="NIFTI_GZ", merged_file="merged_lsa_betas.nii.gz"),
        iterfield=["in_files"],
        name="merge_lsa_by_condition",
    )

    # ------------------------------------------------------------------
    # LSS branch
    cond2lss = pe.Node(
        niu.Function(
            input_names=["subject_info"],
            output_names=["lss_subject_infos"],
            function=cond_bunch_to_lss,
        ),
        name="cond2lss",
    )
    spec_lss = pe.MapNode(model.SpecifyModel(), iterfield=["subject_info"], name="specify_lss")
    spec_lss.inputs.high_pass_filter_cutoff = np.inf
    if hasattr(feature, "high_pass_filter_cutoff"):
        hpfc = feature.high_pass_filter_cutoff
        if isinstance(hpfc, float) and isfinite(hpfc):
            spec_lss.inputs.high_pass_filter_cutoff = hpfc

    l1_lss = pe.MapNode(
        fslm.Level1Design(model_serial_correlations=False, bases=bases),
        iterfield=["session_info"],
        name="l1design_lss",
    )
    feat_lss = pe.MapNode(FEATModel(), iterfield=["fsf_file", "ev_files"], name="featmodel_lss")
    film_lss = pe.MapNode(
        FILMGLS(autocorr_noestimate=True),
        iterfield=["design_file"],
        name="filmgls_lss",
    )

    group_lss = pe.Node(
        niu.Function(
            input_names=["subject_infos", "param_estimates"],
            output_names=["conditions", "param_estimates", "events"],
            function=group_lss_parameter_estimates,
        ),
        name="group_lss_parameter_estimates",
    )
    merge_lss = pe.MapNode(
        Merge(dimension="t", output_type="NIFTI_GZ", merged_file="merged_lss_betas.nii.gz"),
        iterfield=["in_files"],
        name="merge_lss_by_condition",
    )

    # ------------------------------------------------------------------
    # Naming, metadata, and sinking
    format_lsa_bids_names = pe.Node(
        niu.Function(
            input_names=["tags", "conditions", "estimator"],
            output_names=["statmap_names", "metadata_names", "mask_names"],
            function=format_condition_bids_filenames,
        ),
        name="format_lsa_bids_names",
    )
    format_lsa_bids_names.inputs.estimator = "LSA"

    format_lss_bids_names = pe.Node(
        niu.Function(
            input_names=["tags", "conditions", "estimator"],
            output_names=["statmap_names", "metadata_names", "mask_names"],
            function=format_condition_bids_filenames,
        ),
        name="format_lss_bids_names",
    )
    format_lss_bids_names.inputs.estimator = "LSS"

    rename_lsa = pe.MapNode(
        niu.Rename(keep_ext=False),
        iterfield=["in_file", "format_string"],
        name="rename_lsa_outputs",
    )
    rename_lss = pe.MapNode(
        niu.Rename(keep_ext=False),
        iterfield=["in_file", "format_string"],
        name="rename_lss_outputs",
    )

    repeat_lsa_mask = pe.Node(
        niu.Function(
            input_names=["in_file", "items"],
            output_names=["out_files"],
            function=repeat_file_to_match,
        ),
        name="repeat_lsa_mask",
    )
    repeat_lss_mask = pe.Node(
        niu.Function(
            input_names=["in_file", "items"],
            output_names=["out_files"],
            function=repeat_file_to_match,
        ),
        name="repeat_lss_mask",
    )
    rename_lsa_masks = pe.MapNode(
        niu.Rename(keep_ext=False),
        iterfield=["in_file", "format_string"],
        name="rename_lsa_masks",
    )
    rename_lss_masks = pe.MapNode(
        niu.Rename(keep_ext=False),
        iterfield=["in_file", "format_string"],
        name="rename_lss_masks",
    )

    write_lsa_metadata = pe.Node(
        niu.Function(
            input_names=["metadata", "conditions", "events", "estimator"],
            output_names=["out_files"],
            function=write_condition_metadata_jsons,
        ),
        name="write_lsa_metadata",
    )
    write_lsa_metadata.inputs.estimator = "LSA"

    write_lss_metadata = pe.Node(
        niu.Function(
            input_names=["metadata", "conditions", "events", "estimator"],
            output_names=["out_files"],
            function=write_condition_metadata_jsons,
        ),
        name="write_lss_metadata",
    )
    write_lss_metadata.inputs.estimator = "LSS"

    format_lsa_metadata_names = pe.Node(
        niu.Function(
            input_names=["tags", "conditions", "estimator"],
            output_names=["out_names"],
            function=format_condition_metadata_filenames,
        ),
        name="format_lsa_metadata_names",
    )
    format_lsa_metadata_names.inputs.estimator = "LSA"

    format_lss_metadata_names = pe.Node(
        niu.Function(
            input_names=["tags", "conditions", "estimator"],
            output_names=["out_names"],
            function=format_condition_metadata_filenames,
        ),
        name="format_lss_metadata_names",
    )
    format_lss_metadata_names.inputs.estimator = "LSS"

    rename_lsa_metadata = pe.MapNode(
        niu.Rename(keep_ext=False),
        iterfield=["in_file", "format_string"],
        name="rename_lsa_metadata",
    )
    rename_lss_metadata = pe.MapNode(
        niu.Rename(keep_ext=False),
        iterfield=["in_file", "format_string"],
        name="rename_lss_metadata",
    )

    format_design_names = pe.Node(
        niu.Function(input_names=["tags"], output_names=["out_names"], function=format_design_filenames),
        name="format_design_names",
    )
    select_first_lss_png = pe.Node(
        niu.Function(input_names=["in_files"], output_names=["out_file"], function=select_first),
        name="select_first_lss_png",
    )
    select_first_lss_mat = pe.Node(
        niu.Function(input_names=["in_files"], output_names=["out_file"], function=select_first),
        name="select_first_lss_mat",
    )
    collect_design_files = pe.Node(niu.Merge(4), name="collect_design_files")
    rename_designs = pe.MapNode(
        niu.Rename(keep_ext=False),
        iterfield=["in_file", "format_string"],
        name="rename_designs",
    )

    if derivatives is None:
        derivatives = os.path.join(workdir, "derivatives", "stglm")

    datasink = pe.Node(
        DataSink(parameterization=False, base_directory=str(derivatives)),
        name="stglm_datasink",
    )
    get_sink_paths = pe.Node(
        niu.Function(
            input_names=["tags"],
            output_names=["container", "task_id"],
            function=generate_sink_paths
        ),
        name="get_sink_paths",
    )

    get_datasink_substitutions = pe.Node(
        niu.Function(
            input_names=["task_id"],
            output_names=["substitutions"],
            function=generate_datasink_substitutions,
        ),
        name="get_datasink_substitutions",
    )

    if isdefined(inputnode.inputs.confounds_selected):
        workflow.connect([
            (inputnode, fillna, [("confounds_selected", "in_tsv")]),
            (fillna, spec_lsa, [("out_no_header", "realignment_parameters")]),
            (fillna, spec_lss, [("out_no_header", "realignment_parameters")]),
        ])

    # Build the HALFpipe-like vals chain. FD uses the original fMRIPrep
    # confounds TSV. tSNR is computed from the same BOLD series used by the GLM.
    vals_source = inputnode
    vals_output = "vals"
    if confounds_file is not None:
        workflow.connect(inputnode, "vals", update_fd_vals, "vals")
        workflow.connect(inputnode, "confounds_file", update_fd_vals, "confounds_file")
        vals_source = update_fd_vals

    if gm_probseg is not None:
        workflow.connect(inputnode, "bold", tsnr, "in_file")
        workflow.connect(vals_source, vals_output, update_gm_tsnr_vals, "vals")
        workflow.connect(tsnr, "tsnr_file", update_gm_tsnr_vals, "tsnr_file")
        workflow.connect(inputnode, "gm_probseg", update_gm_tsnr_vals, "gm_probseg_file")
        workflow.connect(inputnode, "mask", update_gm_tsnr_vals, "mask_file")
        vals_source = update_gm_tsnr_vals

    workflow.connect(vals_source, vals_output, update_nvols_vals, "vals")
    workflow.connect(inputnode, "bold", update_nvols_vals, "bold_file")

    workflow.connect(update_nvols_vals, "vals", base_metadata, "vals")

    workflow.connect([
        # General parsing and timing
        (inputnode, parse_condition_file, [
            ("condition_names", "condition_names"),
            ("condition_files", "in_any"),
        ]),
        (inputnode, get_scan_start, [("vals", "vals")]),
        (parse_condition_file, apply_condition_offset, [("subject_info", "subject_info")]),
        (get_scan_start, apply_condition_offset, [("scan_start", "scan_start")]),
        (inputnode, base_metadata, [
            ("tags", "tags"),
            ("metadata", "metadata"),
            ("condition_names", "Conditions"),
            ("condition_files", "ConditionFile"),
            ("condition_units", "ConditionUnits"),
            ("bold", "BasedOn"),
        ]),
        (inputnode, get_sink_paths, [("tags", "tags")]),
        (inputnode, format_design_names, [("tags", "tags")]),
        (inputnode, format_tsnr_name, [("tags", "tags")]),
        (tsnr, rename_tsnr, [("tsnr_file", "in_file")]),
        (format_tsnr_name, rename_tsnr, [("out_name", "format_string")]),

        # LSA design/model
        (apply_condition_offset, cond2lsa, [("subject_info", "subject_info")]),
        (inputnode, spec_lsa, [
            ("bold", "functional_runs"),
            ("repetition_time", "time_repetition"),
            ("condition_units", "input_units"),
        ]),
        (cond2lsa, spec_lsa, [("lsa_subject_info", "subject_info")]),
        (spec_lsa, l1_lsa, [("session_info", "session_info")]),
        (inputnode, l1_lsa, [("repetition_time", "interscan_interval")]),
        (l1_lsa, feat_lsa, [("fsf_files", "fsf_file"), ("ev_files", "ev_files")]),
        (feat_lsa, film_lsa, [("design_file", "design_file")]),
        (inputnode, film_lsa, [("bold", "in_file")]),

        # LSA HALFpipe-style grouping/output
        (cond2lsa, group_lsa, [("lsa_subject_info", "subject_infos")]),
        (film_lsa, group_lsa, [("param_estimates", "param_estimates")]),
        (group_lsa, merge_lsa, [("param_estimates", "in_files")]),
        (inputnode, format_lsa_bids_names, [("tags", "tags")]),
        (group_lsa, format_lsa_bids_names, [("conditions", "conditions")]),
        (merge_lsa, rename_lsa, [("merged_file", "in_file")]),
        (format_lsa_bids_names, rename_lsa, [("statmap_names", "format_string")]),
        (base_metadata, write_lsa_metadata, [("metadata", "metadata")]),
        (group_lsa, write_lsa_metadata, [("conditions", "conditions"), ("events", "events")]),
        (write_lsa_metadata, rename_lsa_metadata, [("out_files", "in_file")]),
        (format_lsa_bids_names, rename_lsa_metadata, [("metadata_names", "format_string")]),
        (inputnode, repeat_lsa_mask, [("mask", "in_file")]),
        (group_lsa, repeat_lsa_mask, [("conditions", "items")]),
        (repeat_lsa_mask, rename_lsa_masks, [("out_files", "in_file")]),
        (format_lsa_bids_names, rename_lsa_masks, [("mask_names", "format_string")]),

        # LSS design/model
        (apply_condition_offset, cond2lss, [("subject_info", "subject_info")]),
        (inputnode, spec_lss, [
            ("bold", "functional_runs"),
            ("repetition_time", "time_repetition"),
            ("condition_units", "input_units"),
        ]),
        (cond2lss, spec_lss, [("lss_subject_infos", "subject_info")]),
        (spec_lss, l1_lss, [("session_info", "session_info")]),
        (inputnode, l1_lss, [("repetition_time", "interscan_interval")]),
        (l1_lss, feat_lss, [("fsf_files", "fsf_file"), ("ev_files", "ev_files")]),
        (feat_lss, film_lss, [("design_file", "design_file")]),
        (inputnode, film_lss, [("bold", "in_file")]),

        # LSS HALFpipe-style grouping/output
        (cond2lss, group_lss, [("lss_subject_infos", "subject_infos")]),
        (film_lss, group_lss, [("param_estimates", "param_estimates")]),
        (group_lss, merge_lss, [("param_estimates", "in_files")]),
        (inputnode, format_lss_bids_names, [("tags", "tags")]),
        (group_lss, format_lss_bids_names, [("conditions", "conditions")]),
        (merge_lss, rename_lss, [("merged_file", "in_file")]),
        (format_lss_bids_names, rename_lss, [("statmap_names", "format_string")]),
        (base_metadata, write_lss_metadata, [("metadata", "metadata")]),
        (group_lss, write_lss_metadata, [("conditions", "conditions"), ("events", "events")]),
        (write_lss_metadata, rename_lss_metadata, [("out_files", "in_file")]),
        (format_lss_bids_names, rename_lss_metadata, [("metadata_names", "format_string")]),
        (inputnode, repeat_lss_mask, [("mask", "in_file")]),
        (group_lss, repeat_lss_mask, [("conditions", "items")]),
        (repeat_lss_mask, rename_lss_masks, [("out_files", "in_file")]),
        (format_lss_bids_names, rename_lss_masks, [("mask_names", "format_string")]),

        # Design matrices/images: keep one LSA design and the first LSS design for QC
        (feat_lsa, collect_design_files, [("design_image", "in1"), ("design_file", "in2")]),
        (feat_lss, select_first_lss_png, [("design_image", "in_files")]),
        (feat_lss, select_first_lss_mat, [("design_file", "in_files")]),
        (select_first_lss_png, collect_design_files, [("out_file", "in3")]),
        (select_first_lss_mat, collect_design_files, [("out_file", "in4")]),
        (collect_design_files, rename_designs, [("out", "in_file")]),
        (format_design_names, rename_designs, [("out_names", "format_string")]),

        # Output node
        (rename_lsa, outputnode, [("out_file", "lsa_betas")]),
        (rename_lss, outputnode, [("out_file", "lss_betas")]),
        (rename_lsa_metadata, outputnode, [("out_file", "lsa_metadata")]),
        (rename_lss_metadata, outputnode, [("out_file", "lss_metadata")]),
        (rename_lsa_masks, outputnode, [("out_file", "lsa_masks")]),
        (rename_lss_masks, outputnode, [("out_file", "lss_masks")]),
        (rename_tsnr, outputnode, [("out_file", "tsnr_map")]),
        (group_lsa, outputnode, [("conditions", "lsa_conditions"), ("events", "lsa_events")]),
        (group_lss, outputnode, [("conditions", "lss_conditions"), ("events", "lss_events")]),

        # DataSink
        (get_sink_paths, get_datasink_substitutions, [("task_id", "task_id")]),

        (get_sink_paths, datasink, [("container", "container")]),
        (get_datasink_substitutions, datasink, [("substitutions", "substitutions")]),

        (rename_lsa, datasink, [("out_file", "func.task.@lsa_betas")]),
        (rename_lss, datasink, [("out_file", "func.task.@lss_betas")]),
        (rename_lsa_metadata, datasink, [("out_file", "func.task.@lsa_metadata")]),
        (rename_lss_metadata, datasink, [("out_file", "func.task.@lss_metadata")]),
        (rename_lsa_masks, datasink, [("out_file", "func.task.@lsa_masks")]),
        (rename_lss_masks, datasink, [("out_file", "func.task.@lss_masks")]),
        (rename_designs, datasink, [("out_file", "func.task.@designs")]),

        # Run-level QC belongs in the parent func directory.
        (rename_tsnr, datasink, [("out_file", "func.@tsnr")]),
    ])

    return workflow
