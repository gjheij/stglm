# -*- coding: utf-8 -*-
"""Run-level QC helpers used by the single-trial workflow."""

from __future__ import annotations

from pathlib import Path


def update_vals_with_fd(vals, confounds_file, fd_thres=0.2):
    """Add HALFpipe-compatible FD summaries from an fMRIPrep confounds TSV."""
    import numpy as np
    import pandas as pd

    out = dict(vals or {})
    data_frame = pd.read_csv(
        confounds_file,
        sep="\t",
        index_col=None,
        dtype=np.float64,
        na_filter=True,
        na_values="n/a",
    )
    if "framewise_displacement" not in data_frame.columns:
        raise KeyError(
            f"Confounds file {confounds_file!r} has no framewise_displacement column"
        )

    fd = data_frame["framewise_displacement"].dropna()
    if fd.empty:
        raise ValueError(f"No valid framewise_displacement values in {confounds_file!r}")

    out["fd_mean"] = float(fd.mean())
    out["fd_max"] = float(fd.max())
    out["fd_perc"] = float((fd > float(fd_thres)).mean() * 100.0)
    return out


def update_vals_with_mean_gm_tsnr(
    vals,
    tsnr_file,
    gm_probseg_file,
    mask_file=None,
    gm_threshold=0.5,
):
    """Average a computed tSNR image in fMRIPrep gray matter.

    The GM probability map is resampled to the tSNR/BOLD grid with linear
    interpolation. Voxels with probability >= ``gm_threshold`` are selected,
    optionally intersected with the run mask.
    """
    import nibabel as nib
    import numpy as np
    from nibabel.processing import resample_from_to

    out = dict(vals or {})
    tsnr_img = nib.load(tsnr_file)
    gm_img = nib.load(gm_probseg_file)

    if tsnr_img.ndim != 3:
        raise ValueError(f"Expected a 3D tSNR image, got shape {tsnr_img.shape}")

    target = (tsnr_img.shape, tsnr_img.affine)
    if gm_img.shape != tsnr_img.shape or not np.allclose(
        gm_img.affine, tsnr_img.affine, rtol=1e-5, atol=1e-5
    ):
        gm_img = resample_from_to(gm_img, target, order=1)

    tsnr_data = np.asanyarray(tsnr_img.dataobj, dtype=np.float64)
    gm_data = np.asanyarray(gm_img.dataobj, dtype=np.float64)
    select = np.isfinite(gm_data) & (gm_data >= float(gm_threshold))

    if mask_file not in (None, ""):
        mask_img = nib.load(mask_file)
        if mask_img.shape != tsnr_img.shape or not np.allclose(
            mask_img.affine, tsnr_img.affine, rtol=1e-5, atol=1e-5
        ):
            mask_img = resample_from_to(mask_img, target, order=0)
        select &= np.asanyarray(mask_img.dataobj) > 0

    values = tsnr_data[select & np.isfinite(tsnr_data)]
    if values.size == 0:
        raise ValueError(
            "No finite tSNR voxels remain after applying the GM probability map"
        )

    out["mean_gm_tsnr"] = float(values.mean())
    return out


def collect_base_metadata(
    tags,
    vals,
    metadata,
    Conditions,
    ConditionFile,
    ConditionUnits,
    BasedOn,
    Model,
):
    """Build a flat HALFpipe-like statmap metadata object.

    Upstream metadata and run values are retained, common snake_case aliases are
    normalized to the capitalization used by HALFpipe, and workflow bookkeeping
    fields that are absent from the reference sidecar are omitted.
    """
    canonical_keys = {
        "acquisition_orientation": "AcquisitionOrientation",
        "acquisition_volume_shape": "AcquisitionVolumeShape",
        "acquisition_voxel_size": "AcquisitionVoxelSize",
        "confound_regressors": "ConfoundRegressors",
        "dummy_scans": "DummyScans",
        "fd_max": "FDMax",
        "fd_mean": "FDMean",
        "fd_perc": "FDPerc",
        "fallback_registration": "FallbackRegistration",
        "halfpipe_version": "HALFpipeVersion",
        "mean_gm_tsnr": "MeanGMTSNR",
        "model": "Model",
        "number_of_volumes": "NumberOfVolumes",
        "raw_sources": "RawSources",
        "repetition_time": "RepetitionTime",
        "scan_start": "ScanStart",
        "setting": "Setting",
        "task_name": "TaskName",
    }

    out = {}
    for source in (metadata, vals):
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if value is None:
                continue
            out[canonical_keys.get(key, key)] = value

    if isinstance(Model, dict) and Model:
        existing_model = dict(out.get("Model") or {})
        existing_model.update(Model)
        out["Model"] = existing_model

    # Prefer an existing RawSources list; otherwise compose it from known inputs.
    raw_sources = out.get("RawSources", [])
    if isinstance(raw_sources, str):
        raw_sources = [raw_sources]
    elif not isinstance(raw_sources, list):
        raw_sources = list(raw_sources) if raw_sources else []

    for source in (ConditionFile, BasedOn):
        if source is None:
            continue
        values = source if isinstance(source, (list, tuple)) else [source]
        for value in values:
            if value not in raw_sources:
                raw_sources.append(value)
    if raw_sources:
        out["RawSources"] = raw_sources

    if isinstance(tags, dict):
        task = tags.get("task")
        if task and "TaskName" not in out:
            out["TaskName"] = task
        space = tags.get("space")
        if space:
            setting = dict(out.get("Setting") or {})
            setting.setdefault("Space", space)
            out["Setting"] = setting

    # These describe parsing/model setup rather than the reference statmap metadata.
    out.pop("Conditions", None)
    out.pop("ConditionFile", None)
    out.pop("ConditionUnits", None)
    out.pop("Sources", None)
    out.pop("Tags", None)
    out.pop("events", None)
    out.pop("condition", None)
    out.pop("estimation", None)
    return out
