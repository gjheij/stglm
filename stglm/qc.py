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
