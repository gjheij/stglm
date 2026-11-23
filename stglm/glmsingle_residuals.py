"""Recover run-wise TYPE-D residuals from existing GLMsingle fits.

Implementation
--------------
For each voxel, undo PSC conversion and TYPE-D autoscaling in this order::

    beta_input = (beta_saved * abs(meanvol) / 100 - offset) / scale

Convolve each run's saved ``designSINGLE`` with its voxel-selected saved HRF.
Only columns belonging to that run are needed. For reconstructed task prediction
``X beta_input``, export two definitions::

    taskpoly = R_P (Y - X beta_input)
    fullnuisance = R_[P,E,C] (Y - X beta_input)

``P`` spans the exact configured polynomial degrees, ``E`` contains the fitted
additional regressors from DESIGNINFO.params, and ``C`` is the first ``pcnum``
columns of that run's saved ``pcregressors``. Projection uses an SVD basis;
dependent nuisance columns are removed by rank rather than counted twice.
GLMsingle evaluates R2run using taskpoly (polynomials only), even when nuisance
regressors were used to estimate task betas. Validate reconstructed R2run in
input units before publishing any residual NIfTIs for that run.

Output units default to ``beta``: multiply residuals by
``100 * scale / abs(meanvol)``. This is the linear voxel scaling of the exported
TYPE-D betas. Do not add the beta offset to residuals. ``input`` preserves BOLD
units. Full-nuisance residuals do not reproduce GLMsingle's stored R2run and can
still contain underfitted task signal because TYPE-D is ridge-regularized.

There is no HRF selection, ridge optimization or beta refit here. The original
ordered BOLD files and DESIGNINFO must agree with the saved fit. Normal Python
library HRF indices are zero-based; out-of-range indices fail validation rather
than being silently changed. Invalid/background betas, means, scales or R2run
values, or numerically constant detrended data, are excluded and written to a per-run valid-voxel mask (not valid zero
noise). Use mask intersections when comparing runs.

The exact ridge noise degrees of freedom are not recovered. They are generally
voxel dependent and are NOT N minus the unregularized design rank. For a fixed
linear residual operator A, noise-variance degrees of freedom would be
trace(A A.T); adaptive HRF/ridge/PC selection adds further qualification.
JSON sets ResidualDegreesOfFreedom to null and supplies nuisance and full-design
ranks only as diagnostics. Do not pass those ranks as ridge residual DOF to RSA.

Memory is bounded by a voxel block and a few time-by-regressor matrices beyond
the already-loaded model dictionary. Each BOLD run is read once, volume by
volume, into a Fortran-ordered disk-backed array so each volume write is
contiguous. Residual work arrays remain C-ordered for contiguous voxel blocks.
Buffering, reconstruction and image writing have separate timing messages.
Residual work arrays are also
disk-backed, runs are processed sequentially, and NIfTIs are atomically
published only after validation. Temporary data are removed on exit.
"""
import hashlib
from pathlib import Path
import tempfile
from time import monotonic

import nibabel as nib
import numpy as np

from .logger import get_logger

logger = get_logger(__name__)
logger.setLevel('INFO')


def _basis(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError('Nuisance regressors must be a finite time-by-regressors matrix')
    if not matrix.shape[1]:
        return matrix.copy()
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular[0]
    return u[:, singular > tolerance]


def _project(values, basis):
    return values - basis @ (basis.T @ values)


def _polynomials(params, index, count, n_time):
    degrees = params['maxpolydeg']
    if np.ndim(degrees) != 0:
        if len(degrees) != count:
            raise ValueError('maxpolydeg does not match the fitted run count')
        degrees = degrees[index]
    if np.ndim(degrees) == 0:
        if float(degrees) != int(degrees) or int(degrees) < 0:
            raise ValueError('Invalid polynomial degree')
        degrees = np.arange(int(degrees) + 1)
    degrees = np.asarray(degrees)
    if degrees.ndim != 1 or np.any(degrees < 0) or np.any(degrees != degrees.astype(int)):
        raise ValueError('Invalid polynomial degrees')
    time = np.linspace(-1, 1, n_time)
    return np.column_stack([time ** int(degree) for degree in degrees]), degrees.tolist()


def _run_regressors(values, index, count, n_time, label):
    if values is None or values is False:
        return np.empty((n_time, 0))
    if len(values) == 1 and values[0] is False:
        return np.empty((n_time, 0))
    if len(values) != count:
        raise ValueError(label + ' does not match the fitted run count')
    value = values[index]
    if value is None or value is False:
        return np.empty((n_time, 0))
    value = np.asarray(value, dtype=float)
    if not value.size:
        return np.empty((n_time, 0))
    if value.ndim != 2 or value.shape[0] != n_time or not np.isfinite(value).all():
        raise ValueError(label + ': invalid shape or non-finite values')
    return value


def export_type_d_residuals(fit, design_info, bold_files, output_dir,
                            units='beta', voxel_chunk=2048, r2_atol=0.1):
    """Export taskpoly/fullnuisance residuals, masks and validation sidecars.

    Parameters
    ----------
    fit, design_info : dict
        Loaded TYPED and DESIGNINFO dictionaries from the SAME completed fit.
    bold_files : sequence of paths
        Original input BOLD files in their fitted order. The stGLM export-only
        path validates these inputs and saved designs before calling this helper.
    output_dir : path
        Stream directory; outputs are placed in residuals/model-typed.
    units : {'beta', 'input'}
        Saved beta-compatible linear scaling (default), or input BOLD units.
    voxel_chunk : int
        Maximum number of voxels reconstructed together; no full BOLD get_fdata.
    r2_atol : float
        Maximum absolute reconstruction error in R2 percentage points.

    Returns
    -------
    list of pathlib.Path
        Files to register in fit_provenance.json. Does not modify fitted betas.
    """
    from .glmsingle_io import atomic_json, entities, identity

    if units not in ('beta', 'input') or voxel_chunk < 1 or not np.isfinite(r2_atol) or r2_atol <= 0:
        raise ValueError('Invalid residual units, voxel chunk or R2 tolerance')
    params = design_info['params']
    designs = design_info['designSINGLE']
    if len(bold_files) != len(designs) or not designs:
        raise ValueError('BOLD/design run counts differ')
    count = len(designs)
    hrfs = np.asarray(params['hrflibrary'])
    if hrfs.ndim != 2 or not np.isfinite(hrfs).all():
        raise ValueError('Invalid saved HRF library')
    spatial = np.asarray(fit['meanvol']).shape
    n_voxels = int(np.prod(spatial))
    betas = np.asarray(fit['betasmd']).reshape(n_voxels, -1)
    means = np.abs(np.asarray(fit['meanvol']).reshape(-1))
    indices = np.asarray(fit['HRFindex']).reshape(-1)
    if indices.size != n_voxels or not np.isfinite(indices).all() or np.any(indices != indices.astype(int)):
        raise ValueError('Invalid saved HRF indices')
    indices = indices.astype(int)
    if np.any(indices < 0) or np.any(indices >= hrfs.shape[1]):
        raise ValueError('Saved HRF indices are outside the zero-based library')
    scaleoffset = np.asarray(fit.get('scaleoffset', []))
    if scaleoffset.size:
        scales, offsets = scaleoffset.reshape(n_voxels, 2).T
    else:
        scales, offsets = np.ones(n_voxels), np.zeros(n_voxels)
    expected_r2 = np.asarray(fit['R2run']).reshape(n_voxels, count)
    valid_base = (np.isfinite(means) & (means > 0) & np.isfinite(scales)
                  & (scales != 0) & np.isfinite(offsets))
    saved_pcnum = np.asarray(fit.get('pcnum', 0)).item()
    pcnum = int(saved_pcnum)
    if pcnum < 0 or pcnum != saved_pcnum:
        raise ValueError('Invalid saved noise-PC count')
    if pcnum and 'pcregressors' not in fit:
        raise ValueError('Selected noise PCs missing from the saved fit')
    folder = Path(output_dir) / 'residuals/model-typed'
    folder.mkdir(parents=True, exist_ok=True)
    outputs = []
    reference_affine = None
    for run_index, (bold, design) in enumerate(zip(bold_files, designs)):
        logger.info('Reconstructing TYPE-D residuals: run %d/%d; units=%s; voxel chunk=%d; BOLD=%s',
                    run_index+1, count, units, voxel_chunk, bold)
        image = nib.load(str(bold), keep_file_open=True)
        if image.shape[:3] != spatial or len(image.shape) != 4:
            raise ValueError('BOLD grid differs from saved fit: ' + str(bold))
        if reference_affine is not None and not np.allclose(reference_affine, image.affine):
            raise ValueError('Fitted runs must share their spatial grid')
        reference_affine = image.affine
        n_time = image.shape[-1]
        design = np.asarray(design)
        if design.shape != (n_time, betas.shape[1]) or not np.isfinite(design).all():
            raise ValueError('Saved single-trial design differs from BOLD/betas')
        active = np.flatnonzero(np.any(design != 0, axis=0))
        poly, degrees = _polynomials(params, run_index, count, n_time)
        q_poly = _basis(poly)
        extra = _run_regressors(params.get('extra_regressors'), run_index, count, n_time, 'extra_regressors')
        pcs = (_run_regressors(fit['pcregressors'], run_index, count, n_time, 'pcregressors')
               if pcnum else np.empty((n_time, 0)))
        if pcs.shape[1] < pcnum:
            raise ValueError('Saved PCs fewer than selected pcnum')
        nuisance = np.column_stack([poly, extra, pcs[:, :pcnum]])
        q_full = _basis(nuisance)
        tags = entities(bold)
        prefix = '_'.join(f'{key}-{tags[key]}' for key in ('sub','ses','task','acq','run','echo','space') if tags.get(key))
        if not prefix:
            prefix = f'run-{run_index+1:02d}'
        valid = valid_base & np.isfinite(expected_r2[:, run_index])
        max_error = 0.
        compared = 0
        ranks = {}
        task_cache = {}
        with tempfile.TemporaryDirectory(prefix='.reconstruct-', dir=folder) as temporary:
            temporary = Path(temporary)
            started = monotonic()
            logger.info('Buffering BOLD: run %d/%d; %d volumes; %.2f GiB; contiguous volume writes',
                        run_index+1, count, n_time, n_voxels*n_time*4/2**30)
            data = np.memmap(temporary/'bold.dat', mode='w+', dtype='float32',
                             shape=(n_voxels, n_time), order='F')
            next_message = started + 60
            for time in range(n_time):
                data[:, time] = np.asarray(image.dataobj[..., time], dtype=np.float32).reshape(-1)
                now = monotonic()
                if now >= next_message:
                    logger.info('Buffering BOLD: run %d; volume %d/%d; elapsed %.1f s',
                                run_index+1, time+1, n_time, now-started)
                    next_message = now+60
            logger.info('BOLD buffering finished: run %d; elapsed %.1f s',
                        run_index+1, monotonic()-started)
            reconstruct_started = monotonic()
            next_message = reconstruct_started+60
            arrays = {kind: np.memmap(temporary/(kind+'.dat'), mode='w+', dtype='float32', shape=(n_voxels, n_time))
                      for kind in ('taskpoly', 'fullnuisance')}
            for start in range(0, n_voxels, voxel_chunk):
                now = monotonic()
                if now >= next_message:
                    logger.info('Reconstructing: run %d; voxel %d/%d; elapsed %.1f s',
                                run_index+1, start, n_voxels, now-reconstruct_started)
                    next_message = now+60
                stop = min(n_voxels, start+voxel_chunk)
                block_betas = np.asarray(betas[start:stop], dtype=float)
                block_valid = valid[start:stop].copy()
                block_valid &= np.isfinite(block_betas).all(axis=1)
                block_valid &= np.isfinite(data[start:stop]).all(axis=1)
                valid[start:stop] = block_valid
                for array in arrays.values():
                    array[start:stop] = 0
                selected = start + np.flatnonzero(block_valid)
                if not selected.size:
                    continue
                y = np.asarray(data[selected], dtype=float).T
                beta = np.asarray(betas[selected], dtype=float)
                beta = (beta * means[selected, None]/100 - offsets[selected, None]) / scales[selected, None]
                error = y.copy()
                for hrf in np.unique(indices[selected]):
                    columns = np.flatnonzero(indices[selected] == hrf)
                    if hrf not in task_cache:
                        task_cache[hrf] = (np.column_stack([np.convolve(design[:, trial], hrfs[:, hrf])[:n_time] for trial in active])
                                           if active.size else np.empty((n_time, 0)))
                        ranks[str(int(hrf))] = int(np.linalg.matrix_rank(np.column_stack([nuisance, task_cache[hrf]])))
                    task = task_cache[hrf]
                    error[:, columns] -= task @ beta[np.ix_(columns, active)].T
                taskpoly = _project(error, q_poly)
                denominator = np.square(_project(y, q_poly)).sum(axis=0)
                # GLMsingle's projections use float32. A nominally nonzero
                # denominator below that numerical floor cannot validate R2.
                floor = 64*np.finfo(np.float32).eps**2*np.square(y).sum(axis=0)
                usable = denominator > floor
                valid[selected[~usable]] = False
                if usable.any():
                    actual = 100 * (1 - np.square(taskpoly[:, usable]).sum(axis=0)/denominator[usable])
                    delta = np.abs(actual-expected_r2[selected[usable], run_index])
                    if not np.isfinite(delta).all():
                        raise ValueError('Non-finite reconstructed R2run')
                    max_error = max(max_error, float(delta.max()))
                    compared += int(usable.sum())
                factor = 100*scales[selected]/means[selected] if units == 'beta' else np.ones(selected.size)
                converted = {'taskpoly': (taskpoly*factor).T,
                             'fullnuisance': (_project(error, q_full)*factor).T}
                for kind, values in converted.items():
                    if np.any(np.abs(values[usable]) > np.finfo(np.float32).max) or not np.isfinite(values[usable]).all():
                        raise ValueError('Residual values are not representable as finite float32')
                    values[~usable] = 0
                    arrays[kind][selected] = values
            if not compared or max_error > r2_atol:
                raise ValueError(f'Run {run_index}: R2run reconstruction failed; compared={compared}; '
                                 f'maximum error={max_error:.6g} percentage points; tolerance={r2_atol}')
            logger.info('Residual R2run validation passed: run %d; valid=%d; excluded=%d; maximum error=%.6g percentage points',
                        run_index+1, valid.sum(), (~valid).sum(), max_error)
            logger.info('Reconstruction finished: run %d; elapsed %.1f s',
                        run_index+1, monotonic()-reconstruct_started)
            mask = folder/(prefix+'_model-typed_desc-residualvalid_mask.nii.gz')
            base = dict(Model='GLMsingleTYPED', RunIndex=run_index, Sources=[identity(bold)],
                        ReconstructionMethod='typed-inverse-autoscale-projection-v1',
                        ImplementationSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        SourceDescription=tags.get('desc'), NumberOfVolumes=n_time,
                        RepetitionTime=float(design_info['tr']),
                        ModelFile=str(Path(output_dir)/'TYPED_FITHRF_GLMDENOISE_RR.npy'),
                        DesignInfoFile=str(Path(output_dir)/'DESIGNINFO.npy'), ValidMaskFile=str(mask),
                        Units='input_BOLD' if units == 'input' else 'saved_beta_linear_scaling',
                        ResidualScaling='1' if units == 'input' else '100 * autoscale_slope / abs(meanvol); no offset',
                        PolynomialDegrees=degrees, PolynomialRank=q_poly.shape[1],
                        NuisanceRank=q_full.shape[1], ExtraRegressorCount=extra.shape[1], NoisePCCount=pcnum,
                        FullDesignRankByHRF=ranks, ResidualDegreesOfFreedom=None,
                        DegreesOfFreedomNote='Ridge/adaptive residual DOF not estimated; ranks are diagnostics only',
                        ValidVoxels=int(valid.sum()), ExcludedVoxels=int((~valid).sum()),
                        UninvertibleScaleVoxels=int((~np.isfinite(scales) | (scales == 0)).sum()),
                        R2Validation=dict(Definition='taskpoly in input units', ComparedVoxels=compared,
                                          MaxAbsoluteErrorPercentagePoints=max_error, Tolerance=r2_atol,
                                          DetrendedVarianceFloor='64 * eps(float32)^2 * sum(input_BOLD^2)'),
                        TemporalWhitening=False, TaskSignalMayRemain=True)
            for kind, array in arrays.items():
                writing_started = monotonic()
                logger.info('Writing residual NIfTI: run %d; definition=%s', run_index+1, kind)
                array.flush()
                name = prefix + f'_model-typed_desc-{kind}_residuals'
                path = folder/(name+'.nii.gz')
                header = image.header.copy()
                header.set_data_dtype(np.float32)
                staged = temporary/(name+'.nii.gz')
                nib.save(nib.Nifti1Image(array.reshape(image.shape), image.affine, header), staged)
                staged.replace(path)
                sidecar = folder/(name+'.json')
                atomic_json(sidecar, dict(base, ResidualDefinition=kind))
                outputs.extend([path, sidecar])
                logger.info('Residual NIfTI finished: %s; elapsed %.1f s', path,
                            monotonic()-writing_started)
            staged_mask = temporary/mask.name
            nib.save(nib.Nifti1Image(valid.reshape(spatial).astype('uint8'), image.affine), staged_mask)
            staged_mask.replace(mask)
            outputs.append(mask)
    return outputs
