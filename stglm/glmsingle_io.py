"""Validated GLMsingle run designs and explicit beta-to-trial provenance."""
import fnmatch
import json
import hashlib
import os
from pathlib import Path
import re
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd


def entities(path):
    return dict(re.findall(r'(?:^|_)(sub|ses|task|run|space|desc|acq|echo)-([^_/.]+)', Path(path).name))


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix='.provenance.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, default=lambda obj: obj.tolist() if isinstance(obj, np.ndarray) else obj.item() if isinstance(obj, np.generic) else str(obj))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_runs(bold_files, event_files, tr, include_trials=None, stim_duration=0,
                 event_durations=(), additional_confounds=()):
    if not np.isfinite(tr) or tr <= 0 or not np.isfinite(stim_duration) or stim_duration < 0:
        raise ValueError('TR must be positive and stimulus duration must be nonnegative')
    if not bold_files or len(set(map(str, bold_files))) != len(bold_files):
        raise ValueError('BOLD inputs are empty or duplicated')
    rules = []
    for rule in event_durations:
        pattern, value = rule.rsplit('=', 1)
        value = float(value)
        if not pattern or not np.isfinite(value) or value < 0:
            raise ValueError('Invalid event duration rule: ' + rule)
        rules.append((pattern, value))
    pairs = []
    used = set()
    for bold in bold_files:
        tags = entities(bold)
        candidates = []
        for events in event_files:
            other = entities(events)
            def matches(key):
                left, right = other.get(key), tags.get(key)
                if key == 'run' and left and right and left.isdigit() and right.isdigit():
                    return int(left) == int(right)
                return left == right
            if all(matches(key) for key in ('sub', 'ses', 'task', 'run', 'acq')):
                candidates.append(events)
        if len(candidates) != 1 or str(candidates[0]) in used:
            raise ValueError('Expected exactly one distinct events match for ' + str(bold))
        used.add(str(candidates[0]))
        pairs.append((bold, candidates[0], tags))
    def order(item):
        return tuple(value.zfill(20) if value.isdigit() else value
                     for value in (item[2].get(k, '') for k in ('sub','ses','task','run','acq','echo')))
    pairs.sort(key=order)
    streams = {tuple(tags.get(k, '') for k in ('sub','task','space','desc','acq','echo')) for _, _, tags in pairs}
    if len(streams) != 1:
        raise ValueError('GLMsingle inputs mix subjects, tasks, spaces or preprocessing streams')
    frames = []
    for bold, events, tags in pairs:
        df = pd.read_csv(events, sep='\t')
        if not {'onset','duration','trial_type'} <= set(df.columns):
            raise ValueError('Events require onset, duration and trial_type: ' + str(events))
        if df.trial_type.isna().any() or df.trial_type.astype(str).str.strip().eq('').any():
            raise ValueError('Events contain missing trial types: ' + str(events))
        df['source_row'] = np.arange(len(df))
        if include_trials is not None:
            df = df[df.trial_type.isin(include_trials)].copy()
        if df.empty:
            raise ValueError('No selected events for ' + str(events))
        frames.append(df)
    conditions = sorted({str(label) for df in frames for label in df.trial_type})
    for condition in conditions:
        duration = stim_duration
        for pattern, value in rules:
            if fnmatch.fnmatchcase(condition, pattern):
                duration = value
        if duration != stim_duration:
            raise ValueError('GLMsingle uses one common stimulus duration; mixed event durations are unsupported. '
                             f'{condition} requests {duration}s but --stim-duration is {stim_duration}s')
    data, designs, trials, regressors, inputs = [], [], [], [], []
    affine = shape = None
    for run_index, ((bold, events, tags), df) in enumerate(zip(pairs, frames)):
        img = nib.load(str(bold))
        if len(img.shape) != 4:
            raise ValueError('Expected 4D BOLD: ' + str(bold))
        if shape is not None and (img.shape[:3] != shape or not np.allclose(img.affine, affine, rtol=0, atol=1e-5)):
            raise ValueError('BOLD grids differ across runs')
        shape, affine = img.shape[:3], img.affine
        values = img.get_fdata(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError('BOLD contains non-finite values: ' + str(bold))
        onsets = pd.to_numeric(df.onset, errors='raise').to_numpy(float)
        durations = pd.to_numeric(df.duration, errors='raise').to_numpy(float)
        if not np.isfinite(onsets).all() or not np.isfinite(durations).all() or (durations < 0).any():
            raise ValueError('Invalid event timing: ' + str(events))
        volumes = np.rint(onsets / tr).astype(int)
        if (onsets < 0).any() or (volumes < 0).any() or (volumes >= img.shape[-1]).any():
            raise ValueError('Selected event outside acquired run: ' + str(events))
        if len(np.unique(volumes)) != len(volumes):
            raise ValueError('Selected events collide after onset-to-TR rounding: ' + str(events))
        df = df.assign(design_volume=volumes).sort_values('design_volume', kind='stable')
        design = np.zeros((img.shape[-1], len(conditions)), dtype=int)
        for _, row in df.iterrows():
            design[int(row.design_volume), conditions.index(str(row.trial_type))] = 1
            trials.append(dict(beta_index=len(trials)+1, subject=tags.get('sub',''), session=tags.get('ses',''),
                task=tags.get('task',''), run=tags.get('run',''), run_index=run_index,
                trial_id=f"{Path(events).name.removesuffix('_events.tsv')}_row-{int(row.source_row)+1}",
                trial_type=str(row.trial_type), space=tags.get('space',''), desc=tags.get('desc',''),
                acquisition=tags.get('acq',''), echo=tags.get('echo',''),
                onset=float(row.onset), duration=float(row.duration), modeled_duration=float(stim_duration),
                design_volume=int(row.design_volume), modeled_onset=float(row.design_volume)*tr,
                source_row=int(row.source_row), events_file=str(Path(events).resolve()), bold_file=str(Path(bold).resolve())))
        columns = []
        for template in additional_confounds:
            source = Path(template.format(subject=tags.get('sub',''), session=tags.get('ses',''), run=tags.get('run',''), task=tags.get('task','')))
            frame = pd.read_csv(source, sep='\t').apply(pd.to_numeric, errors='raise')
            array = frame.to_numpy(float)
            if array.shape[0] != img.shape[-1] or not np.isfinite(array).all():
                raise ValueError('Confounds must be finite with one row per BOLD volume: ' + str(source))
            columns.append(array); inputs.append(identity(source))
        if columns:
            array = np.column_stack(columns)
            centered = array - array.mean(axis=0)
            if np.linalg.matrix_rank(centered) < array.shape[1]:
                raise ValueError('Additional confounds contain constant or linearly dependent columns')
            regressors.append(array)
        inputs.extend([identity(bold), identity(events)])
        data.append(values); designs.append(design)
    return pairs, data, designs, pd.DataFrame(trials), conditions, regressors, inputs


def fit_glmsingle(bold_files, event_files, tr, include_trials=None, output_dir=None,
                  stim_duration=0, event_durations=(), additional_confounds=(), tags=None, params=None):
    from glmsingle.glmsingle import GLM_single
    from stglm import utils
    import importlib.metadata
    import fcntl
    pairs, data, designs, trials, conditions, regressors, inputs = prepare_runs(
        bold_files, event_files, tr, include_trials, stim_duration, event_durations, additional_confounds)
    tags = pairs[0][2]
    stream = '_'.join(f'{key}-{tags[key]}' for key in ('task','space','desc','acq','echo') if tags.get(key))
    # Sessions may be fitted jointly; run_index disambiguates repeated run labels.
    final = Path(output_dir) / ('sub-' + tags['sub']) / stream
    final.mkdir(parents=True, exist_ok=True)
    with (final / '.fit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = final / 'fit_provenance.json'
        if marker.exists():
            raise ValueError('Completed fit already exists; use a separate output directory: ' + str(final))
        import logging
        if not np.allclose(trials.duration.to_numpy(float), float(stim_duration)):
            logging.getLogger(__name__).warning(
                'Recorded event durations differ from the common GLMsingle modeled duration (%s s); see trial manifests', stim_duration)
        parameters = dict(params or {})
        if regressors:
            if parameters.get('extra_regressors') is not None and parameters.get('extra_regressors') is not False:
                raise ValueError('Specify additional confounds either through config or files, not both')
            parameters['extra_regressors'] = regressors
        # Prevent files from a failed older fit being mistaken for current output.
        existing = list(final.glob('TYPE*.npy')) + list(final.glob('model-*_beta-*.nii.gz'))
        if existing:
            raise ValueError('Partial fit outputs already exist; use a fresh output directory: ' + str(final))
        session_values = [tags.get('ses','') for _, _, tags in pairs]
        if len(set(session_values)) > 1 and 'sessionindicator' not in parameters:
            parameters['sessionindicator'] = np.asarray([[sorted(set(session_values)).index(v)+1 for v in session_values]])
        import copy
        requested_parameters = copy.deepcopy(parameters)
        extra_info = []
        supplied_extra = requested_parameters.get('extra_regressors')
        if supplied_extra is not None and supplied_extra is not False:
            if not isinstance(supplied_extra, (list, tuple)) or len(supplied_extra) != len(data):
                raise ValueError('extra_regressors must contain one matrix per run')
            for matrix, run_data in zip(supplied_extra, data):
                matrix = np.asarray(matrix, dtype=np.float64)
                if matrix.ndim != 2 or matrix.shape[0] != run_data.shape[-1] or not np.isfinite(matrix).all():
                    raise ValueError('extra_regressors must be finite time-by-regressor matrices')
                extra_info.append(dict(shape=list(matrix.shape), sha256=hashlib.sha256(matrix.tobytes()).hexdigest()))
        GLM_single(params=copy.deepcopy(parameters)).fit(designs, data, stim_duration, tr,
            outputdir=str(final), figuredir=str(final/'figures'))
        models = {'typeb':'TYPEB_FITHRF.npy', 'typed':'TYPED_FITHRF_GLMDENOISE_RR.npy'}
        results = {}
        outputs = []
        for model, filename in models.items():
            raw = final / filename
            if not raw.is_file():
                continue
            fitted = np.load(raw, allow_pickle=True).item()
            if fitted['betasmd'].shape != (*data[0].shape[:3], len(trials)):
                raise ValueError('GLMsingle beta dimensions differ from the modeled trial table')
            utils.save_glmsingle_outputs(fitted, pairs[0][0], str(final), prefix='model-'+model)
            files = sorted(final.glob(f'model-{model}_beta-*.nii.gz'))
            if len(files) != len(trials):
                raise ValueError('Exported beta count differs from modeled trials')
            frame = trials.copy();frame['model'] = model;frame['beta_file'] = [p.name for p in files]
            manifest = final / f'model-{model}_trial_manifest.tsv'
            frame.to_csv(manifest, sep='\t', index=False)
            outputs.extend(identity(p) for p in files+[manifest, raw])
            results[model] = list(map(str, files))
        if not results:
            raise ValueError('GLMsingle produced no supported B/D beta files; enable file outputs')
        np.savetxt(final/'trial_list.txt', trials.trial_type.to_numpy(), fmt='%s')
        versions = {}
        for package in ('stglm','glmsingle','numpy','nibabel'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = 'unknown'
        for saved in inputs:
            if identity(saved['path']) != saved:
                raise ValueError('Input changed during GLMsingle fitting: ' + saved['path'])
        atomic_json(marker, dict(status='complete', fit_scope='joint_runs', inputs=inputs, outputs=outputs,
            conditions=conditions, tr=float(tr), stim_duration=float(stim_duration),
            parameters={key: value for key, value in requested_parameters.items() if key != 'extra_regressors'},
            extra_regressors=extra_info,
            run_order=[dict(bold=str(b), events=str(e), entities=t, n_volumes=values.shape[-1])
                       for (b,e,t), values in zip(pairs,data)], versions=versions))
        return results
