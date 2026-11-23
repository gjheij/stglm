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


def validate_event_times(df, img, bold, events, tr, drop_invalid=False):
    """Validate timing and optionally enforce GLMsingle's eight-second end buffer."""
    onsets = pd.to_numeric(df.onset, errors='raise').to_numpy(float)
    durations = pd.to_numeric(df.duration, errors='raise').to_numpy(float)
    if not np.isfinite(onsets).all() or not np.isfinite(durations).all() or (durations < 0).any():
        raise ValueError('Invalid event timing: ' + str(events))
    volumes = np.rint(onsets / tr).astype(int)
    outside_grid = (onsets < 0) | (volumes < 0) | (volumes >= img.shape[-1])
    # Match GLMsingle exactly: sampled volumes AFTER the rounded onset row.
    # Its warning uses (number_of_volumes - onset_row - 1) * TR, not the
    # continuous acquisition endpoint minus the original event onset.
    end_buffer = (img.shape[-1] - volumes - 1) * tr
    invalid = outside_grid | ((end_buffer < 8.) if drop_invalid else False)
    excluded = pd.DataFrame()
    if invalid.any():
        bad = df.loc[invalid, ['source_row', 'trial_type', 'onset', 'duration']].copy()
        bad['tsv_line'] = bad.pop('source_row').astype(int) + 2
        bad['design_volume'] = volumes[invalid]
        bad['end_buffer_seconds'] = end_buffer[invalid]
        acquisition_end = img.shape[-1] * tr
        bad['reason'] = np.where(onsets[invalid] < 0, 'negative onset',
            np.where(onsets[invalid] >= acquisition_end, 'onset beyond acquisition',
                np.where((volumes[invalid] < 0) | (volumes[invalid] >= img.shape[-1]),
                         'rounding beyond final design row', 'less than 8 s after modeled onset')))
        if not drop_invalid:
            raise ValueError(
                'Selected event outside acquired run/design grid.\n'
                f'BOLD: {bold}\nEvents: {events}\n'
                f'Volumes: {img.shape[-1]}; TR: {tr:g} s; acquisition span: [0, {acquisition_end:g}) s; '
                f'last sampled time: {(img.shape[-1]-1)*tr:g} s; valid design indices: 0..{img.shape[-1]-1}.\n'
                f'Invalid selected events: {len(bad)} of {len(df)} (first 10 shown):\n'
                + bad.head(10).to_string(index=False)
                + '\nCheck the BOLD/events match, TR, time origin and any removed volumes. '
                  'No events were dropped or shifted and fitting has not started.')
        excluded = df.loc[invalid].copy()
        excluded['tsv_line'] = excluded['source_row'].astype(int) + 2
        excluded['design_volume'] = volumes[invalid]
        excluded['end_buffer_seconds'] = end_buffer[invalid]
        excluded['reason'] = bad['reason'].to_numpy()
        df = df.loc[~invalid].copy()
        volumes = volumes[~invalid]
        if df.empty:
            raise ValueError('No valid selected events remain after --drop-invalid: ' + str(events))
    if len(np.unique(volumes)) != len(volumes):
        raise ValueError('Selected events collide after onset-to-TR rounding: ' + str(events))
    df = df.assign(design_volume=volumes).sort_values('design_volume', kind='stable')
    return df, excluded


def prepare_runs(bold_files, event_files, tr, include_trials=None, stim_duration=0,
                 event_durations=(), additional_confounds=(), drop_invalid=False, return_diagnostics=False, load_data=True):
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
    frames, excluded_frames, late_frames, run_summaries = [], [], [], []
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
        img = nib.load(str(bold))
        if len(img.shape) != 4:
            raise ValueError('Expected 4D BOLD: ' + str(bold))
        n_selected = len(df)
        df, excluded = validate_event_times(df, img, bold, events, tr, drop_invalid)
        metadata = dict(subject=tags.get('sub',''), session=tags.get('ses',''), run=tags.get('run',''),
                        bold_file=str(Path(bold).resolve()), events_file=str(Path(events).resolve()))
        if not excluded.empty:
            excluded_frames.append(excluded.assign(**metadata))
        # This is a QC warning only: a late event is not automatically excluded.
        remaining = img.shape[-1]*tr - df.onset.to_numpy(float)
        late = df.loc[remaining < 16.].copy()
        if not late.empty:
            late['remaining_response_seconds'] = remaining[remaining < 16.]
            late_frames.append(late.assign(**metadata))
        run_summaries.append(dict(**metadata, n_volumes=img.shape[-1], tr=float(tr),
                                 selected_events=n_selected, excluded_events=len(excluded), retained_events=len(df)))
        if drop_invalid:
            import logging
            logging.getLogger(__name__).warning('Event selection for %s: selected=%d; excluded=%d; retained=%d',
                                                bold, n_selected, len(excluded), len(df))
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
                end_buffer_seconds=float((img.shape[-1]-int(row.design_volume)-1)*tr),
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
        designs.append(design)
    # Finish timing/design validation for every run before reading large arrays.
    for bold, _, _ in pairs:
        image = nib.load(str(bold))
        if load_data:
            values = image.get_fdata(dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError('BOLD contains non-finite values: ' + str(bold))
            data.append(values)
        else:
            data.append(image)
    result = (pairs, data, designs, pd.DataFrame(trials), conditions, regressors, inputs)
    if not return_diagnostics:
        return result
    empty_columns = ['source_row','tsv_line','trial_type','onset','duration','design_volume','end_buffer_seconds','reason',
                     'subject','session','run','bold_file','events_file']
    diagnostics = dict(
        excluded_events=pd.concat(excluded_frames, ignore_index=True) if excluded_frames else pd.DataFrame(columns=empty_columns),
        late_events=pd.concat(late_frames, ignore_index=True) if late_frames else pd.DataFrame(columns=[*empty_columns, 'remaining_response_seconds']),
        run_summary=pd.DataFrame(run_summaries))
    return (*result, diagnostics)


def validate_saved_design(folder, designs, tr, duration, conditions, trials, parameters):
    """Verify legacy raw outputs against the exact designs before export recovery."""
    path = folder / 'DESIGNINFO.npy'
    if not path.is_file():
        raise ValueError('Export recovery requires DESIGNINFO.npy: ' + str(path))
    saved = np.load(path, allow_pickle=True).item()
    actual = saved.get('design', [])
    if (len(actual) != len(designs)
            or not np.isclose(float(saved['tr']), tr, rtol=0, atol=1e-6)
            or not np.isclose(float(saved['stimdur']), duration, rtol=0, atol=1e-9)
            or any(not np.array_equal(old, new) for old,new in zip(actual,designs))):
        raise ValueError('Saved GLMsingle design/TR/duration differs from requested export settings')
    expected_order = [conditions.index(value) for value in trials.trial_type]
    if not np.array_equal(np.asarray(saved['stimorder']).ravel(),expected_order):
        raise ValueError('Saved GLMsingle trial order differs from the reconstructed trial table')
    if not np.array_equal(np.asarray(saved['numtrialrun']).ravel(),[int(design.sum()) for design in designs]):
        raise ValueError('Saved GLMsingle per-run trial counts differ')
    original = saved.get('params', {})
    def matrices(value):
        if value is None or value is False:
            return []
        if isinstance(value,(list,tuple)) and len(value)==1 and value[0] is False:
            return []
        return list(value)
    old_extra = matrices(original.get('extra_regressors'))
    new_extra = matrices(parameters.get('extra_regressors'))
    if len(old_extra)!=len(new_extra) or any(
            np.asarray(a).shape!=np.asarray(b).shape or not np.allclose(a,b,rtol=0,atol=1e-12)
            for a,b in zip(old_extra,new_extra)):
        raise ValueError('Saved GLMsingle additional regressors differ from current inputs')
    outputs = original.get('wantfileoutputs', [1,1,1,1])
    required = []
    if outputs[1]:
        required.append('TYPEB_FITHRF.npy')
    if outputs[3] and original.get('wantfracridge',1):
        required.append('TYPED_FITHRF_GLMDENOISE_RR.npy')
    if not required or any(not (folder/name).is_file() for name in required):
        raise ValueError('Requested GLMsingle raw B/D outputs are incomplete; refusing recovery')


def export_conditionwise(betas, reference, folder, model, trials, run_order, tr, run_metadata=None):
    """Write 4D trial series and HALFpipe-like JSON metadata per run/condition."""
    image = nib.load(str(reference))
    target = folder / 'conditionwise' / ('model-' + model)
    target.mkdir(parents=True, exist_ok=True)
    outputs = []
    names = set()
    for (run_index, condition), group in trials.groupby(['run_index','trial_type'],sort=False):
        tags = run_order[int(run_index)]['entities']
        prefix = '_'.join(f'{key}-{tags[key]}' for key in ('sub','ses','task','acq','run','echo','space','desc') if tags.get(key))
        # Keep original condition spelling, including percent signs and underscores,
        # but forbid path separators and reject sanitization collisions.
        safe_condition = re.sub(r'[^A-Za-z0-9_%+.-]', '-',str(condition))
        name = f'{prefix}_feature-GLMsingle{model.upper()}_condition-{safe_condition}_stat-effect_statmap'
        if name in names:
            raise ValueError('Condition-wise output filename collision: ' + name)
        names.add(name)
        selected = group.beta_index.to_numpy(int)-1
        values = np.asarray(betas[...,selected],dtype=np.float32)
        header = image.header.copy()
        header.set_data_dtype(np.float32)
        header.set_data_shape(values.shape)
        header.set_zooms((*image.header.get_zooms()[:3],1.))
        header.set_xyzt_units(t='unknown')
        path = target/(name+'.nii.gz')
        nib.save(nib.Nifti1Image(values,image.affine,header),path)
        sidecar = target/(name+'.json')
        import copy
        metadata = copy.deepcopy((run_metadata or {}).get(int(run_index),{}))
        model_info = dict(metadata.get('Model') or {})
        model_info.update(Estimator='GLMsingle',OutputType=model,Condition=str(condition),JointRuns=True)
        metadata.update(RepetitionTime=float(tr), NumberOfVolumes=run_order[int(run_index)]['n_volumes'],
            Model=model_info,FourthDimension='trial',
            Sources=[run_order[int(run_index)]['bold'],run_order[int(run_index)]['events']],
            Events=[dict(onset=float(row.onset),duration=float(row.modeled_duration),
                         RecordedDuration=float(row.duration),ModeledOnset=float(row.modeled_onset),
                         TrialID=row.trial_id,BetaIndex=int(row.beta_index)) for _,row in group.iterrows()])
        atomic_json(sidecar,metadata)
        metadata = target/(name+'_trials.tsv')
        group = group.copy()
        group['volume_index'] = np.arange(len(group))
        group.to_csv(metadata,sep='\t',index=False)
        outputs.extend([path,sidecar,metadata])
    return outputs


def fit_glmsingle(bold_files, event_files, tr, include_trials=None, output_dir=None,
                  stim_duration=0, event_durations=(), additional_confounds=(), tags=None, params=None, drop_invalid=False,
                  export_only=False, split_trial_types=False, qc_runs=None,
                  fd_threshold=.2, gm_threshold=.5, tsnr_regress_poly=2,
                  export_residuals=False, residual_units='beta', residual_chunk_voxels=2048):
    """Fit/recover GLMsingle and optionally reconstruct TYPE-D residuals.

    ``export_only=True, export_residuals=True`` uses completed models without
    invoking GLM_single.fit. Reconstruction, unit conversion, nuisance handling,
    validation and ridge-DOF limitations are documented in
    ``glmsingle_residuals.export_type_d_residuals`` and its module docstring.
    """
    if export_residuals and (residual_units not in ('beta', 'input') or residual_chunk_voxels < 1):
        raise ValueError('Invalid residual units or voxel chunk')
    from glmsingle.glmsingle import GLM_single
    from stglm import utils
    import importlib.metadata
    import fcntl
    pairs, data, designs, trials, conditions, regressors, inputs, diagnostics = prepare_runs(
        bold_files, event_files, tr, include_trials, stim_duration, event_durations, additional_confounds,
        drop_invalid=drop_invalid, return_diagnostics=True, load_data=False)
    tags = pairs[0][2]
    stream = '_'.join(f'{key}-{tags[key]}' for key in ('task','space','desc','acq','echo') if tags.get(key))
    # Sessions may be fitted jointly; run_index disambiguates repeated run labels.
    final = Path(output_dir) / ('sub-' + tags['sub']) / stream
    final.mkdir(parents=True, exist_ok=True)
    # GLMsingle recreates outputdir; locks and the input plan must live outside it.
    lock_path = final.parent / ('.' + final.name + '.fit.lock')
    plan_path = final.parent / ('.' + final.name + '.fit_input.json')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = final / 'fit_provenance.json'
        completed_export = marker.is_file()
        if completed_export:
            if not export_only:
                raise ValueError('Completed fit already exists; use --export-only to add exports or a separate output directory to refit: ' + str(final))
            previous = json.loads(marker.read_text())
            if previous.get('inputs') != inputs:
                raise ValueError('Inputs differ from the completed export; refusing re-export')
            for saved in previous['outputs']:
                if identity(saved['path']) != saved:
                    raise ValueError('Existing exported output changed: ' + saved['path'])
        # Refresh existing condition-wise sidecars when run QC is added later.
        split_trial_types = split_trial_types or (final/'conditionwise').is_dir()
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
        if existing and not export_only:
            raise ValueError('Existing model outputs found; use --export-only to recover exports, or a fresh output directory to refit: ' + str(final))
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
        run_order = [dict(bold=str(b), events=str(e), entities=t, n_volumes=image.shape[-1])
                     for (b,e,t), image in zip(pairs,data)]
        spatial_shape = tuple(data[0].shape[:3])
        plan = dict(inputs=inputs, conditions=conditions, run_order=run_order,
                    tr=float(tr), stim_duration=float(stim_duration), drop_invalid=bool(drop_invalid),
                    minimum_end_buffer_seconds=8. if drop_invalid else None,
                    parameters={key:value for key,value in requested_parameters.items() if key != 'extra_regressors'},
                    extra_regressors=extra_info,
                    trials_sha256=hashlib.sha256(trials.to_csv(index=False).encode()).hexdigest())
        # Convert numpy settings identically to atomic_json before comparing a saved plan.
        plan = json.loads(json.dumps(plan, default=lambda obj: obj.tolist() if isinstance(obj,np.ndarray) else obj.item() if isinstance(obj,np.generic) else str(obj)))
        recovery_validation = 'original_input_plan'
        if export_only:
            if plan_path.is_file():
                if json.loads(plan_path.read_text()) != plan:
                    raise ValueError('Inputs/settings differ from the saved fit plan; refusing export recovery')
            else:
                recovery_validation = 'saved_design_only_legacy_input_history_unavailable'
                logging.getLogger(__name__).warning(
                    'Legacy export recovery: checking DESIGNINFO.npy and beta dimensions; original input identities were not saved before this fit')
            validate_saved_design(final, designs, tr, stim_duration, conditions, trials, requested_parameters)
            if completed_export:
                # Do not leave a stale success marker if metadata enrichment fails
                # halfway through. Raw fits and the previous provenance survive.
                atomic_json(final.parent/('.'+final.name+'.previous_export.json'),previous)
                marker.unlink()
        else:
            atomic_json(plan_path, plan)
            if not diagnostics['late_events'].empty:
                logging.getLogger(__name__).warning(
                    '%d retained events have less than 16 s after onset; QC will be saved after fitting',len(diagnostics['late_events']))
            # Load large arrays only after output guards and save the input plan
            # outside the directory that GLMsingle deletes at initialization.
            fit_data = []
            for bold,_,_ in pairs:
                values = nib.load(str(bold)).get_fdata(dtype=np.float32)
                if not np.isfinite(values).all():
                    raise ValueError('BOLD contains non-finite values: ' + str(bold))
                fit_data.append(values)
            del values
            try:
                GLM_single(params=copy.deepcopy(parameters)).fit(designs, fit_data, stim_duration, tr,
                    outputdir=str(final), figuredir=str(final/'figures'))
            finally:
                # Release BOLD data before loading large model dictionaries for export.
                del fit_data
        # GLMsingle has finished deleting/recreating its outputdir. Diagnostics
        # belong here, not before fit(). Recovery never invokes the fitter.
        final.mkdir(parents=True, exist_ok=True)
        diagnostic_paths = []
        for name, frame in diagnostics.items():
            path = final / (name + '.tsv')
            frame.to_csv(path, sep='\t', index=False)
            diagnostic_paths.append(path)
        run_metadata = {}
        qc_outputs = []
        qc_inputs = []
        if qc_runs is not None:
            from stglm.glmsingle_qc import build_run_metadata
            run_metadata,qc_outputs,qc_inputs=build_run_metadata(
                pairs,qc_runs,final,additional_confounds=additional_confounds,
                fd_threshold=fd_threshold,gm_threshold=gm_threshold,
                regress_poly=tsnr_regress_poly,extra_regressors=extra_info)
        design_info=final/'DESIGNINFO.npy'
        actual_parameters=np.load(design_info,allow_pickle=True).item()['params'] if design_info.is_file() else requested_parameters
        actual_parameters=dict(actual_parameters)
        actual_parameters['ExtraRegressorsByRun']=extra_info
        if export_residuals and not (final/'TYPED_FITHRF_GLMDENOISE_RR.npy').is_file():
            raise ValueError('TYPE-D model is required for residual reconstruction')
        models = {'typeb':'TYPEB_FITHRF.npy', 'typed':'TYPED_FITHRF_GLMDENOISE_RR.npy'}
        results = {}
        outputs = [identity(path) for path in diagnostic_paths+qc_outputs]
        # A later metadata/condition export must not drop existing residual
        # products from the completion manifest when no new residuals are requested.
        residual_record = dict(enabled=bool(export_residuals), units=residual_units,
                               voxel_chunk=int(residual_chunk_voxels))
        if completed_export and not export_residuals and previous.get('residual_export', {}).get('enabled'):
            outputs.extend(item for item in previous['outputs']
                           if Path(item['path']).is_relative_to((final/'residuals').resolve()))
            residual_record = previous['residual_export']
        for model, filename in models.items():
            raw = final / filename
            if not raw.is_file():
                continue
            fitted = np.load(raw, allow_pickle=True).item()
            if fitted['betasmd'].shape != (*spatial_shape, len(trials)):
                raise ValueError('GLMsingle beta dimensions differ from the modeled trial table')
            if export_residuals and model == 'typed':
                from stglm.glmsingle_residuals import export_type_d_residuals
                residual_files = export_type_d_residuals(
                    fitted, np.load(design_info, allow_pickle=True).item(),
                    [bold for bold,_,_ in pairs], final, units=residual_units,
                    voxel_chunk=residual_chunk_voxels)
                outputs.extend(identity(path) for path in residual_files)
            if not completed_export:
                utils.save_glmsingle_outputs(fitted, pairs[0][0], str(final), prefix='model-'+model)
            model_run_metadata = run_metadata
            if run_metadata:
                from stglm.glmsingle_qc import model_metadata
                model_run_metadata={index:model_metadata(base,model,actual_parameters,fitted,index)
                                    for index,base in run_metadata.items()}
            grouped_paths = export_conditionwise(fitted['betasmd'], pairs[0][0], final, model, trials, run_order, tr,
                                                run_metadata=model_run_metadata) if split_trial_types else []
            files = sorted(final.glob(f'model-{model}_beta-*.nii.gz'))
            if len(files) != len(trials):
                raise ValueError('Exported beta count differs from modeled trials')
            # Individual betas get the same run QC/metadata as the optional 4D files.
            sidecars=[]
            for (_,row),path in zip(trials.iterrows(),files):
                base=copy.deepcopy(model_run_metadata.get(int(row.run_index),{}))
                info=dict(base.get('Model') or {})
                info.update(Estimator='GLMsingle',OutputType=model,Condition=row.trial_type,JointRuns=True)
                base.update(Model=info,RepetitionTime=float(tr),NumberOfVolumes=run_order[int(row.run_index)]['n_volumes'],
                    Events=[dict(onset=float(row.onset),duration=float(row.modeled_duration),
                                 RecordedDuration=float(row.duration),ModeledOnset=float(row.modeled_onset),
                                 TrialID=row.trial_id,BetaIndex=int(row.beta_index))])
                sidecar=Path(str(path).removesuffix('.nii.gz')+'.json')
                atomic_json(sidecar,base);sidecars.append(sidecar)
            frame = trials.copy();frame['model'] = model;frame['beta_file'] = [p.name for p in files]
            manifest = final / f'model-{model}_trial_manifest.tsv'
            if not completed_export:
                frame.to_csv(manifest, sep='\t', index=False)
            outputs.extend(identity(p) for p in files+[manifest, raw]+grouped_paths+sidecars)
            results[model] = list(map(str, files))
            del fitted
        if not results:
            raise ValueError('GLMsingle produced no supported B/D beta files; enable file outputs')
        np.savetxt(final/'trial_list.txt', trials.trial_type.to_numpy(), fmt='%s')
        versions = {}
        for package in ('stglm','glmsingle','numpy','nibabel'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = 'unknown'
        for saved in inputs+qc_inputs:
            if identity(saved['path']) != saved:
                raise ValueError('Input changed during GLMsingle fitting: ' + saved['path'])
        atomic_json(marker, dict(status='complete', fit_scope='joint_runs', inputs=inputs, outputs=outputs,
            conditions=conditions, tr=float(tr), stim_duration=float(stim_duration),
            drop_invalid=bool(drop_invalid), minimum_end_buffer_seconds=8. if drop_invalid else None,
            excluded_event_count=len(diagnostics['excluded_events']),
            late_event_warning_window_seconds=16., late_event_count=len(diagnostics['late_events']),
            parameters={key: value for key, value in requested_parameters.items() if key != 'extra_regressors'},
            extra_regressors=extra_info,
            run_order=run_order, versions=versions,
            export_only=bool(export_only), recovery_validation=recovery_validation,
            split_trial_types=bool(split_trial_types), qc_inputs=qc_inputs,
            residual_export=residual_record,
            qc_settings=dict(fd_threshold=float(fd_threshold),gm_threshold=float(gm_threshold),
                             tsnr_regress_poly=int(tsnr_regress_poly)),
            qc_status={str(index):meta['QCStatus'] for index,meta in run_metadata.items()}, input_plan=str(plan_path) if plan_path.is_file() else None))
        return results
