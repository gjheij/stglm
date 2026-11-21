"""Run-level HALFpipe-compatible QC for GLMsingle fits and export recovery."""
import hashlib
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import Legendre

from .qc import collect_base_metadata, update_vals_with_fd, update_vals_with_mean_gm_tsnr

logger = logging.getLogger(__name__)


def file_identity(path):
    path = Path(path).resolve();stat = path.stat()
    return dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def streaming_tsnr(bold_file, output, regress_poly=2):
    """Nipype TSNR: Legendre detrending without intercept removal, ddof=0.

    Two sequential passes avoid holding a full voxel-by-time array in memory.
    Welford variance avoids subtracting large nearly equal squared signals.
    """
    image = nib.load(str(bold_file),keep_file_open=True)
    if len(image.shape)!=4 or not 0 <= regress_poly < image.shape[-1]:
        raise ValueError('Invalid tSNR BOLD shape/polynomial order: ' + str(bold_file))
    spatial=image.shape[:3];n=image.shape[-1];n_voxels=int(np.prod(spatial))
    design=np.ones((n,regress_poly+1),dtype=float)
    for degree in range(regress_poly):
        design[:,degree+1]=Legendre.basis(degree+1)(np.linspace(-1,1,n))
    beta=np.zeros((regress_poly,n_voxels),dtype=float)
    if regress_poly:
        inverse=np.linalg.pinv(design)[1:]
        for time in range(n):
            volume=np.asarray(image.dataobj[...,time],dtype=np.float32).reshape(-1)
            if not np.isfinite(volume).all():
                raise ValueError('Non-finite input BOLD values during tSNR: ' + str(bold_file))
            beta += inverse[:,time,None]*volume[None,:]
    mean=np.zeros(n_voxels,dtype=float);m2=np.zeros(n_voxels,dtype=float)
    for time in range(n):
        volume=np.asarray(image.dataobj[...,time],dtype=np.float32).reshape(-1)
        if not np.isfinite(volume).all():
            raise ValueError('Non-finite input BOLD values during tSNR: ' + str(bold_file))
        residual=volume.astype(float)-design[time,1:].dot(beta)
        delta=residual-mean
        mean += delta/(time+1)
        m2 += delta*(residual-mean)
    std=np.sqrt(np.maximum(m2/n,0))
    tsnr=np.zeros(n_voxels,dtype=np.float32)
    valid=std>1.e-3
    tsnr[valid]=(mean[valid]/std[valid]).astype(np.float32)
    header=image.header.copy();header.set_data_dtype(np.float32)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    nib.save(nib.Nifti1Image(tsnr.reshape(spatial),image.affine,header),str(output))
    return str(output)


def _save_json(path,value):
    from .glmsingle_io import atomic_json
    atomic_json(Path(path),value)


def build_run_metadata(pairs, qc_runs, folder, additional_confounds=(),
                       fd_threshold=.2, gm_threshold=.5, regress_poly=2, extra_regressors=()):
    """Compute/cache metrics once per run; never reinterpret QC as fit regressors."""
    by_bold={str(Path(run['bold']).resolve()):run for run in qc_runs}
    if len(by_bold)!=len(pairs) or set(by_bold)!={str(Path(bold).resolve()) for bold,_,_ in pairs}:
        raise ValueError('QC run metadata must match exactly one record per fitted BOLD run')
    folder=Path(folder)/'qc';folder.mkdir(exist_ok=True)
    metadata={};outputs=[];all_inputs={}
    for index,(bold,events,tags) in enumerate(pairs):
        run=by_bold[str(Path(bold).resolve())]
        names=[];extra_sources=[]
        for template in additional_confounds:
            path=Path(template.format(subject=tags.get('sub',''),session=tags.get('ses',''),
                                       run=tags.get('run',''),task=tags.get('task','')))
            columns=pd.read_csv(path,sep='\t',nrows=0).columns.astype(str).tolist()
            names.extend(columns);extra_sources.append(dict(file=str(path.resolve()),columns=columns))
        generated_names = not names and bool(extra_regressors)
        if generated_names:
            names=[f'extra_regressor-{column+1:03d}' for column in range(extra_regressors[index]['shape'][1])]
        sources=[bold,events]+[run.get(key) for key in ('mask','gm_probseg','fd_confounds') if run.get(key)]
        sources += [source['file'] for source in extra_sources]
        inputs=[file_identity(path) for path in sources]
        all_inputs.update({item['path']:item for item in inputs})
        settings=dict(fd_threshold=float(fd_threshold),gm_threshold=float(gm_threshold),
                      regress_poly=int(regress_poly),tsnr_std_threshold=1.e-3,
                      algorithm='nipype-tsnr-legendre-welford-v1',numpy=np.__version__,nibabel=nib.__version__,
                      implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        upstream=dict(run.get('metadata') or {})
        for stale_key in ('FDMean','FDMax','FDPerc','MeanGMTSNR','TSNRFile','fd_mean','fd_max','fd_perc','mean_gm_tsnr'):
            upstream.pop(stale_key,None)
        request=dict(inputs=inputs,settings=settings,metadata=upstream,additional_confounds=extra_sources)
        signature=hashlib.sha256(json.dumps(request,sort_keys=True,default=str).encode()).hexdigest()
        prefix='_'.join(f'{key}-{tags[key]}' for key in ('sub','ses','task','acq','run','echo','space','desc') if tags.get(key))
        tsnr=folder/(prefix+'_stat-tsnr_statmap.nii.gz')
        sidecar=folder/(prefix+'_qc.json');cache=folder/(prefix+'_qc.provenance.json')
        tsnr_request=dict(bold=file_identity(bold),regress_poly=int(regress_poly),
                          algorithm=settings['algorithm'],numpy=np.__version__,nibabel=nib.__version__,
                          implementation=settings['implementation_sha256'])
        tsnr_signature=hashlib.sha256(json.dumps(tsnr_request,sort_keys=True).encode()).hexdigest()
        cached=None;tsnr_valid=False
        if cache.is_file():
            try:
                saved=json.loads(cache.read_text())
                tsnr_valid=(saved.get('tsnr_signature')==tsnr_signature
                            and file_identity(tsnr)==saved.get('tsnr_identity'))
                if saved['signature']==signature and all(file_identity(item['path'])==item for item in saved['outputs']):
                    cached=json.loads(sidecar.read_text())
            except (OSError,ValueError,KeyError,TypeError):
                pass
        if cached is not None:
            logger.info('Reusing run QC: %s',prefix)
            metadata[index]=cached;outputs.extend([tsnr,sidecar,cache]);continue
        logger.info('Computing run QC: %s',prefix)
        if not tsnr_valid:
            streaming_tsnr(bold,tsnr,int(regress_poly))
        image=nib.load(str(bold));issues=[]
        vals=dict(scan_start=0.,repetition_time=float(run['tr']),number_of_volumes=image.shape[-1],
                  acquisition_orientation=''.join(nib.aff2axcodes(image.affine)),
                  acquisition_volume_shape=list(image.shape[:3]),
                  acquisition_voxel_size=list(map(float,image.header.get_zooms()[:3])),
                  confound_regressors=names)
        fd=run.get('fd_confounds')
        if fd:
            if len(pd.read_csv(fd,sep='\t',usecols=['framewise_displacement']))!=image.shape[-1]:
                raise ValueError('FD confounds row count differs from BOLD: ' + str(fd))
            vals=update_vals_with_fd(vals,fd,float(fd_threshold))
        else:
            issues.append('FD metrics unavailable: no matched fMRIPrep confounds timeseries')
        if run.get('gm_probseg'):
            vals=update_vals_with_mean_gm_tsnr(vals,str(tsnr),run['gm_probseg'],run.get('mask'),float(gm_threshold))
        else:
            issues.append('MeanGMTSNR unavailable: no matched gray-matter probability map')
        raw_sources=upstream.get('RawSources',[]) or []
        if isinstance(raw_sources,str):raw_sources=[raw_sources]
        vals['raw_sources']=list(dict.fromkeys([*raw_sources,str(bold),str(events)]))
        vals['setting']=dict(upstream.get('Setting') or {})
        vals['setting']['Space']=tags.get('space') or 'native'
        vals['setting']['ConfoundsRemoval']=names
        vals['model']=dict(Estimator='GLMsingle',JointRuns=True,AdditionalConfounds=extra_sources,
                           ConfoundColumnNamesSource='generated-indices' if generated_names else 'TSV headers')
        base=collect_base_metadata(tags,vals,upstream,None,str(events),'secs',str(bold),vals['model'])
        base.update(TSNRFile=str(tsnr.resolve()),TSNRRegressPoly=int(regress_poly),
                    TSNRSource=str(Path(bold).resolve()),FDThreshold=float(fd_threshold),
                    GMThreshold=float(gm_threshold),QCStatus='partial' if issues else 'complete',QCIssues=issues,
                    QCSources={key:run.get(key) for key in ('mask','gm_probseg','fd_confounds')})
        _save_json(sidecar,base)
        for item in inputs:
            if file_identity(item['path'])!=item:
                raise ValueError('QC input changed during computation: ' + item['path'])
        _save_json(cache,dict(signature=signature,inputs=inputs,settings=settings,
                              tsnr_signature=tsnr_signature,tsnr_identity=file_identity(tsnr),
                              outputs=[file_identity(path) for path in (tsnr,sidecar)]))
        metadata[index]=base;outputs.extend([tsnr,sidecar,cache])
        for issue in issues:logger.warning('%s: %s',prefix,issue)
    metrics=folder/'run_metrics.tsv'
    records=[]
    for index,base in metadata.items():
        records.append(dict(run_index=index,subject=pairs[index][2].get('sub',''),
                            session=pairs[index][2].get('ses',''),run=pairs[index][2].get('run',''),
                            **{key:base.get(key) for key in ('FDMean','FDMax','FDPerc','MeanGMTSNR','QCStatus','TSNRFile')}))
    pd.DataFrame(records).to_csv(metrics,sep='\t',index=False);outputs.append(metrics)
    return metadata,outputs,list(all_inputs.values())


def model_metadata(base,model,parameters,fitted,run_index):
    """Describe actual GLMsingle processing, separately from external confounds."""
    import copy
    result=copy.deepcopy(base)
    info=dict(result.get('Model') or {})
    denoise=model in ('typec','typed') and bool(parameters.get('wantglmdenoise',1))
    info.update(Estimator='GLMsingle',OutputType=model,JointRuns=True,
                HRF='voxelwise-library' if parameters.get('wantlibrary',1) else 'assumed',
                GLMdenoise=denoise,FractionalRidge=model=='typed' and bool(parameters.get('wantfracridge',1)),
                LSS=model=='typeb' and bool(parameters.get('wantlss',0)))
    info['GLMdenoisePCCount']=int(np.asarray(fitted['pcnum']).item()) if denoise and 'pcnum' in fitted else (None if denoise else 0)
    degrees=parameters.get('maxpolydeg')
    if degrees is not None:
        values=np.asarray(degrees).ravel()
        info['BaselinePolynomialDegree']=int(values[int(run_index)] if values.size>1 else values[0])
    if parameters.get('ExtraRegressorsByRun'):
        info['ExtraRegressors']=parameters['ExtraRegressorsByRun'][int(run_index)]
    result['Model']=info
    return result
