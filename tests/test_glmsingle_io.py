"""Synthetic run/metadata tests; GLMsingle fitting is replaced by a deterministic stub."""
import importlib.util
import ast
import logging
import shutil
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('glmsingle_io_under_test', ROOT/'stglm/glmsingle_io.py')
io = importlib.util.module_from_spec(spec);spec.loader.exec_module(io)


class Runs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bold, self.events = [], []
        for run, length, conditions in [('01',7,['US','CS']),('02',9,['CS'])]:
            prefix = f'sub-01_task-test_run-{run}'
            bold = self.root/(prefix+'_space-T1w_desc-masked_bold.nii.gz')
            events = self.root/(prefix+'_events.tsv')
            nib.save(nib.Nifti1Image(np.ones((2,2,2,length),dtype=np.float32), np.eye(4)),bold)
            pd.DataFrame(dict(onset=[4.,0.][:len(conditions)],duration=[0.]*len(conditions),trial_type=conditions)).to_csv(events,sep='\t',index=False)
            self.bold.append(str(bold));self.events.append(str(events))

    def prepare(self, **kwargs):
        return io.prepare_runs(self.bold[::-1], self.events, 2., **kwargs)

    def test_run_matching_common_columns_and_trial_order(self):
        pairs,data,designs,trials,conditions,_,_ = self.prepare()
        self.assertEqual([pair[2]['run'] for pair in pairs],['01','02'])
        self.assertEqual([v.shape for v in designs],[(7,2),(9,2)])
        self.assertEqual(conditions,['CS','US'])
        self.assertEqual(trials.trial_type.tolist(),['CS','US','CS'])
        self.assertEqual(trials.run.tolist(),['01','01','02'])
        self.assertEqual(trials.beta_index.tolist(),[1,2,3])
        self.assertEqual(designs[1][:,1].sum(),0)
        self.assertEqual(data[0].dtype,np.float32)

    def test_include_none_and_selected_conditions(self):
        with self.assertRaisesRegex(ValueError, 'No selected events'):
            self.prepare(include_trials=['US'])

    def test_colliding_events_are_rejected(self):
        frame=pd.read_csv(self.events[0],sep='\t');frame.onset=[.1,.2];frame.to_csv(self.events[0],sep='\t',index=False)
        with self.assertRaisesRegex(ValueError,'collide'):self.prepare()

    def test_out_of_run_events_are_not_silently_lost(self):
        frame=pd.read_csv(self.events[0],sep='\t');frame.onset=[-1.,0.];frame.to_csv(self.events[0],sep='\t',index=False)
        with self.assertRaisesRegex(ValueError,'outside'):self.prepare()

    def test_error_distinguishes_acquisition_boundary_from_rounding(self):
        frame=pd.read_csv(self.events[0],sep='\t')
        frame.onset=[13.2,14.0]  # 7 volumes * 2 s: both round to row 7.
        frame.to_csv(self.events[0],sep='\t',index=False)
        with self.assertRaises(ValueError) as caught:self.prepare()
        message=str(caught.exception)
        self.assertIn('rounding beyond final design row',message)
        self.assertIn('onset beyond acquisition',message)
        self.assertIn('Volumes: 7; TR: 2 s',message)
        self.assertIn('tsv_line',message)
        self.assertIn('No events were dropped or shifted',message)
        self.assertIn(self.bold[0],message)
        self.assertIn(self.events[0],message)

    def test_ambiguous_event_matches_are_rejected(self):
        self.events.append(self.events[0])
        with self.assertRaisesRegex(ValueError,'exactly one'):self.prepare()

    def test_mixed_duration_rules_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'one common stimulus duration'):
            self.prepare(event_durations=['CS=8','US=0'])
        self.prepare(event_durations=['CS=0','US=0'])

    def test_additional_confounds_are_run_matched_and_validated(self):
        for run,length in [('01',7),('02',9)]:
            pd.DataFrame({'physio':np.arange(length)**2}).to_csv(self.root/f'{run}.tsv',sep='\t',index=False)
        result=self.prepare(additional_confounds=[str(self.root/'{run}.tsv')])
        self.assertEqual([v.shape for v in result[-2]],[(7,1),(9,1)])
        pd.DataFrame({'physio':[0,1]}).to_csv(self.root/'01.tsv',sep='\t',index=False)
        with self.assertRaisesRegex(ValueError,'one row'):self.prepare(additional_confounds=[str(self.root/'{run}.tsv')])

    def test_dependent_confounds_are_rejected(self):
        for run,length in [('01',7),('02',9)]:
            pd.DataFrame({'x':np.arange(length),'y':np.arange(length)*2}).to_csv(self.root/f'{run}.tsv',sep='\t',index=False)
        with self.assertRaisesRegex(ValueError,'dependent'):self.prepare(additional_confounds=[str(self.root/'{run}.tsv')])

    def test_mismatched_bold_grids_are_rejected(self):
        image=nib.load(self.bold[1]);affine=np.eye(4);affine[0,0]=2
        nib.save(nib.Nifti1Image(image.get_fdata(),affine),self.bold[1])
        with self.assertRaisesRegex(ValueError,'grids differ'):self.prepare()

    def fit_stub(self, **kwargs):
        class Engine:
            def __init__(inner,params):
                self.engine_params=params
                self.engine_calls=getattr(self,'engine_calls',0)+1
            def fit(inner,designs,data,duration,tr,outputdir,figuredir):
                # Reproduce the real SDK's destructive initialization, including
                # removal of any diagnostic files written inside outputdir.
                shutil.rmtree(outputdir)
                Path(outputdir).mkdir()
                order=[int(np.flatnonzero(row)[0]) for design in designs for row in design if row.any()]
                np.save(Path(outputdir)/'DESIGNINFO.npy',dict(design=designs,tr=tr,stimdur=duration,
                    stimorder=order,numtrialrun=[int(d.sum()) for d in designs],params=self.engine_params))
                count=sum(design.sum() for design in designs)
                arrays=np.stack([np.full(data[0].shape[:3],i,dtype=np.float32) for i in range(count)],axis=-1)
                for name in ['TYPEB_FITHRF.npy','TYPED_FITHRF_GLMDENOISE_RR.npy']:
                    np.save(Path(outputdir)/name,dict(betasmd=arrays))
        def save(results,bold,outputdir,prefix):
            arrays=results['betasmd'];digits=len(str(arrays.shape[-1]))
            for i in range(arrays.shape[-1]):
                nib.save(nib.Nifti1Image(arrays[...,i],np.eye(4)),Path(outputdir)/f'{prefix}_beta-{i+1:0{digits}d}.nii.gz')
        engine_module=types.ModuleType('glmsingle.glmsingle');engine_module.GLM_single=Engine
        parent=types.ModuleType('glmsingle');parent.glmsingle=engine_module
        package=types.ModuleType('stglm');package.__path__=[str(ROOT/'stglm')];package.utils=types.SimpleNamespace(save_glmsingle_outputs=save)
        with patch.dict(sys.modules,{'glmsingle':parent,'glmsingle.glmsingle':engine_module,'stglm':package}):
            return io.fit_glmsingle(self.bold,self.events,2.,output_dir=self.root/'output',params={'wantfileoutputs':[1,1,1,1]}, **kwargs)

    def test_residual_export_reuses_completed_fit_and_stays_in_manifest(self):
        result = self.fit_stub()
        calls = self.engine_calls
        folder = Path(result['typed'][0]).parent
        stub = types.ModuleType('stglm.glmsingle_residuals')
        def export(fitted, info, bold, target, **kwargs):
            self.assertEqual(kwargs['units'], 'input')
            self.assertEqual(kwargs['voxel_chunk'], 3)
            self.assertEqual(list(bold), list(self.bold))
            path = Path(target)/'residuals/model-typed/test.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}')
            return [path]
        stub.export_type_d_residuals = export
        existing = {path: path.stat().st_mtime_ns for path in folder.rglob('*')
                    if path.is_file() and path.name != 'fit_provenance.json'}
        with patch.dict(sys.modules, {'stglm.glmsingle_residuals': stub}):
            original_load = np.load
            def load(path, *args, **kwargs):
                if Path(path).name == 'TYPEB_FITHRF.npy':
                    raise AssertionError('Residual-only recovery must not load TYPE-B')
                return original_load(path, *args, **kwargs)
            with patch.object(io, 'export_conditionwise', side_effect=AssertionError('Unnecessary beta re-export')):
                with patch.object(io.np, 'load', side_effect=load):
                    self.fit_stub(export_only=True, export_residuals=True, residual_units='input', residual_chunk_voxels=3)
        self.assertEqual(self.engine_calls, calls)
        self.assertTrue(all(path.stat().st_mtime_ns == stamp for path, stamp in existing.items()))
        marker = json.loads((folder/'fit_provenance.json').read_text())
        path = folder/'residuals/model-typed/test.json'
        self.assertIn(str(path.resolve()), [item['path'] for item in marker['outputs']])
        self.fit_stub(export_only=True)
        marker = json.loads((folder/'fit_provenance.json').read_text())
        self.assertEqual(marker['residual_export']['units'], 'input')
        self.assertTrue(marker['residual_export']['enabled'])
        self.assertIn(str(path.resolve()), [item['path'] for item in marker['outputs']])

    def test_drop_invalid_excludes_only_bad_rows_and_rebuilds_conditions(self):
        frame=pd.read_csv(self.events[0],sep='\t');frame.onset=[100.,0.]
        frame.to_csv(self.events[0],sep='\t',index=False)
        result=self.prepare(drop_invalid=True,return_diagnostics=True)
        _,_,designs,trials,conditions,_,_,diagnostics=result
        self.assertEqual(conditions,['CS'])
        self.assertEqual(trials.trial_type.tolist(),['CS','CS'])
        self.assertEqual(trials.beta_index.tolist(),[1,2])
        self.assertEqual(trials.onset.tolist(),[0.,4.])
        self.assertEqual([int(design.sum()) for design in designs],[1,1])
        excluded=diagnostics['excluded_events']
        self.assertEqual(excluded.trial_type.tolist(),['US'])
        self.assertEqual(excluded.source_row.tolist(),[0])
        self.assertEqual(excluded.tsv_line.tolist(),[2])
        self.assertEqual(excluded.reason.tolist(),['onset beyond acquisition'])
        self.assertEqual(diagnostics['run_summary'].excluded_events.tolist(),[1,0])

    def test_drop_invalid_never_drops_malformed_events_or_collisions(self):
        frame=pd.read_csv(self.events[0],sep='\t');frame.onset=[.1,.2]
        frame.to_csv(self.events[0],sep='\t',index=False)
        with self.assertRaisesRegex(ValueError,'collide'):self.prepare(drop_invalid=True)
        frame.duration=[-1.,0.];frame.to_csv(self.events[0],sep='\t',index=False)
        with self.assertRaisesRegex(ValueError,'Invalid event timing'):self.prepare(drop_invalid=True)

    def test_drop_invalid_matches_glmsingle_eight_second_rule(self):
        # Seven volumes at TR=2: onset row 2 has exactly 8 s, row 3 has 6 s.
        frame=pd.DataFrame(dict(onset=[4.,5.1,12.],duration=[0.]*3,
                                trial_type=['boundary','rounded_late','last_volume'],source_row=[0,1,2]))
        image=nib.load(self.bold[0])
        retained,excluded=io.validate_event_times(frame,image,self.bold[0],self.events[0],2.,True)
        self.assertEqual(retained.trial_type.tolist(),['boundary'])
        self.assertEqual(excluded.trial_type.tolist(),['rounded_late','last_volume'])
        self.assertEqual(excluded.end_buffer_seconds.tolist(),[6.,0.])
        self.assertEqual(set(excluded.reason),{'less than 8 s after modeled onset'})
        for volume in retained.design_volume:
            self.assertGreaterEqual((image.shape[-1]-volume-1)*2.,8.)

    def test_eight_second_exclusion_requires_explicit_flag(self):
        frame=pd.DataFrame(dict(onset=[5.1],duration=[0.],trial_type=['late'],source_row=[0]))
        retained,excluded=io.validate_event_times(frame,nib.load(self.bold[0]),self.bold[0],self.events[0],2.,False)
        self.assertEqual(len(retained),1)
        self.assertTrue(excluded.empty)

    def test_drop_invalid_refuses_to_discard_an_entire_run(self):
        frame=pd.read_csv(self.events[0],sep='\t');frame.onset=[100.,200.]
        frame.to_csv(self.events[0],sep='\t',index=False)
        with self.assertRaisesRegex(ValueError,'No valid selected events'):self.prepare(drop_invalid=True)

    def test_drop_invalid_export_records_exclusions_and_keeps_beta_alignment(self):
        frame=pd.read_csv(self.events[0],sep='\t');frame.onset=[100.,0.]
        frame.to_csv(self.events[0],sep='\t',index=False)
        results=self.fit_stub(drop_invalid=True)
        folder=Path(results['typed'][0]).parent
        manifest=pd.read_csv(folder/'model-typed_trial_manifest.tsv',sep='\t',dtype=str)
        excluded=pd.read_csv(folder/'excluded_events.tsv',sep='\t',dtype=str)
        provenance=json.loads((folder/'fit_provenance.json').read_text())
        self.assertEqual(manifest.trial_type.tolist(),['CS','CS'])
        self.assertEqual(len(results['typed']),2)
        self.assertEqual(excluded.trial_type.tolist(),['US'])
        self.assertTrue(provenance['drop_invalid'])
        self.assertEqual(provenance['minimum_end_buffer_seconds'],8.)
        self.assertEqual(provenance['excluded_event_count'],1)
        self.assertTrue((folder/'late_events.tsv').is_file())
        self.assertTrue((folder/'run_summary.tsv').is_file())
        self.assertIn(str((folder/'excluded_events.tsv').resolve()),[item['path'] for item in provenance['outputs']])

    def test_reported_335_volume_128_event_case_keeps_118(self):
        nib.save(nib.Nifti1Image(np.ones((2,2,2,335),dtype=np.float32),np.eye(4)),self.bold[0])
        onsets=list(np.arange(118)*4.2)+[707.3591,713.3591,720.3738,726.3738,733.2999,739.2999,744.2328,750.2328,756.1507,762.1507]
        pd.DataFrame(dict(onset=onsets,duration=[0.]*128,trial_type=['CS']*118+['outside']*10)).to_csv(self.events[0],sep='\t',index=False)
        result=io.prepare_runs(self.bold,self.events,2.1,drop_invalid=True,return_diagnostics=True)
        self.assertEqual(int(result[2][0].sum()),118)
        self.assertEqual(len(result[-1]['excluded_events']),10)
        self.assertNotIn('outside',result[4])
        self.assertEqual(result[-1]['run_summary'].retained_events.tolist(),[118,1])

    def test_diagnostics_and_external_lock_survive_sdk_output_reset(self):
        results=self.fit_stub()
        folder=Path(results['typed'][0]).parent
        for name in ['excluded_events.tsv','late_events.tsv','run_summary.tsv','fit_provenance.json']:
            self.assertTrue((folder/name).is_file(),name)
        self.assertTrue((folder.parent/('.'+folder.name+'.fit_input.json')).is_file())
        self.assertTrue((folder.parent/('.'+folder.name+'.fit.lock')).is_file())
        self.assertFalse((folder/'.fit.lock').exists())

    def test_export_only_recovers_without_fitting_or_loading_bold(self):
        results=self.fit_stub()
        folder=Path(results['typed'][0]).parent
        (folder/'fit_provenance.json').unlink()
        (folder/'excluded_events.tsv').unlink()
        calls=self.engine_calls
        with patch.object(nib.Nifti1Image,'get_fdata',side_effect=AssertionError('export must not load BOLD arrays')):
            recovered=self.fit_stub(export_only=True)
        self.assertEqual(self.engine_calls,calls)
        self.assertEqual(recovered,results)
        self.assertTrue((folder/'excluded_events.tsv').is_file())
        saved=json.loads((folder/'fit_provenance.json').read_text())
        self.assertEqual(saved['recovery_validation'],'original_input_plan')

    def test_legacy_export_recovery_records_missing_input_history(self):
        results=self.fit_stub()
        folder=Path(results['typed'][0]).parent
        (folder/'fit_provenance.json').unlink()
        (folder.parent/('.'+folder.name+'.fit_input.json')).unlink()
        calls=self.engine_calls
        self.fit_stub(export_only=True)
        self.assertEqual(self.engine_calls,calls)
        saved=json.loads((folder/'fit_provenance.json').read_text())
        self.assertEqual(saved['recovery_validation'],'saved_design_only_legacy_input_history_unavailable')

    def qc_inputs(self):
        mask=self.root/'mask.nii.gz';gm=self.root/'gm.nii.gz'
        nib.save(nib.Nifti1Image(np.ones((2,2,2),dtype=np.uint8),np.eye(4)),mask)
        nib.save(nib.Nifti1Image(np.ones((2,2,2),dtype=np.float32),np.eye(4)),gm)
        records=[]
        for bold,length in zip(self.bold,[7,9]):
            fd=self.root/(Path(bold).name+'.fd.tsv')
            pd.DataFrame({'framewise_displacement':[np.nan]+[.3]*(length-1)}).to_csv(fd,sep='\t',index=False)
            records.append(dict(bold=bold,tr=2.,mask=str(mask),gm_probseg=str(gm),fd_confounds=str(fd),metadata={'TaskName':'test'}))
        return records

    def test_completed_fit_qc_enrichment_reuses_raw_models_and_voxel_maps(self):
        results=self.fit_stub();calls=self.engine_calls
        folder=Path(results['typed'][0]).parent
        image=Path(results['typed'][0]);before=image.stat().st_mtime_ns
        records=self.qc_inputs()
        with patch.object(nib.Nifti1Image,'get_fdata',side_effect=AssertionError('QC export must not load full BOLD arrays')):
            self.fit_stub(export_only=True,split_trial_types=True,qc_runs=records)
        self.assertEqual(self.engine_calls,calls)
        self.assertEqual(image.stat().st_mtime_ns,before)
        individual=json.loads(Path(str(image).replace('.nii.gz','.json')).read_text())
        condition_path=next((folder/'conditionwise/model-typed').glob('*.json'))
        condition=json.loads(condition_path.read_text())
        for metadata in [individual,condition]:
            self.assertEqual(metadata['QCStatus'],'complete')
            self.assertIn('MeanGMTSNR',metadata)
            self.assertEqual(metadata['FDPerc'],100.)
            self.assertEqual(metadata['ConfoundRegressors'],[])
            self.assertTrue(Path(metadata['TSNRFile']).is_file())
            self.assertTrue(metadata['Model']['GLMdenoise'])
        self.assertEqual(len(list((folder/'qc').glob('*_stat-tsnr_boldmap.nii.gz'))),2)
        self.assertTrue((folder.parent/('.'+folder.name+'.previous_export.json')).is_file())
        saved=json.loads((folder/'fit_provenance.json').read_text())
        self.assertEqual(saved['qc_status'],{'0':'complete','1':'complete'})

    def test_qc_failure_does_not_leave_a_false_success_marker(self):
        results=self.fit_stub();calls=self.engine_calls
        folder=Path(results['typed'][0]).parent
        records=self.qc_inputs()
        # Invalid QC-only rows must not force refitting or leave success behind.
        pd.DataFrame({'framewise_displacement':[.1]}).to_csv(records[0]['fd_confounds'],sep='\t',index=False)
        with self.assertRaisesRegex(ValueError,'row count'):
            self.fit_stub(export_only=True,qc_runs=records)
        self.assertFalse((folder/'fit_provenance.json').exists())
        self.assertTrue((folder/'TYPED_FITHRF_GLMDENOISE_RR.npy').exists())
        self.assertEqual(self.engine_calls,calls)
        records=self.qc_inputs()
        self.fit_stub(export_only=True,qc_runs=records)
        self.assertTrue((folder/'fit_provenance.json').exists())
        self.assertEqual(self.engine_calls,calls)

    def test_export_only_can_add_conditionwise_files_to_complete_export(self):
        results=self.fit_stub()
        folder=Path(results['typed'][0]).parent
        before=Path(results['typed'][0]).stat().st_mtime_ns
        calls=self.engine_calls
        self.fit_stub(export_only=True,split_trial_types=True)
        self.assertEqual(self.engine_calls,calls)
        self.assertEqual(Path(results['typed'][0]).stat().st_mtime_ns,before)
        self.assertTrue(list((folder/'conditionwise/model-typed').glob('*.nii.gz')))

    def test_export_recovery_rejects_changed_saved_design(self):
        results=self.fit_stub()
        folder=Path(results['typed'][0]).parent
        (folder/'fit_provenance.json').unlink()
        (folder.parent/('.'+folder.name+'.fit_input.json')).unlink()
        path=folder/'DESIGNINFO.npy';info=np.load(path,allow_pickle=True).item()
        info['design'][0][0]=0;np.save(path,info)
        with self.assertRaisesRegex(ValueError,'design/TR/duration'):self.fit_stub(export_only=True)
        self.assertFalse((folder/'fit_provenance.json').exists())

    def test_export_recovery_rejects_incomplete_raw_models(self):
        results=self.fit_stub()
        folder=Path(results['typed'][0]).parent
        (folder/'fit_provenance.json').unlink()
        (folder/'TYPED_FITHRF_GLMDENOISE_RR.npy').unlink()
        with self.assertRaisesRegex(ValueError,'raw B/D outputs are incomplete'):self.fit_stub(export_only=True)

    def test_conditionwise_exports_retain_run_and_trial_order(self):
        pd.DataFrame(dict(onset=[4.,0.,8.],duration=[0.,8.,8.],trial_type=['US','CS','CS'])).to_csv(self.events[0],sep='\t',index=False)
        results=self.fit_stub(split_trial_types=True)
        folder=Path(results['typed'][0]).parent
        paths=list((folder/'conditionwise/model-typed').glob('*.nii.gz'))
        self.assertEqual(len(paths),3)
        path=next(path for path in paths if 'run-01_' in path.name and 'condition-CS_' in path.name)
        image=nib.load(path)
        self.assertEqual(image.shape,(2,2,2,2))
        self.assertEqual(image.get_fdata()[0,0,0].tolist(),[0.,2.])
        self.assertEqual(image.header.get_zooms()[-1],1.)
        metadata=json.loads(Path(str(path).replace('.nii.gz','.json')).read_text())
        self.assertEqual(metadata['NumberOfVolumes'],7)
        self.assertEqual(metadata['FourthDimension'],'trial')
        self.assertEqual([row['BetaIndex'] for row in metadata['Events']],[1,3])
        self.assertEqual([row['duration'] for row in metadata['Events']],[0.,0.])
        self.assertEqual([row['RecordedDuration'] for row in metadata['Events']],[8.,8.])

    def test_complete_export_and_existing_fit_guard(self):
        results=self.fit_stub()
        folder=Path(results['typed'][0]).parent
        table=pd.read_csv(folder/'model-typed_trial_manifest.tsv',sep='\t',dtype=str)
        self.assertEqual(table.run.tolist(),['01','01','02'])
        self.assertEqual(table.trial_type.tolist(),['CS','US','CS'])
        self.assertEqual(len(table),len(results['typed']))
        provenance=json.loads((folder/'fit_provenance.json').read_text())
        self.assertEqual(provenance['status'],'complete')
        self.assertEqual([r['n_volumes'] for r in provenance['run_order']],[7,9])
        with self.assertRaisesRegex(ValueError,'Completed fit'):self.fit_stub()


class DesignHelper(unittest.TestCase):
    def function(self):
        tree=ast.parse((ROOT/'stglm/conditions.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='events_to_glmsingle_dm')
        ns=dict(np=np,pd=pd,logger=logging.getLogger(__name__))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'events_to_glmsingle_dm','exec'),ns)
        return ns['events_to_glmsingle_dm']

    def test_common_columns_include_missing_conditions(self):
        frame=pd.DataFrame(dict(onset=[0.],duration=[0.],trial_type=['US']))
        design=self.function()(frame,5,tr=2.,condition_names=['CS','US'])
        self.assertEqual(design.shape,(5,2))
        self.assertEqual(design[0].tolist(),[0,1])

    def test_same_condition_collision_is_rejected(self):
        frame=pd.DataFrame(dict(onset=[.1,.2],duration=[0.,0.],trial_type=['US','US']))
        with self.assertRaisesRegex(ValueError,'collide'):self.function()(frame,5,tr=2.)

    def test_beta_export_uses_geometry_without_loading_bold_data(self):
        tree=ast.parse((ROOT/'stglm/utils.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='save_glmsingle_outputs')
        import os
        ns=dict(np=np,nb=nib,os=os,logger=logging.getLogger(__name__))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'save_glmsingle_outputs','exec'),ns)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);bold=root/'bold.nii.gz'
            nib.save(nib.Nifti1Image(np.ones((2,2,2,5),dtype=np.float32),np.eye(4)),bold)
            reference=nib.load(bold)
            with patch.object(reference,'get_fdata',side_effect=AssertionError('BOLD should not be loaded')),\
                 patch.object(nib,'load',return_value=reference):
                ns['save_glmsingle_outputs']({'betasmd':np.ones((2,2,2,3),dtype=np.float32)},str(bold),str(root))
            self.assertEqual(len(list(root.glob('model_beta-*.nii.gz'))),3)

    def test_out_of_range_is_rejected(self):
        frame=pd.DataFrame(dict(onset=[20.],duration=[0.],trial_type=['US']))
        with self.assertRaisesRegex(ValueError,'outside'):self.function()(frame,5,tr=2.)

if __name__=='__main__':unittest.main()
