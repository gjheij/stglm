"""Synthetic run/metadata tests; GLMsingle fitting is replaced by a deterministic stub."""
import importlib.util
import ast
import logging
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

    def fit_stub(self):
        class Engine:
            def __init__(inner,params):
                self.engine_params=params
            def fit(inner,designs,data,duration,tr,outputdir,figuredir):
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
        package=types.ModuleType('stglm');package.utils=types.SimpleNamespace(save_glmsingle_outputs=save)
        with patch.dict(sys.modules,{'glmsingle':parent,'glmsingle.glmsingle':engine_module,'stglm':package}):
            return io.fit_glmsingle(self.bold,self.events,2.,output_dir=self.root/'output',params={'wantfileoutputs':[1,1,1,1]})

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
