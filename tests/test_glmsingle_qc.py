"""QC parity with direct Nipype-style regression and bounded-volume IO."""
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
from numpy.polynomial.legendre import Legendre
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
package=types.ModuleType('qc_test_stglm');package.__path__=[str(ROOT/'stglm')]
with patch.dict(sys.modules,{'qc_test_stglm':package}):
    for name in ['qc','glmsingle_io','glmsingle_qc']:
        spec=importlib.util.spec_from_file_location('qc_test_stglm.'+name,ROOT/'stglm'/(name+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        if name=='glmsingle_qc':q=module
        if name=='glmsingle_io':io=module
# Preserve the lightweight test package for the module's relative imports.
sys.modules['qc_test_stglm']=package
sys.modules['qc_test_stglm.glmsingle_qc']=q
sys.modules['qc_test_stglm.glmsingle_io']=io
spec=importlib.util.spec_from_file_location('qc_test_stglm.qc',ROOT/'stglm/qc.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared);sys.modules[spec.name]=shared


class QC(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.folder=self.root/'export';self.folder.mkdir()
        self.bold=self.root/'sub-01_task-test_run-01_space-T1w_desc-masked_bold.nii.gz'
        rng=np.random.default_rng(12);self.n=30
        time=np.linspace(-1,1,self.n)
        self.data=(1000+12*time+8*time**2+rng.normal(0,2,(3,3,2,self.n))).astype(np.float32)
        self.data[0,0,0]=0
        nib.save(nib.Nifti1Image(self.data,np.eye(4)),self.bold)
        self.events=self.root/'events.tsv';self.events.write_text('onset\tduration\ttrial_type\n0\t0\tCS\n')
        self.fd=self.root/'confounds.tsv'
        pd.DataFrame({'framewise_displacement':[np.nan]+[.1]*14+[.4]*15}).to_csv(self.fd,sep='\t',index=False)
        self.mask=self.root/'mask.nii.gz';self.gm=self.root/'gm.nii.gz'
        nib.save(nib.Nifti1Image(np.ones((3,3,2),dtype=np.uint8),np.eye(4)),self.mask)
        nib.save(nib.Nifti1Image(np.ones((3,3,2),dtype=np.float32),np.eye(4)),self.gm)
        self.tags=dict(sub='01',run='01',task='test',space='T1w',desc='masked')
        self.pairs=[(str(self.bold),str(self.events),self.tags)]
        self.run=dict(bold=str(self.bold),tr=2.,mask=str(self.mask),gm_probseg=str(self.gm),
                      fd_confounds=str(self.fd),metadata={'TaskName':'test','SkullStripped':True})

    def build(self,**kwargs):
        return q.build_run_metadata(self.pairs,[self.run],self.folder,**kwargs)

    def test_streamed_tsnr_matches_full_nipype_formula(self):
        for degree in [0,1,2]:
            design=np.column_stack([np.ones(self.n)]+[Legendre.basis(p)(np.linspace(-1,1,self.n)) for p in range(1,degree+1)])
            flat=self.data.reshape(-1,self.n)
            coefficients=np.linalg.pinv(design).dot(flat.T)
            residual=flat-design[:,1:].dot(coefficients[1:]).T
            mean=residual.mean(axis=1);std=residual.std(axis=1)
            expected=np.zeros_like(mean);valid=std>1.e-3;expected[valid]=mean[valid]/std[valid]
            path=self.root/f'tsnr{degree}.nii.gz'
            # Loading a full 4D array is forbidden in the streaming implementation.
            with patch.object(nib.Nifti1Image,'get_fdata',side_effect=AssertionError('full 4D load')):
                q.streaming_tsnr(self.bold,path,degree)
            actual=nib.load(path).get_fdata().ravel()
            np.testing.assert_allclose(actual,expected,rtol=2.e-6,atol=1.e-5)

    def test_metadata_metrics_match_shared_fd_and_gm_helpers(self):
        records,outputs,inputs=self.build()
        base=records[0]
        expected=shared.update_vals_with_fd({},self.fd,.2)
        self.assertAlmostEqual(base['FDMean'],expected['fd_mean'])
        self.assertAlmostEqual(base['FDMax'],expected['fd_max'])
        self.assertAlmostEqual(base['FDPerc'],expected['fd_perc'])
        gm=shared.update_vals_with_mean_gm_tsnr({},base['TSNRFile'],str(self.gm),str(self.mask))
        self.assertAlmostEqual(base['MeanGMTSNR'],gm['mean_gm_tsnr'])
        self.assertEqual(base['QCStatus'],'complete')
        self.assertEqual(base['AcquisitionVolumeShape'],[3,3,2])
        self.assertTrue(base['SkullStripped'])
        self.assertEqual(base['TSNRRegressPoly'],2)
        self.assertTrue((self.folder/'qc/run_metrics.tsv').is_file())
        self.assertTrue(all(Path(path).is_file() for path in outputs))

    def test_cache_reuses_map_and_invalidates_metrics_without_recomputing_tsnr(self):
        self.build()
        with patch.object(q,'streaming_tsnr',side_effect=AssertionError('QC should be cached')):
            self.build()
            changed=self.build(fd_threshold=.5)[0][0]
        self.assertEqual(changed['FDPerc'],0.)

    def test_missing_fd_and_gm_are_explicit_not_zero(self):
        self.run['fd_confounds']=None;self.run['gm_probseg']=None
        base=self.build()[0][0]
        self.assertEqual(base['QCStatus'],'partial')
        self.assertNotIn('FDMean',base)
        self.assertNotIn('MeanGMTSNR',base)
        self.assertEqual(len(base['QCIssues']),2)

    def test_model_metadata_distinguishes_external_and_adaptive_confounds(self):
        physio=self.root/'physio.tsv'
        pd.DataFrame({'cardiac':np.arange(self.n)}).to_csv(physio,sep='\t',index=False)
        base=self.build(additional_confounds=[str(physio)])[0][0]
        self.assertEqual(base['ConfoundRegressors'],['cardiac'])
        self.assertEqual(base['Setting']['ConfoundsRemoval'],['cardiac'])
        params=dict(wantglmdenoise=1,wantfracridge=1,maxpolydeg=[2])
        typed=q.model_metadata(base,'typed',params,{'pcnum':3},0)
        typeb=q.model_metadata(base,'typeb',params,{},0)
        self.assertTrue(typed['Model']['GLMdenoise'])
        self.assertEqual(typed['Model']['GLMdenoisePCCount'],3)
        self.assertFalse(typeb['Model']['GLMdenoise'])
        self.assertEqual(typeb['Model']['GLMdenoisePCCount'],0)
        self.assertEqual(typed['Model']['BaselinePolynomialDegree'],2)
        self.assertNotIn('GLMdenoise',base['Model'])

    def test_fd_row_count_must_match_run(self):
        pd.DataFrame({'framewise_displacement':[.1]}).to_csv(self.fd,sep='\t',index=False)
        with self.assertRaisesRegex(ValueError,'row count'):self.build()

class InputDiscovery(unittest.TestCase):
    def test_glmsingle_discovers_qc_inputs_without_adding_them_as_regressors(self):
        tree=ast.parse((ROOT/'stglm/cli.py').read_text())
        nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef)
               and node.name in ('_entity_match','_find_bids_match','build_inputs')]
        ns=dict(np=np,logger=logging.getLogger(__name__))
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'build_inputs','exec'),ns)
        class Layout:
            def __init__(self):self.calls=[]
            def get(self,**kwargs):
                self.calls.append(kwargs)
                suffix=kwargs['suffix']
                if suffix=='probseg':
                    return [types.SimpleNamespace(path='/data/GM.nii.gz',entities={'subject':'01','space':'T1w'})]
                roots={'bold':'bold','events':'events','mask':'mask','timeseries':'fd'}
                if suffix not in roots:return []
                return [types.SimpleNamespace(path=f'/data/{roots[suffix]}-{run}',
                    entities={'subject':'01','task':'test','run':run,**({'space':'T1w'} if suffix in ('bold','mask') else {})})
                        for run in ['01','02']]
            def get_metadata(self,path):return {'RepetitionTime':2.}
        layout=Layout()
        result=ns['build_inputs'](layout,'01',glm_single=True,task='test',space='T1w',confounds_suffix='physio')
        self.assertEqual(len(result['qc_runs']),2)
        for run in result['qc_runs']:
            number=run['bold'].split('-')[-1]
            self.assertEqual(run['fd_confounds'],'/data/fd-'+number)
            self.assertEqual(run['gm_probseg'],'/data/GM.nii.gz')
            self.assertEqual(run['mask'],'/data/mask-'+number)
        self.assertEqual(result['params'],{})
        self.assertNotIn('confounds',result)
        self.assertTrue(any(call['suffix']=='timeseries' and call.get('desc')=='confounds' for call in layout.calls))

if __name__=='__main__':unittest.main()
