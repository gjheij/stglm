import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'residual_test_stglm'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / 'stglm')]
sys.modules[PACKAGE] = package
for name in ('glmsingle_io', 'glmsingle_residuals'):
    spec = importlib.util.spec_from_file_location(PACKAGE + '.' + name, ROOT / 'stglm' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
res = module


class Residuals(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.shape = (2, 2, 2)
        self.voxels = 8
        self.rng = np.random.default_rng(23)
        self.hrfs = np.array([[0, 0], [1, .4], [.6, 1], [.1, .3]])
        self.indices = np.arange(8) % 2
        self.beta = self.rng.normal(size=(8, 6)) * 2
        self.scale = np.linspace(.7, 1.4, 8)
        self.offset = np.linspace(-.5, .5, 8)
        self.designs = []
        self.data = []
        self.extra = []
        self.pcs = []
        self.bold = []
        self.polys = []
        self.errors = []
        self.affine = np.diag([2., 2., 2., 1.])
        for run, n_time in enumerate((36, 41)):
            design = np.zeros((n_time, 6))
            for trial, onset in enumerate((3, 11, 21)):
                design[onset, run*3+trial] = 1
            time = np.linspace(-1, 1, n_time)
            poly = np.column_stack([np.ones(n_time), time, time**2])
            extra = np.sin(np.arange(n_time)*.8)[:, None]
            pcs = np.column_stack([np.cos(np.arange(n_time)*1.2), np.sin(np.arange(n_time)*1.7)])
            prediction = np.zeros((n_time, 8))
            for voxel, hrf in enumerate(self.indices):
                task = np.column_stack([np.convolve(design[:, trial], self.hrfs[:, hrf])[:n_time] for trial in range(6)])
                prediction[:, voxel] = task @ self.beta[voxel]
            noise = self.rng.normal(scale=.3, size=prediction.shape)
            nuisance = poly @ self.rng.normal(size=(3, 8)) + 100
            nuisance += extra @ self.rng.normal(size=(1, 8))*3 + pcs[:, :1] @ self.rng.normal(size=(1, 8))*2
            data = (prediction + nuisance + noise).astype('float32')
            data[:, 0] = 0  # genuine masked background
            self.beta[0] = 0
            self.designs.append(design)
            self.data.append(data)
            self.errors.append(data.astype(float)-prediction)
            self.polys.append(poly)
            self.extra.append(extra)
            self.pcs.append(pcs)
            path = self.root / f'sub-01_task-test_run-{run+1:02d}_space-T1w_desc-masked_bold.nii.gz'
            nib.save(nib.Nifti1Image(data.T.reshape(self.shape + (n_time,)), self.affine), path)
            self.bold.append(path)
        self.means = np.concatenate(self.data).mean(axis=0)
        self.means[0] = 0
        saved = np.zeros_like(self.beta)
        saved[1:] = (self.beta[1:]*self.scale[1:, None] + self.offset[1:, None])*100/self.means[1:, None]
        r2 = []
        for data, error, poly in zip(self.data, self.errors, self.polys):
            projection = np.eye(len(data)) - poly @ np.linalg.pinv(poly)
            den = np.square(projection @ data).sum(axis=0)
            value = np.full(8, np.nan)
            value[1:] = 100*(1-np.square(projection @ error).sum(axis=0)[1:]/den[1:])
            r2.append(value)
        self.fit = dict(betasmd=saved.reshape(self.shape+(6,)), meanvol=self.means.reshape(self.shape),
                        HRFindex=self.indices.reshape(self.shape), scaleoffset=np.column_stack([self.scale, self.offset]),
                        pcnum=1, pcregressors=self.pcs, R2run=np.stack(r2, axis=-1).reshape(self.shape+(2,)))
        self.info = dict(tr=2.1, params=dict(hrflibrary=self.hrfs, maxpolydeg=[2, 2], extra_regressors=self.extra),
                         designSINGLE=self.designs)

    def test_both_definitions_match_direct_projection_and_units(self):
        for units in ('input', 'beta'):
            with patch.object(nib.Nifti1Image, 'get_fdata', side_effect=AssertionError('full 4D BOLD load')):
                files = res.export_type_d_residuals(self.fit, self.info, self.bold, self.root/units,
                                                    units=units, voxel_chunk=3, r2_atol=.001)
            self.assertEqual(len(files), 10)
            for run, (error, poly) in enumerate(zip(self.errors, self.polys)):
                for kind in ('taskpoly', 'fullnuisance'):
                    design = poly if kind == 'taskpoly' else np.column_stack([poly, self.extra[run], self.pcs[run][:, :1]])
                    expected = error - design @ np.linalg.lstsq(design, error, rcond=None)[0]
                    expected[:, 0] = 0
                    if units == 'beta':
                        expected[:, 1:] *= 100*self.scale[1:]/self.means[1:]
                    path = next(path for path in files if path.name.endswith(f'desc-{kind}_residuals.nii.gz')
                                and f'run-{run+1:02d}' in path.name)
                    image = nib.load(path)
                    np.testing.assert_allclose(image.get_fdata().reshape(8, -1).T, expected, atol=2e-5)
                    np.testing.assert_array_equal(image.affine, self.affine)
                    self.assertEqual(image.header.get_data_dtype(), np.dtype('float32'))
                    metadata = json.loads(path.with_name(path.name[:-7]+'.json').read_text())
                    self.assertIsNone(metadata['ResidualDegreesOfFreedom'])
                    self.assertEqual(metadata['ValidVoxels'], 7)
                    self.assertLess(metadata['R2Validation']['MaxAbsoluteErrorPercentagePoints'], .001)
                    self.assertEqual(metadata['NoisePCCount'], 1)
            self.assertFalse(list((self.root/units).rglob('.reconstruct-*')))

    def test_r2_mismatch_fails_before_publishing_run(self):
        self.fit['R2run'][1, 0, 0, 0] += 10
        with self.assertRaisesRegex(ValueError, 'R2run reconstruction failed'):
            res.export_type_d_residuals(self.fit, self.info, self.bold, self.root/'failed')
        self.assertFalse(list((self.root/'failed').rglob('*.nii.gz')))

    def test_invalid_indices_and_missing_pcs_fail(self):
        self.fit['HRFindex'][1, 1, 1] = 2
        with self.assertRaisesRegex(ValueError, 'zero-based library'):
            res.export_type_d_residuals(self.fit, self.info, self.bold, self.root/'bad')
        self.fit['HRFindex'][1, 1, 1] = 1
        self.fit.pop('pcregressors')
        with self.assertRaisesRegex(ValueError, 'noise PCs missing'):
            res.export_type_d_residuals(self.fit, self.info, self.bold, self.root/'bad')

    def test_unidentifiable_scale_is_explicitly_masked(self):
        self.fit['scaleoffset'][1, 0] = 0
        files = res.export_type_d_residuals(self.fit, self.info, self.bold, self.root/'masked')
        masks = [path for path in files if path.name.endswith('_mask.nii.gz')]
        self.assertEqual(len(masks), 2)
        self.assertTrue(all(nib.load(path).get_fdata().sum() == 6 for path in masks))

    def test_no_autoscale_and_no_extra_nuisance(self):
        self.fit['scaleoffset'] = np.array([])
        saved = np.zeros_like(self.beta)
        saved[1:] = self.beta[1:] * 100/self.means[1:, None]
        self.fit['betasmd'] = saved.reshape(self.shape+(6,))
        self.fit['pcnum'] = 0
        self.info['params']['extra_regressors'] = [False]
        files = res.export_type_d_residuals(self.fit, self.info, self.bold, self.root/'simple', units='input')
        images = [path for path in files if path.name.endswith('_residuals.nii.gz')]
        for run in ('01', '02'):
            pair = [nib.load(path).get_fdata() for path in images if f'run-{run}' in path.name]
            np.testing.assert_allclose(pair[0], pair[1], atol=1e-6)

    def test_bold_buffer_is_contiguous_by_volume(self):
        original = np.memmap
        buffers = []
        def memmap(path, **kwargs):
            value = original(path, **kwargs)
            if Path(path).name == 'bold.dat':
                self.assertTrue(value[:, 0].flags.c_contiguous)
                buffers.append(kwargs.get('order'))
            return value
        with patch.object(res.np, 'memmap', side_effect=memmap):
            res.export_type_d_residuals(self.fit, self.info, self.bold, self.root/'order')
        self.assertEqual(buffers, ['F', 'F'])

    def test_dependent_nuisance_columns_use_rank(self):
        self.info['params']['extra_regressors'] = [np.column_stack([value, value, np.ones(len(value))]) for value in self.extra]
        files = res.export_type_d_residuals(self.fit, self.info, self.bold, self.root/'dependent')
        metadata = json.loads(next(path for path in files if path.suffix == '.json').read_text())
        self.assertEqual(metadata['NuisanceRank'], 5)
        self.assertEqual(metadata['ExtraRegressorCount'], 3)


if __name__ == '__main__':
    unittest.main()
