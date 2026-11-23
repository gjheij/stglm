import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location('save_npy_test', Path(__file__).resolve().parents[1] / 'glmsingle/utils/save_npy.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SaveNPY(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'TYPEB.npy'

    def test_dictionary_uses_protocol_four_and_roundtrips(self):
        values = {'betasmd': np.arange(24, dtype=np.float32).reshape(2, 3, 4), 'R2': np.ones(6)}
        original = module.pickle.dump
        with patch.object(module.pickle, 'dump', wraps=original) as dump:
            module.save_npy(self.path, values)
        self.assertEqual(dump.call_args.kwargs['protocol'], 4)
        recovered = np.load(self.path, allow_pickle=True).item()
        for key in values:
            np.testing.assert_array_equal(recovered[key], values[key])

    def test_numeric_array_and_automatic_extension(self):
        module.save_npy(self.path.with_suffix(''), np.arange(12).reshape(3, 4))
        np.testing.assert_array_equal(np.load(self.path), np.arange(12).reshape(3, 4))

    def test_failed_save_preserves_previous_file_and_removes_temp(self):
        self.path.write_bytes(b'previous valid file')
        with patch.object(module.pickle, 'dump', side_effect=OverflowError('simulated serialization error')):
            with self.assertRaises(OverflowError):
                module.save_npy(self.path, {'data': np.ones(5)})
        self.assertEqual(self.path.read_bytes(), b'previous valid file')
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_object_arrays_require_pickle(self):
        with self.assertRaises(ValueError):
            module.save_npy(self.path, {'a': 1}, allow_pickle=False)
        self.assertFalse(self.path.exists())


if __name__ == '__main__':
    unittest.main()
