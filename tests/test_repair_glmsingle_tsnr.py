import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'stglm/repair_glmsingle_tsnr.py'
spec = importlib.util.spec_from_file_location('repair_tsnr', MODULE)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


class Repair(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stream = Path(self.tmp.name) / 'sub-01/task-test'
        qc = self.stream / 'qc'
        qc.mkdir(parents=True)
        self.old = qc / ('sub-01_run-01_space-T1w' + repair.OLD)
        self.old.write_bytes(b'unchanged image bytes')
        self.new = qc / ('sub-01_run-01_space-T1w' + repair.NEW)
        self.qc = qc / 'sub-01_run-01_space-T1w_qc.json'
        self.qc.write_text(json.dumps({'TSNRFile': str(self.old), 'Setting': {'Space': 'T1w'}}))
        self.provenance = self.stream / 'fit_provenance.json'
        self.provenance.write_text(json.dumps({'outputs': [{'path': str(self.old)}]}))

    def test_preview_apply_and_repeat_preserve_legacy_identity(self):
        original = (self.old.stat(), self.qc.read_bytes(), self.provenance.read_bytes())
        self.assertEqual(repair.repair(self.tmp.name), (1, 0))
        self.assertFalse(self.new.exists())
        self.assertEqual(repair.repair(self.tmp.name, True), (1, 1))
        self.assertEqual(self.old.read_bytes(), self.new.read_bytes())
        self.assertEqual(self.old.stat().st_mtime_ns, original[0].st_mtime_ns)
        self.assertEqual(self.qc.read_bytes(), original[1])
        self.assertEqual(self.provenance.read_bytes(), original[2])
        sidecar = self.new.with_name(self.new.name[:-7] + '.json')
        metadata = json.loads(sidecar.read_text())
        self.assertEqual(metadata['TSNRFile'], str(self.new))
        self.assertEqual(metadata['Setting']['Space'], 'T1w')
        self.assertEqual(repair.repair(self.tmp.name, True), (1, 0))

    def test_conflicting_image_is_not_overwritten(self):
        self.new.write_bytes(b'other data')
        with self.assertRaisesRegex(ValueError, 'Conflicting destination image'):
            repair.repair(self.tmp.name, True)
        self.assertEqual(self.new.read_bytes(), b'other data')

    def test_conflicting_json_is_not_overwritten(self):
        self.new.with_name(self.new.name[:-7] + '.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Conflicting destination sidecar'):
            repair.repair(self.tmp.name, True)
        self.assertFalse(self.new.exists())

    def test_active_fit_is_not_modified(self):
        with repair._fit_lock(self.stream):
            with self.assertRaises(BlockingIOError):
                repair.repair(self.tmp.name, True)
        self.assertFalse(self.new.exists())


if __name__ == '__main__':
    unittest.main()
