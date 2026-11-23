import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / 'stglm/cli.py').read_text())
names = {'_run_number', '_entity_match', '_find_bids_match', 'build_inputs'}
namespace = dict(np=np, logger=logging.getLogger('run_selection_test'))
exec(compile(ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
                        type_ignores=[]), 'cli-input-discovery', 'exec'), namespace)


class Layout:
    def __init__(self, missing_events=(2,)):
        self.missing_events = missing_events
        self.calls = []

    def get(self, **kwargs):
        self.calls.append(kwargs)
        suffix = kwargs['suffix']
        if suffix == 'probseg':
            return [] if 'run' in kwargs else [SimpleNamespace(path='/GM.nii.gz', entities={'subject': '29', 'space': 'T1w'})]
        runs = kwargs.get('run', [1, 2, 3, 4])
        if suffix == 'events':
            runs = [run for run in runs if run not in self.missing_events]
        return [SimpleNamespace(path=f'/{suffix}_run-{run:02d}', entities=dict(subject='29', task='test', run=run,
                    **({'space': 'T1w'} if suffix in ('bold', 'mask') else {}))) for run in runs]

    def get_metadata(self, path):
        if path == '/bold_run-02':
            raise AssertionError('Excluded run must not have its metadata read')
        return {'RepetitionTime': 2.1}


class RunSelection(unittest.TestCase):
    def build(self, layout, **kwargs):
        return namespace['build_inputs'](layout, '29', glm_single=True, task='test', space='T1w', **kwargs)

    def test_explicit_selection_precedes_pairing_and_qc(self):
        layout = Layout()
        result = self.build(layout, runs=['01', 'run-03', 4, '01'])
        self.assertEqual(layout.calls[0]['run'], [1, 3, 4])
        self.assertEqual(result['bold_files'], ['/bold_run-01', '/bold_run-03', '/bold_run-04'])
        self.assertEqual(result['event_files'], ['/events_run-01', '/events_run-03', '/events_run-04'])
        self.assertEqual([record['bold'] for record in result['qc_runs']], result['bold_files'])

    def test_default_skips_missing_events_with_warning(self):
        with self.assertLogs('run_selection_test', level='WARNING') as records:
            result = self.build(Layout())
        self.assertEqual(len(result['bold_files']), 3)
        self.assertTrue(any('/bold_run-02' in message for message in records.output))
        self.assertTrue(any('retained 3/4' in message for message in records.output))

    def test_all_missing_events_fail(self):
        with self.assertRaisesRegex(FileNotFoundError, 'No BOLD runs with matching events'):
            self.build(Layout(missing_events=(1, 2, 3, 4)))

    def test_requested_missing_bold_fails(self):
        layout = Layout()
        original = layout.get
        layout.get = lambda **kwargs: [item for item in original(**kwargs) if item.entities.get('run') != 4]
        with self.assertRaisesRegex(FileNotFoundError, 'Requested BOLD runs missing'):
            self.build(layout, runs=[1, 4])

    def test_ambiguous_events_still_fail(self):
        layout = Layout(missing_events=())
        original = layout.get
        def get(**kwargs):
            result = original(**kwargs)
            return result + [SimpleNamespace(path='/duplicate_events', entities=result[0].entities)] if kwargs['suffix'] == 'events' else result
        layout.get = get
        with self.assertRaisesRegex(ValueError, 'Ambiguous events'):
            self.build(layout)

    def test_reject_invalid_run_labels(self):
        for value in ('0', '-1', 'abc', '1.5'):
            with self.assertRaises(ValueError):
                namespace['_run_number'](value)

    def test_standard_loop_keeps_events_and_confounds_aligned(self):
        result = namespace['build_inputs'](Layout(), '29', task='test', space='T1w', evs=['US'])
        self.assertEqual([record['bold'] for record in result], ['/bold_run-01', '/bold_run-03', '/bold_run-04'])
        self.assertEqual([record['events'] for record in result], ['/events_run-01', '/events_run-03', '/events_run-04'])
        self.assertEqual([record['confounds'] for record in result], ['/timeseries_run-01', '/timeseries_run-03', '/timeseries_run-04'])

    def test_standard_concat_does_not_merge_excluded_run(self):
        merged = {}
        def merge(kind, values):
            merged[kind] = values
            return '/merged_' + kind
        namespace['utils'] = SimpleNamespace(
            merge_bold=lambda values: merge('bold', values),
            merge_confounds=lambda values, **kwargs: merge('confounds', values),
            merge_events=lambda values, *args, **kwargs: merge('events', values))
        namespace['build_inputs'](Layout(), '29', task='test', space='T1w', evs=['US'], run_mode='concat')
        for values in merged.values():
            self.assertEqual(len(values), 3)
            self.assertFalse(any('run-02' in value for value in values))


if __name__ == '__main__':
    unittest.main()
