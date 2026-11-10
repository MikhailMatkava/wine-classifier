"""Проверяем ошибки ввода и независимость от порядка JSON-полей."""
import json
import os
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_wine
from predict import read_sample
from train import main as train_project


class InputTests(unittest.TestCase):
    def read(self, values):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sample.json'
            path.write_text(json.dumps(values))
            return read_sample(path, ['alcohol', 'ash'])

    def test_reorders_features(self):
        np.testing.assert_array_equal(self.read({'ash': 2.5, 'alcohol': 13}), [[13, 2.5]])

    def test_rejects_incorrect_fields(self):
        for values in [{'alcohol': 13}, {'alcohol': 13, 'ash': 2, 'extra': 0}]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.read(values)

    def test_rejects_bad_values(self):
        for value in ['13', True, None, float('nan'), float('inf')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.read({'alcohol': value, 'ash': 2})

    def test_rejects_non_object_json(self):
        with self.assertRaises(ValueError):
            self.read([13, 2.5])

    def test_accepts_finite_int_and_float_values(self):
        row = self.read({'alcohol': 0, 'ash': -2.5})
        self.assertEqual(row.shape, (1, 2))
        np.testing.assert_array_equal(row, [[0, -2.5]])

    def test_training_outputs_match_saved_model_and_split(self):
        with tempfile.TemporaryDirectory() as folder:
            previous = Path.cwd()
            try:
                os.chdir(folder)
                train_project()
                report = json.loads(Path('reports/metrics.json').read_text())
                saved = joblib.load('artifacts/model.joblib')
                example = read_sample('example.json', saved['features'])
            finally:
                os.chdir(previous)
        self.assertEqual(set(report['train_indices']) & set(report['test_indices']), set())
        self.assertEqual(report['train_size'] + report['test_size'], len(load_wine().target))
        self.assertEqual(sum(map(sum, report['confusion_matrix'])), report['test_size'])
        self.assertEqual(example.shape, (1, len(saved['features'])))
        self.assertIn(int(saved['model'].predict(example)[0]), range(3))


if __name__ == '__main__':
    unittest.main()
