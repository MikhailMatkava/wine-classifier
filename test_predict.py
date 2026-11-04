"""Проверяем ошибки ввода и независимость от порядка JSON-полей."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from predict import read_sample


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


if __name__ == '__main__':
    unittest.main()
