"""Предсказание для одного объекта из JSON с именами признаков."""
import argparse
import json

import joblib
import numpy as np

def read_sample(path, feature_names):
    with open(path, encoding='utf-8') as file:
        values = json.load(file)
    if not isinstance(values, dict) or set(values) != set(feature_names):
        raise ValueError('JSON должен содержать ровно те же поля, что example.json.')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values.values()):
        raise ValueError('Все показатели должны быть числами.')
    # Порядок ключей в JSON не должен менять порядок признаков модели.
    row = np.array([[values[name] for name in feature_names]], dtype=float)
    if not np.isfinite(row).all():
        raise ValueError('NaN и бесконечные значения не поддерживаются.')
    return row

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', default='example.json')
    args = parser.parse_args()
    try:
        # Загружайте только доверенные joblib-файлы: они могут выполнять код.
        saved = joblib.load('artifacts/model.joblib')
        row = read_sample(args.input, saved['features'])
        print(f'Predicted class: {int(saved["model"].predict(row)[0])}')
    except (OSError, ValueError) as error:
        parser.exit(2, f'Ошибка: {error}\n')

if __name__ == '__main__':
    main()
