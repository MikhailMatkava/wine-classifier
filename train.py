"""Обучение простой модели классификации на встроенном датасете Wine."""
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_wine
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    data = load_wine()
    # Делим индексы, чтобы сохранить номера объектов и воспроизвести эксперимент.
    train_ids, test_ids = train_test_split(
        np.arange(len(data.target)), test_size=0.2,
        random_state=42, stratify=data.target,
    )
    x_train, x_test = data.data[train_ids], data.data[test_ids]
    y_train, y_test = data.target[train_ids], data.target[test_ids]

    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(x_train, y_train)

    # Scaler вычисляет средние и стандартные отклонения только на train.
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    results = {
        'seed': 42,
        'train_size': len(train_ids),
        'test_size': len(test_ids),
        'baseline_accuracy': accuracy_score(y_test, baseline.predict(x_test)),
        'model_accuracy': accuracy_score(y_test, predicted),
        'confusion_matrix': confusion_matrix(y_test, predicted).tolist(),
        'classification_report': classification_report(y_test, predicted, output_dict=True, zero_division=0),
        'train_indices': train_ids.tolist(),
        'test_indices': test_ids.tolist(),
        'sklearn_version': sklearn.__version__,
    }
    Path('artifacts').mkdir(exist_ok=True)
    Path('reports').mkdir(exist_ok=True)
    joblib.dump({'model': model, 'features': data.feature_names}, 'artifacts/model.joblib')
    Path('reports/metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    # Пример из test для проверки predict.py, а не новый независимый объект.
    sample = dict(zip(data.feature_names, map(float, x_test[0])))
    Path('example.json').write_text(json.dumps(sample, indent=2) + '\n')
    print(f'Train: {len(train_ids)}, test: {len(test_ids)}')
    print(f'Baseline accuracy: {results["baseline_accuracy"]:.3f}')
    print(f'Logistic regression accuracy: {results["model_accuracy"]:.3f}')
    print('Confusion matrix (rows = true, columns = predicted):')
    print(confusion_matrix(y_test, predicted))
    print(f'example.json: true class = {int(y_test[0])}')


if __name__ == '__main__':
    main()
