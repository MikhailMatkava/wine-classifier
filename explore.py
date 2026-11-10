"""Compare a simple baseline and logistic regression on several splits."""
import argparse
import json

from sklearn.datasets import load_wine
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 21, 42])
    parser.add_argument('--c', type=float, default=1.0)
    args = parser.parse_args()

    data = load_wine()
    results = []
    for seed in args.seeds:
        x_train, x_val, y_train, y_val = train_test_split(
            data.data, data.target, test_size=0.2, random_state=seed,
            stratify=data.target,
        )
        baseline = DummyClassifier(strategy='most_frequent').fit(x_train, y_train)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=args.c))
        model.fit(x_train, y_train)
        results.append({
            'seed': seed,
            'train_size': len(y_train),
            'validation_size': len(y_val),
            'baseline_accuracy': round(float(accuracy_score(y_val, baseline.predict(x_val))), 4),
            'logistic_accuracy': round(float(accuracy_score(y_val, model.predict(x_val))), 4),
        })
    print(json.dumps({'C': args.c, 'results': results}, indent=2))


if __name__ == '__main__':
    main()
