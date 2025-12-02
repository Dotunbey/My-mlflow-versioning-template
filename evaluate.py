"""Evaluate or load a registered model by version or stage.

Examples:
    # Load version 1
    python evaluate.py --model-name RegressionModel --version 1

    # Load production stage
    python evaluate.py --model-name RegressionModel --stage Production
"""
import argparse
import mlflow
import mlflow.sklearn
from utils import set_tracking_uri_from_env
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
import numpy as np

def load_and_eval(model_name, version=None, stage=None):
    set_tracking_uri_from_env()
    if version:
        model_uri = f"models:/{model_name}/{version}"
    elif stage:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        raise ValueError('Either version or stage must be provided.')

    print('Loading model from:', model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    # Generate small test data
    X, y = make_regression(n_samples=100, n_features=10, noise=5.0, random_state=42)
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    print(f'Evaluated loaded model — R2: {r2:.4f}')
    return r2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--stage', type=str, default=None)
    args = parser.parse_args()
    load_and_eval(args.model_name, version=args.version, stage=args.stage)
