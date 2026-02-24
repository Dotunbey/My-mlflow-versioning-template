"""Train and register a simple regression model with MLflow.

This script:
 - Generates synthetic regression data
 - Trains a scikit-learn LinearRegression model
 - Logs params, metrics, artifacts to MLflow
 - Optionally registers the model under a given registered model name

Usage step:
    python train.py --experiment "MyExperiment" --register-name "RegressionModel"
"""
import argparse
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import os
from utils import set_tracking_uri_from_env

def train_and_log(experiment_name, register_name=None, random_state=42, test_size=0.2):
    tracking_uri = set_tracking_uri_from_env()
    print(f"Using MLflow tracking URI: {tracking_uri}")
    mlflow.set_experiment(experiment_name)

    X, y = make_regression(n_samples=1000, n_features=10, noise=5.0, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)

        # Log params & metrics
        mlflow.log_param('model', 'LinearRegression')
        mlflow.log_param('n_features', X.shape[1])
        mlflow.log_param('random_state', random_state)
        mlflow.log_metric('r2', float(r2))
        mlflow.log_metric('mse', float(mse))

        # Log model (artifact) and register if requested
        artifact_path = 'sklearn-regression'
        if register_name:
            mlflow.sklearn.log_model(model, artifact_path=artifact_path, registered_model_name=register_name)
            print(f"Logged and registered model under name: {register_name}")
        else:
            mlflow.sklearn.log_model(model, artifact_path=artifact_path)
            print("Logged model as an artifact (not registered).")

        # Save a small numpy example artifact (for reproducibility)
        np.save('example_input.npy', X_test[:5])
        mlflow.log_artifact('example_input.npy')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='Default', help='MLflow experiment name')
    parser.add_argument('--register-name', type=str, default=None, help='Registered model name (optional)')
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()
    train_and_log(args.experiment, args.register_name, random_state=args.random_state)
