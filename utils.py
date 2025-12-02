import os
import mlflow

def get_tracking_uri():
    # Prefer environment variable, otherwise default to local file store
    return os.environ.get('MLFLOW_TRACKING_URI', 'file:' + os.path.abspath('./mlruns'))

def set_tracking_uri_from_env():
    uri = get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    return uri
