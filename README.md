# MLflow Model Versioning Template

This is a minimal, GitHub-ready example project demonstrating **model versioning with MLflow**.
It includes:
- A training script that logs experiments and registers models to MLflow.
- An evaluation script to load specific model versions or stages.
- A small script to run a local MLflow tracking & registry server (SQLite backend).
- Requirements, .gitignore, and helpful usage instructions.

## Quick start (local, no server)
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate      # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

2. Run training (uses local file-based mlruns):
   ```bash
   python train.py --experiment "Regression_Example" --register-name "RegressionModel"
   ```

3. List registered model versions (if you set up an MLflow server with registry):
   Run the included `mlflow_server.sh` to start a local MLflow server with model registry enabled, then use the MLflow UI.

## Quick start (with local MLflow server & model registry)
1. Start the MLflow server (this exposes the registry UI):
   ```bash
   ./mlflow_server.sh
   ```
   This starts MLflow UI at http://127.0.0.1:5000 by default, with SQLite backend for registry and `./mlruns` as the artifact store.

2. In another terminal, run training with tracking URI pointing to the server:
   ```bash
   export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
   python train.py --experiment "Regression_Example" --register-name "RegressionModel"
   ```

3. Open the MLflow UI (http://127.0.0.1:5000) to see experiments and the model registry.

## Files included
- `train.py` — trains a simple sklearn model, logs params/metrics, and registers the model.
- `evaluate.py` — demonstrates loading a specific model version or stage.
- `mlflow_server.sh` — helper script to run mlflow server with SQLite backend.
- `requirements.txt`, `.gitignore`, `LICENSE`
- `utils.py` — shared helper functions.
- `examples/` — (optional) place to add notebooks or extra scripts.

## Notes & Best Practices
- For production model registry, use a proper relational backend and object store (e.g., PostgreSQL + S3).
- Always log the training data version (commit hash or data checksum), hyperparameters, and environment.
- Use CI to test reproducibility of registered models before promoting to `Production`.

