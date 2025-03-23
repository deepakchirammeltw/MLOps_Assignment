import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from mlflow.models.signature import infer_signature
from alibi_detect.cd import KSDrift
import numpy as np
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
print("MLflow Tracking URI:", mlflow.get_tracking_uri())

# Load Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Preprocess the data
# Normalize pixel values to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape data to 2D (flatten images)
X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)

# Initialize drift detector
drift_detector = KSDrift(X_train, p_val=0.05)

# Simulate multiple runs
for run in range(5):  # Run 5 times
    with mlflow.start_run():
        print(f"Run {run + 1}")

        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)

        # Train a model
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42 + run)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Infer model signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Log the model with signature and input example
        mlflow.sklearn.log_model(
            model, 
            "model", 
            signature=signature,
            input_example=X_train[:1]
        )

        # Detect drift
        drift_preds = drift_detector.predict(X_test)
        mlflow.log_metric("drift_detected", int(drift_preds['data']['is_drift']))

        # Get the current run ID
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")

        # Download artifacts for the current run
        artifact_path = f"artifacts_run_{run + 1}"
        os.makedirs(artifact_path, exist_ok=True)  # Create directory for artifacts
        mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=artifact_path)
        print(f"Artifacts for Run {run + 1} downloaded to {artifact_path}")