# Import necessary libraries
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import os
import mlflow.sklearn
import subprocess

# Ensure all necessary libraries are imported and handle missing imports
def check_imports():
    try:
        import subprocess
    except ImportError as e:
        print("Error: Missing required library.", e)
        raise

# Step 1: Data Preparation
# Fetch the Boston Housing dataset
def fetch_and_save_data():
    print("Fetching dataset...")
    data = fetch_openml(name="boston", version=1, as_frame=True)

    # Combine features and target into a single DataFrame
    df = pd.concat([data.data, data.target.rename("PRICE")], axis=1)

    # Save the dataset as a CSV file
    dataset_path = "boston_housing.csv"
    df.to_csv(dataset_path, index=False)
    print(f"Dataset saved at {dataset_path}")
    return dataset_path

# Step 2: Experiment Tracking with MLflow
def train_and_log_experiments(dataset_path):
    print("Starting MLflow experiments...")

    # Load the dataset
    df = pd.read_csv(dataset_path)
    X = df.drop("PRICE", axis=1)
    y = df["PRICE"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set up MLflow experiment
    mlflow.set_experiment("Boston Housing")

    # Experiment with different parameters
    for n_estimators in [10, 50, 100, 200]:
        with mlflow.start_run():
            # Train the model
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            input_example = X_test.head(1)

            # Log parameters, metrics, and the model
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_metric("mse", mse)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example,
                signature=mlflow.models.infer_signature(X_test, y_pred),
            )

            print(f"Logged run with alpha={n_estimators} and mse={mse}")

def setup_dvc(dataset_path):
    print("Setting up DVC...")

    try:
        # Initialize Git repository if not already initialized
        if not os.path.exists(".git"):
            subprocess.run(["git", "init"], check=True)
            print("Initialized Git repository.")

        # Initialize DVC if not already initialized
        if not os.path.exists(".dvc"):
            subprocess.run(["dvc", "init"], check=True)
            print("Initialized DVC.")

        # Configure Git user information (if not already set globally)
        subprocess.run(["git", "config", "--global", "user.name", "Aksharakishore"], check=False)
        subprocess.run(["git", "config", "--global", "user.email", "aksharabalamurugan@gmail.com"], check=False)

        # Add the dataset to DVC
        subprocess.run(["dvc", "add", dataset_path], check=True)
        print(f"Dataset '{dataset_path}' added to DVC.")

        # Add changes to Git
        subprocess.run(["git", "add", ".gitignore"], check=True)
        subprocess.run(["git", "add", f"{dataset_path}.dvc"], check=True)

        # Check for changes before committing
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if result.stdout.strip():
            subprocess.run(["git", "commit", "-m", "Add dataset with DVC"], check=True)
            print("Changes committed to Git.")
        else:
            print("No changes to commit.")

        print("Dataset versioned with DVC successfully.")

    except subprocess.CalledProcessError as e:
        print(f"Error during DVC setup: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr.decode()}")
        if e.stdout:
            print(f"Command output: {e.stdout.decode()}")

# Step 4: Main function
def main():
    # Fetch and save the dataset
    dataset_path = fetch_and_save_data()

    # Set up DVC for data versioning
    setup_dvc(dataset_path)

    # Train and log experiments using MLflow
    train_and_log_experiments(dataset_path)

if __name__ == "__main__":
    main()