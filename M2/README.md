# Boston Housing Project: Data Versioning and ML Experiment Tracking

This project demonstrates a complete MLOps workflow involving data preparation, versioning, and machine learning experiment tracking using DVC and MLflow. The project focuses on training a Random Forest Regressor to predict housing prices using the Boston Housing dataset.

---

## Features
1. **Data Preparation**: Fetches the Boston Housing dataset from OpenML and saves it as a CSV file.
2. **Data Versioning**: Implements version control for the dataset using DVC and Git.
3. **Model Training & Experiment Tracking**: Tracks different model configurations using MLflow.
4. **Reproducibility**: Maintains reproducibility with versioned datasets and experiment logging.

---

## Technologies Used
- **Python Libraries**: `pandas`, `scikit-learn`, `mlflow`, `dvc`
- **Version Control**: Git and DVC
- **Experiment Tracking**: MLflow
- **ML Model**: Random Forest Regressor

---

## Workflow

### 1. Data Preparation
- Fetches the Boston Housing dataset from OpenML.
- Combines features and the target variable into a single CSV file.

### 2. Data Versioning with DVC
- Initializes Git and DVC repositories.
- Adds the dataset to DVC for version control.
- Commits changes to Git.

### 3. ML Experiment Tracking with MLflow
- Splits the dataset into training and testing sets.
- Trains a Random Forest Regressor with varying numbers of estimators (`n_estimators`).
- Tracks parameters, metrics, and models using MLflow.

---

## Setup Instructions

### Prerequisites
Ensure the following tools are installed:
- Python (3.8+)
- DVC
- Git
- MLflow

### Steps to Run the Project
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
2. pip install mlflow dvc scikit-learn pandas numpy matplotlib
3. python script.py
4. mlflow ui
5. For viewing the dvc log use "git log boston_housing.csv.dvc"
