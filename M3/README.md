# Pre-Requisites
1. **Miniconda** Environment using Python v3.10 for Training, Tuning & Packaging
2. **Docker** for Containerising & Deploying the Model as Web based Flask Application

# Setup for training
## Create a Virtual Environment for training using miniconda
`conda create -n mlops python=3.10 -y`
## Activate the created environment
`conda activate mlops`
## Check for Python Version in the activated environment
`python --version` should print 3.10
## Upgrade PIP (Optional)
`pip install --upgrade pip`
## Install required training dependencies
`pip install scikit-learn optuna optuna-dashboard joblib`

# Model Training, Hyper-Parameter Tuning & Packaging
1. Dataset - **Iris** dataset from sklearn
2. Model - **RandomForestClassifier** from sklearn
3. Hyper-Parameter Tuning - **Optuna** for tuning
4. Reports - **Optuna-Dashboard** for tuning results
5. Packaging - **Joblib** for model packaging

`python train.py`

# Deployment using Flask and Docker
1. Build the **model-flask-app** Docker Image

`docker build -t model-flask-app .`

2. Run the Docker container in Port 5000

`docker run -p 5000:5000 model-flask-app`

3. Test the Flask API */predict* endpoint using **curl** or **Postman**

`curl --request POST --header "Content-Type: application/json" --data '{"features": [5.1, 3.5, 1.4, 0.2]}' http://localhost:5000/predict`