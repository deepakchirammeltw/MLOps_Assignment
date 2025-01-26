import joblib
import optuna
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Optuna objective function
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 2, 100)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 5)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

# Run Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Train the best model
best_params = study.best_params
model = RandomForestClassifier(**best_params, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "model2.joblib")
print("Model training complete. Saved as 'model2.joblib'.")
