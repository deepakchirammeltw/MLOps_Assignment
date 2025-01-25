import joblib
import optuna
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load a dataset
data = load_iris()
X = data.data
y = data.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the objective function for Optuna
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 2, 100)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 5)
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


file_path = "./optuna_journal_storage.log"
lock_obj = optuna.storages.journal.JournalFileOpenLock(file_path)

storage = optuna.storages.JournalStorage(
    optuna.storages.journal.JournalFileBackend(file_path, lock_obj=lock_obj),
)

# Create and optimize the study
study = optuna.create_study(direction='maximize', study_name='hyper-parameter-tuning-rf', storage=storage)
study.optimize(objective, n_trials=50)

# Print the best parameters found
print(f"Best hyperparameters: {study.best_params}")
print(f"Best accuracy: {study.best_value}")

# Train the RandomForestClassifier with the best hyperparameters
best_params = study.best_params
model = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'],
                               min_samples_split=best_params['min_samples_split'], random_state=42)
model.fit(X_train, y_train)

# Package the model using joblib
joblib.dump(model, "model.joblib")

print("Random Forest Iris Model training and packaging is complete. The model is saved as 'model.joblib'.")
