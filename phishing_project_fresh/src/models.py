import os
import time
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils import print_info, print_success, print_warning

def create_models() -> dict:
    """
    Instantiates and returns the four machine learning classifiers.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=42, 
            max_iter=1000,
            C=1.0,
            solver='lbfgs'
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            max_depth=8,
            criterion='gini'
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=10,
            n_jobs=-1
        ),
        "Support Vector Machine (SVM)": SVC(
            kernel='rbf',
            probability=True,  # Enables probability estimation for single URL prediction confidence
            random_state=42,
            C=1.0
        )
    }
    return models

def train_and_evaluate_all(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> tuple:
    """
    Trains all four models on X_train, evaluates them on X_test,
    and returns a trained models dictionary and a list of evaluation reports.
    """
    models = create_models()
    trained_models = {}
    reports = []
    
    for name, model in models.items():
        print_info(f"Training {name}...")
        
        # 1. Measure Training Time
        start_train = time.perf_counter()
        model.fit(X_train, y_train)
        end_train = time.perf_counter()
        train_time = end_train - start_train
        
        # 2. Measure Inference Time
        start_infer = time.perf_counter()
        y_pred = model.predict(X_test)
        end_infer = time.perf_counter()
        infer_time = end_infer - start_infer
        
        # 3. Calculate Academic Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        trained_models[name] = model
        
        reports.append({
            "Algorithm": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Train Time (s)": train_time,
            "Inference Time (s)": infer_time
        })
        
        print_success(f"{name} trained in {train_time:.4f}s. Accuracy: {accuracy*100:.2f}%.")
        
    return trained_models, reports

def save_models_and_scaler(trained_models: dict, scaler, models_dir: str = "models") -> bool:
    """
    Saves trained models and the scaler to the disk.
    """
    try:
        os.makedirs(models_dir, exist_ok=True)
        
        # Filename mappings
        file_mappings = {
            "Logistic Regression": "lr_model.joblib",
            "Decision Tree": "dt_model.joblib",
            "Random Forest": "rf_model.joblib",
            "Support Vector Machine (SVM)": "svm_model.joblib"
        }
        
        # Save each model
        for name, model in trained_models.items():
            filename = file_mappings.get(name)
            if filename:
                filepath = os.path.join(models_dir, filename)
                joblib.dump(model, filepath)
                
        # Save scaler
        scaler_path = os.path.join(models_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        
        print_success(f"All 4 models and the scaler were successfully serialized to '{models_dir}/'.")
        return True
    except Exception as e:
        print_warning(f"Error occurred while saving models: {e}")
        return False

def load_trained_models(models_dir: str = "models") -> tuple:
    """
    Loads saved models and the scaler from disk.
    Returns a dictionary of models and the scaler.
    Returns (None, None) if not found.
    """
    file_mappings = {
        "Logistic Regression": "lr_model.joblib",
        "Decision Tree": "dt_model.joblib",
        "Random Forest": "rf_model.joblib",
        "Support Vector Machine (SVM)": "svm_model.joblib"
    }
    
    loaded_models = {}
    
    try:
        scaler_path = os.path.join(models_dir, "scaler.joblib")
        if not os.path.exists(scaler_path):
            return None, None
            
        scaler = joblib.load(scaler_path)
        
        for name, filename in file_mappings.items():
            filepath = os.path.join(models_dir, filename)
            if not os.path.exists(filepath):
                return None, None
            loaded_models[name] = joblib.load(filepath)
            
        return loaded_models, scaler
    except Exception as e:
        print_warning(f"Failed to load trained models from disk: {e}")
        return None, None
