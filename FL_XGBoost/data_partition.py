import logging as log
import wandb
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, precision_score, recall_score, f1_score, log_loss
from flwr.client import Client, ClientApp
from flwr.common import FitIns, FitRes, Parameters, Status, Code

wandb.init()

# _______________________________________

def calculate_optimal_threshold(y_true, X_test):
    """
    Calculates the optimal threshold for the classification based on the Precision-Recall curve.
    The model gets initialized with a dummy classifier to test optimal threshold.
    """
    if len(y_true) == 0:
        return 0.5
    
    model = xgb.XGBClassifier()
    model.fit(X_test, y_true)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    return best_threshold * 0.95 

# _______________________________________

def load_data_for_client(client_id):
    filename = f"bank{chr(65 + client_id)}_final.csv"
    bank_name = f"bank{chr(65 + client_id)}"
    
    try:
        bank_data = pd.read_csv(filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data not found for {bank_name}.")
    
    
    for col in ["education_group", "marital_status_group", "occupation_group", "relationship_group", "workclass_group"]:
        if col in bank_data.columns:
            bank_data[col] = bank_data[col].astype("category")
            bank_data[col] = bank_data[col].cat.codes


    np.random.seed(42)
    shuffled_data = bank_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    test_fraction = 0.2
    train_size = int(len(shuffled_data) * (1 - test_fraction))
    
    train_data = shuffled_data.iloc[:train_size]
    test_data = shuffled_data.iloc[train_size:]
    
    X_train, y_train = train_data.drop(columns=["income"]), train_data["income"]
    X_test, y_test = test_data.drop(columns=["income"]), test_data["income"]
    
    optimal_threshold = calculate_optimal_threshold(y_test, X_test)
    
    return xgb.DMatrix(X_train, label=y_train, enable_categorical=True), \
           xgb.DMatrix(X_test, label=y_test, enable_categorical=True), \
           len(X_train), len(X_test), optimal_threshold, bank_name


# _______________________________________

def replace_keys(input_dict, match="-", target="_"):
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
