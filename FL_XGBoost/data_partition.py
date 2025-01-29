import logging as log
from re import split

import wandb

import xgboost as xgb

import pandas as pd

import numpy as np

from sklearn.metrics import precision_recall_curve


#_______________________________________

wandb.init()

#_______________________________________

def calculate_optimal_threshold(y_true, X_test):
    """
    Calculates the optimal threshold for each client based on the precision-recall curve.

    First a dummy model will be implemented for the threshold which will later be replaced with the real
    model. The rationale behind this is to train and test the threshold recognizing if income is >50K 
    (in the preprocessed data set: >50 K = 1).

    Furthermore, the threshold is being optimized delivering the maximum recall, 
    with regards to an acceptable Precision
    """
    model = xgb.XGBClassifier()
    model.fit(X_test, y_true)
    y_probs = model.predict_proba(X_test)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]

    return best_threshold


#_______________________________________


def load_data_for_client(client_id, num_clients, test_fraction=0.2, seed=42):
    """
    Load and partition the data for a specific client.
    """
    filename = f"Bank{chr(65 + client_id)}_Clean.csv"
    bank_name = f"Bank{chr(65 + client_id)}"

    try:
        bank_data = pd.read_csv(filename)
        wandb.log({"Client Name": bank_name})
        wandb.log({"Dataset Size": len(bank_data)})
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file {filename} for client {client_id} not found.")

    np.random.seed(seed)
    shuffled_data = bank_data.sample(frac=1).reset_index(drop=True)
    wandb.log({"Shuffled Data Size": len(shuffled_data)})

    split_size = len(shuffled_data) // num_clients
    wandb.log({"Split Size": split_size})
    start_idx = client_id * split_size
    end_idx = (client_id + 1) * split_size if client_id < num_clients - 1 else len(shuffled_data)
    client_data = shuffled_data.iloc[start_idx:end_idx]

    train_size = int(len(client_data) * (1 - test_fraction))
    wandb.log({"Train Size": train_size})
    train_data = client_data.iloc[:train_size]
    test_data = client_data.iloc[train_size:]

    X_train = train_data.drop(columns=["income"])
    y_train = train_data["income"]
    X_test = test_data.drop(columns=["income"])
    y_test = test_data["income"]

    feature_columns = list(X_train.columns)

    train_dmatrix = xgb.DMatrix(X_train, label=y_train, feature_names=feature_columns)
    valid_dmatrix = xgb.DMatrix(X_test, label=y_test, feature_names=feature_columns)

    optimal_threshold = calculate_optimal_threshold(y_test, X_test)
    wandb.log({"Optimal Threshold": optimal_threshold})

    return train_dmatrix, valid_dmatrix, len(X_train), len(X_test), optimal_threshold, bank_name

#_______________________________________


def replace_keys(input_dict, match="-", target="_"):
    """
    Recursively replace match string with target string in dictionary keys.
    """
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
