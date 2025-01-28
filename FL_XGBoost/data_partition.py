from logging import INFO
import xgboost as xgb
import pandas as pd
import numpy as np

def load_data_for_client(client_id, num_clients, test_fraction=0.2, seed=42):
    """
    Load and partition the data for a specific client.
    """
    filename = f"Bank{chr(65 + client_id)}_Clean.csv"

    try:
        bank_data = pd.read_csv(filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file {filename} for client {client_id} not found.")

    # Shuffle the data for IID partitioning
    np.random.seed(seed)
    shuffled_data = bank_data.sample(frac=1).reset_index(drop=True)

    # Split data into IID partitions
    split_size = len(shuffled_data) // num_clients
    start_idx = client_id * split_size
    end_idx = (client_id + 1) * split_size if client_id < num_clients - 1 else len(shuffled_data)
    client_data = shuffled_data.iloc[start_idx:end_idx]

    train_size = int(len(client_data) * (1 - test_fraction))
    train_data = client_data.iloc[:train_size]
    test_data = client_data.iloc[train_size:]

    X_train = train_data.drop(columns=["income"])
    y_train = train_data["income"]
    X_test = test_data.drop(columns=["income"])
    y_test = test_data["income"]

    train_dmatrix = xgb.DMatrix(X_train, label=y_train)
    valid_dmatrix = xgb.DMatrix(X_test, label=y_test)

    return train_dmatrix, valid_dmatrix, len(X_train), len(X_test)

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
