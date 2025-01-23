"""xgboost_quickstart: A Flower / XGBoost app."""

from logging import INFO
import xgboost as xgb
import pandas as pd
from flwr.common import log
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from assignment_1.flwr_xgb.utils import apply_age_skew, apply_gender_skew, apply_education_skew


def train_test_split(partition, test_fraction, seed):
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    num_train = len(partition_train)
    num_test = len(partition_test)

    return partition_train, partition_test, num_train, num_test


def transform_dataset_to_dmatrix(data):
    """Transform dataset to DMatrix format for xgboost."""
    x = data["inputs"]
    y = data["label"]
    new_data = xgb.DMatrix(x, label=y)
    return new_data


fds = None


def load_adult_data(partition_id, num_partitions):
    """Load Adult dataset and partition data."""
    df_train = pd.read_csv("/Users/annina/Documents/FH/FHStPoelten/FML/assignment1/adult_data_preprocessed/adult_train.csv")
    df_test = pd.read_csv("/Users/annina/Documents/FH/FHStPoelten/FML/assignment1/adult_data_preprocessed/adult_test.csv")

    # Apply Data Skewing
    #df_train = apply_age_skew(df_train, "age", skew_range=(-0.55, 0.83))
    #df_train = apply_gender_skew(df_train, "sex_Male", male_fraction=0.7)
    #df_train = apply_education_skew(df_train, education_columns=df_train.columns, cutoff="education_HS-grad")

    X_train, y_train = df_train.drop("income", axis=1), df_train["income"]
    X_test, y_test = df_test.drop("income", axis=1), df_test["income"]

    # Partitioning the data for clients
    partition_size = len(X_train) // num_partitions
    start = partition_id * partition_size
    end = (partition_id + 1) * partition_size if partition_id < num_partitions - 1 else len(X_train)

    X_partition, y_partition = X_train.iloc[start:end], y_train.iloc[start:end]

    train_dmatrix = xgb.DMatrix(X_partition, label=y_partition)
    valid_dmatrix = xgb.DMatrix(X_test, label=y_test)

    print(f"Skewed Train Dataset (Gender): {df_train.shape}")
    print(f"Partition (Gender): {X_partition.shape}")

    return train_dmatrix, valid_dmatrix, len(X_partition), len(X_test)



def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
