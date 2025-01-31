import warnings
import logging as log
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import wandb
from sklearn.metrics import log_loss, roc_auc_score, precision_score, recall_score, f1_score
from FL_XGBoost.data_partition import load_data_for_client
from flwr.client import Client, ClientApp
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Status, Code

# _______________________________________

warnings.filterwarnings("ignore", category=UserWarning)

# _______________________________________

class FlowerClient(Client):
    def __init__(self, train_dmatrix, valid_dmatrix, num_train, 
                 num_val, num_local_round, params, optimal_threshold, bank_name):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params
        self.optimal_threshold = optimal_threshold
        self.bank_name = bank_name

        try:
            wandb.init(
                project="fl_data_agg_final_run",
                name=f"{bank_name}_fl_run-{wandb.util.generate_id()}",
                reinit=True,
                config=params,
            )
        except Exception as e:
            log.warning(f"WandB couldn't be intialized: {e}")

# _______________________________________

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        bst = xgb.train(self.params, self.train_dmatrix, num_boost_round=self.num_local_round)

        y_true = self.valid_dmatrix.get_label()
        y_pred = bst.predict(self.valid_dmatrix)

        print(f"First 10 y_true values: {y_true[:10]}")
        print(f"First 10 y_pred values: {y_pred[:10]}")

        y_pred_bin = (y_pred >= self.optimal_threshold).astype(int)

        auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else float("nan")
        precision = precision_score(y_true, y_pred_bin, zero_division=0)
        recall = recall_score(y_true, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true, y_pred_bin, zero_division=0)
        logloss = log_loss(y_true, y_pred) if len(set(y_true)) > 1 else float("nan")

        wandb.log({
            "AUC": auc, "LogLoss": logloss, "Precision": precision, 
            "Recall": recall, "F1-Score": f1, "Round": global_round
        })

        bst.save_model("latest_model.json")
        with open("latest_model.json", "rb") as f:
            model_bytes = f.read()

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[model_bytes]),
            num_examples=self.num_train,
            metrics={
                "AUC": auc, "LogLoss": logloss, "Precision": precision, 
                "Recall": recall, "F1-Score": f1, "Round": global_round, 
                "bank_name": self.bank_name,
            },
        )

# _______________________________________

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        bst = xgb.Booster(params=self.params)
        model_path = "latest_model.json" 
        if not os.path.exists(model_path):
            log.warning(f"Model {model_path} not found! Skipping evaluation.")


        if ins.parameters and len(ins.parameters.tensors) > 0:
            model_bytes = ins.parameters.tensors[0]
            with open("evaluating_model.json", "wb") as f:
                f.write(model_bytes)
            bst.load_model("evaluating_model.json")
        else:
            log.warning(f"Did not revieve any model parameters, loading {model_path} instead.")
            if not os.path.exists(model_path):
                return EvaluateRes(
                    status=Status(code=Code.ERROR, message="Model not found"),
                    loss=float("nan"), num_examples=0, metrics={}
                )
            bst.load_model(model_path)

        y_true = self.valid_dmatrix.get_label()
        y_pred = bst.predict(self.valid_dmatrix)
        y_pred_bin = (y_pred >= self.optimal_threshold).astype(int)

        if len(set(y_true)) < 2 or len(y_pred) == 0:
            log.warning(f"⚠️ Bank {self.bank_name}: No valid labels for AUC calculation.")
            return EvaluateRes(
                status=Status(code=Code.OK, message="No valid labels"),
                loss=float("nan"),
                num_examples=self.num_val,
                metrics={"AUC": float("nan"), "LogLoss": float("nan"), "Precision": float("nan"), "Recall": float("nan"), "F1-Score": float("nan")}
            )

        auc = roc_auc_score(y_true, y_pred)
        logloss = log_loss(y_true, y_pred)
        precision = precision_score(y_true, y_pred_bin, zero_division=0)
        recall = recall_score(y_true, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true, y_pred_bin, zero_division=0)

        wandb.log({
            "Validation AUC": auc,
            "Validation LogLoss": logloss,
            "Validation Precision": precision,
            "Validation Recall": recall,
            "Validation F1-Score": f1,
        })

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=logloss,
            num_examples=self.num_val,
            metrics={"AUC": auc, "LogLoss": logloss, "Precision": precision, "Recall": recall, "F1-Score": f1, "bank_name": self.bank_name},
        )

# _______________________________________

def client_fn(context):
    partition_id = context.node_config["partition-id"]
    train_dmatrix, valid_dmatrix, num_train, num_val, optimal_threshold, bank_name = load_data_for_client(partition_id)

    scale_pos_weights = {"BankA": 1.5, "BankB": 2.5, "BankC": 3.5}
    scale_pos_weight = scale_pos_weights.get(bank_name, 1.0)

    if bank_name not in scale_pos_weights:
        log.warning(f"Bank {bank_name} has no specific `scale_pos_weight`, setting default = 1.0")


    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.85,
        "min_child_weight": 4,
        "n_estimators": 200,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight
    }

    return FlowerClient(train_dmatrix, valid_dmatrix, num_train, num_val, 3, params, optimal_threshold, bank_name)

# _______________________________________


app = ClientApp(client_fn=client_fn)
