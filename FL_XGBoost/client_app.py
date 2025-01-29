import warnings

import logging as log

import pandas as pd

from FL_XGBoost.data_partition import load_data_for_client

from flwr.common.context import Context

from flwr.client import Client, ClientApp
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Status, Code

import xgboost as xgb

import wandb

from sklearn.metrics import log_loss, roc_auc_score, precision_score, recall_score, f1_score

#_________________________________________________

warnings.filterwarnings("ignore", category=UserWarning)


#_______________________________________

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


        wandb.init(
            project="fl_seventh_run",
            name=f"{bank_name}_fl_run-{wandb.util.generate_id()}",
            reinit=True,
            config=params,
        )


#_______________________________________

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])

        num_boost_round = self.num_local_round

        bst = xgb.train(
            self.params,
            self.train_dmatrix,
            num_boost_round=num_boost_round,
            evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
        )
        
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        eval_result = bst.eval(self.valid_dmatrix, iteration=bst.num_boosted_rounds() - 1)
        auc = float(eval_result.split(":")[1])

        y_true = self.valid_dmatrix.get_label()
        y_pred = bst.predict(self.valid_dmatrix)

        y_pred_bin = (y_pred >= self.optimal_threshold).astype(int)

        logloss = log_loss(y_true, y_pred)
        precision = precision_score(y_true, y_pred_bin)
        recall = recall_score(y_true, y_pred_bin)
        f1 = f1_score(y_true, y_pred_bin)

        wandb.log({"AUC": auc, 
                   "LogLoss": logloss,
                   "Precision": precision,
                   "Recall": recall,
                   "F1-Score": f1,
                   "Round": global_round
                   }
                )

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={"AUC": auc,
                     "LogLoss": logloss,
                     "Precision": precision,
                     "Recall": recall,
                     "F1-Score": f1,
                     "Round": global_round},
        )

#_______________________________________


    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        bst = xgb.Booster(params=self.params)
        bst.load_model(bytearray(ins.parameters.tensors[0]))

        y_true = self.valid_dmatrix.get_label()
        y_pred = bst.predict(self.valid_dmatrix)

        y_pred_bin = (y_pred >= self.optimal_threshold).astype(int)

        auc = roc_auc_score(y_true, y_pred)
        logloss = log_loss(y_true, y_pred)
        precision = precision_score(y_true, y_pred_bin)
        recall = recall_score(y_true, y_pred_bin)
        f1 = f1_score(y_true,y_pred_bin)

        wandb.log({
            "Validation AUC": auc,
            "Validation LogLoss": logloss,
            "Validation Precision": precision,
            "Validation Recall": recall,
            "Validation F1-Score": f1,
            }
        )

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=logloss,
            num_examples=self.num_val,
            metrics={
                "AUC": auc,
                "LogLoss": logloss,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
            },
        )

#____________________

#_______________________________________


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    train_dmatrix, valid_dmatrix, num_train, num_val, optimal_threshold, bank_name = load_data_for_client(
        partition_id, num_partitions
    )

    num_local_round = int(context.run_config.get("local_epochs", 3))

    scale_pos_weights = {
        0: context.run_config.get("scale_pos_weights.BankA_Clean", 1.0),
        1: context.run_config.get("scale_pos_weights.BankB_Clean", 1.0),
        2: context.run_config.get("scale_pos_weights.BankC_Clean", 1.0),
    }
    scale_pos_weight = scale_pos_weights.get(partition_id, 1.0)
    wandb.log({"scale_pos_weight": {scale_pos_weight}})

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8631,
        "min_child_weight": 5,
        "n_estimators": 200,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,

    return FlowerClient(
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        params,
        optimal_threshold,
        bank_name,
    )

#_______________________________________

app = ClientApp(client_fn=client_fn)
