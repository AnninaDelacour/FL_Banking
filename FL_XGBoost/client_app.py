"""xgboost_quickstart: A Flower / XGBoost app."""

import warnings
import pandas as pd
from flwr.common.context import Context
import xgboost as xgb
from flwr_xgb.task import load_adult_data, replace_keys
import wandb
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, roc_auc_score
from flwr.client import Client, ClientApp
from flwr.common.config import unflatten_dict
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Status,
)

#____________________________________________

warnings.filterwarnings("ignore", category=UserWarning)


class FlowerClient(Client):
    def __init__(
        self,
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        params
    ):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params

        wandb.init(
            project="client_drift",
            name=f"client_drift-{wandb.util.generate_id()}",
            reinit=True,
        )

    def _local_boost(self, bst_input):
        # Update trees based on local training data.
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for sever aggregation
        bst = bst_input[
            bst_input.num_boosted_rounds()
            - self.num_local_round : bst_input.num_boosted_rounds()
        ]

        return bst

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )
        else:
            bst = xgb.Booster(params=self.params)
            global_model = bytearray(ins.parameters.tensors[0])

            # Load global model into booster
            bst.load_model(global_model)

            # Local training
            bst = self._local_boost(bst)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        eval_result = bst.eval_set(
        evals=[(self.valid_dmatrix, "validate")],
        iteration=bst.num_boosted_rounds() - 1,
        )
        auc = float(eval_result.split("\t")[1].split(":")[1])

        # Get additional metrics
        y_true = self.valid_dmatrix.get_label()
        y_pred = bst.predict(self.valid_dmatrix)
        logloss = log_loss(y_true, y_pred)
        precision = precision_score(y_true, y_pred.round())
        recall = recall_score(y_true, y_pred.round())
        f1 = f1_score(y_true, y_pred.round())

        wandb.log({"AUC": auc, "LogLoss": logloss,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Round": global_round}
            )

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        bst = xgb.Booster(params=self.params)
        para_b = bytearray(ins.parameters.tensors[0])
        bst.load_model(para_b)

        y_true = self.valid_dmatrix.get_label()
        y_pred = bst.predict(self.valid_dmatrix)

        # For Debugging
        print("y_true distribution:", pd.Series(y_true).value_counts())
        print("y_pred distribution:", pd.Series(y_pred.round()).value_counts())


        # Metrics
        auc = roc_auc_score(y_true, y_pred)
        logloss = log_loss(y_true, y_pred)
        precision = precision_score(y_true, y_pred.round())
        recall = recall_score(y_true, y_pred.round())
        f1 = f1_score(y_true, y_pred.round())

        wandb.log({
            "Validation AUC": auc,
            "Validation LogLoss": logloss,
            "Validation Precision": precision,
            "Validation Recall": recall,
            "Validation F1-Score": f1,
            }
        )


        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_val,
            metrics={
            "AUC": auc,
            "LogLoss": logloss,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            },
        )


def client_fn(context: Context):
    # Load model and data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    train_dmatrix, valid_dmatrix, num_train, num_val = load_adult_data(
        partition_id, num_partitions
    )

    cfg = replace_keys(unflatten_dict(context.run_config))
    num_local_round = cfg["local_epochs"]

    # Return Client instance
    return FlowerClient(
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        cfg["params"],
    )


# Flower ClientApp
app = ClientApp(
    client_fn,
)
