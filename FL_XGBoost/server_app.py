"""xgboost_quickstart: A Flower / XGBoost app."""

from typing import Dict
from flwr.common import Context, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedXgbBagging


def evaluate_metrics_aggregation(eval_metrics):
    """Return AUC for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = (
        sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    )
    metrics_aggregated = {"AUC": auc_aggregated}

    global_model_bytes = eval_metrics[0][1].get("global_model_bytes")  # Beispielzugriff
    if global_model_bytes:
        with open("global_model.json", "wb") as f:
            f.write(global_model_bytes)

    return metrics_aggregated


def config_func(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def server_fn(context: Context):
    with open("global_model.json", "rb") as f:
        global_model = f.read()

    # Lade Konfigurationswerte aus dem Kontext
    fraction_fit = context.run_config.get("fraction-fit", 0.7)

    fraction_evaluate = context.run_config.get("fraction-evaluate", 1.0)

    strategy = FedXgbBagging(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=config_func,
        on_fit_config_fn=config_func,
        initial_parameters=Parameters(tensor_type="", tensors=[global_model]),
    )

    config = ServerConfig(num_rounds=context.run_config.get("num-server-rounds", 10))

    return ServerAppComponents(strategy=strategy, config=config)



# Create ServerApp
app = ServerApp(
    server_fn=server_fn,
)
