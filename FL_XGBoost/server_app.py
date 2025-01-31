from typing import Dict
from flwr.common import Context, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedXgbBagging

# _______________________________________

CLIENT_WEIGHTS = {
    "BankA": 0.248092,
    "BankB": 0.344442,
    "BankC": 0.407465,
}

ALPHA = 0.7

# _______________________________________

def evaluate_metrics_aggregation(eval_metrics):
    """Aggregates metrics of clients with weighted AUC and Precision."""
    
    total_weight = sum(CLIENT_WEIGHTS.values())
    weighted_metric = sum(
        CLIENT_WEIGHTS.get(metrics["bank_name"], 1.0) * (
            ALPHA * metrics["AUC"] + (1 - ALPHA) * metrics["Precision"]
        )
        for _, metrics in eval_metrics
    ) / total_weight
    
    metrics_aggregated = {"Aggregated Metric": weighted_metric}

    global_model_bytes = eval_metrics[0][1].get("global_model_bytes")
    if global_model_bytes:
        with open("latest_model.json", "wb") as f:
            f.write(global_model_bytes)

    return metrics_aggregated

# _______________________________________

def config_func(rnd: int) -> Dict[str, str]:
    return {"global_round": str(rnd)}

# _______________________________________

def server_fn(context: Context):
    try:
        with open("latest_model.json", "rb") as f:
            global_model = f.read()
    except FileNotFoundError:
        global_model = b"" 


    fraction_fit = context.run_config.get("fraction-fit", 1.0)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 1.0)

    initial_params = None if not global_model else Parameters(tensor_type="", tensors=[global_model])

    strategy = FedXgbBagging(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=config_func,
        on_fit_config_fn=config_func,
        initial_parameters=initial_params,
)


    config = ServerConfig(num_rounds=context.run_config.get("num-server-rounds", 30))
    
    return ServerAppComponents(strategy=strategy, config=config)

# _______________________________________

app = ServerApp(server_fn=server_fn)