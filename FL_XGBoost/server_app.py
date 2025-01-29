from typing import Dict

from flwr.common import Context, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedXgbBagging


#_______________________________________


def calculate_weights(eval_metrics):
    """
    Berechnet Gewichtungen für die Clients basierend auf ihrem Recall.
    Clients mit niedrigem Recall erhalten ein deutlich höheres Gewicht.
    """
    recalls = [metrics["Recall"] for _, metrics in eval_metrics]
    weights = []

    for r in recalls:
        if r < 0.5:
            weight = 1 / (r + 1e-10)  # Stärkste Gewichtung für sehr niedrigen Recall
        elif r < 0.7:
            weight = 1 / (r + 0.2)  # Mäßige Verstärkung für mittleren Recall
        else:
            weight = 1 / (r + 0.5)  # Leichte Verstärkung für hohen Recall
        weights.append(weight)

    # Normiere Gewichtungen
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    return weights



def evaluate_metrics_aggregation(eval_metrics):
    """Aggregiert die Metriken der Clients unter Berücksichtigung der Gewichtung."""
    weights = calculate_weights(eval_metrics)

    # Aggregierte Metriken berechnen
    auc_aggregated = sum(w * metrics["AUC"] for w, (_, metrics) in zip(weights, eval_metrics))
    recall_aggregated = sum(w * metrics["Recall"] for w, (_, metrics) in zip(weights, eval_metrics))

    metrics_aggregated = {
        "AUC (aggregated)": auc_aggregated,
        "Recall (aggregated)": recall_aggregated,
    }

    # Speichern des globalen Modells
    global_model_bytes = eval_metrics[0][1].get("global_model_bytes")
    if global_model_bytes:
        with open("global_model.json", "wb") as f:
            f.write(global_model_bytes)

    return metrics_aggregated


#_______________________________________

def config_func(rnd: int) -> Dict[str, str]:
    """Konfigurationsfunktion für jede Runde."""
    return {"global_round": str(rnd)}


#_______________________________________

def server_fn(context: Context):
    try:
        with open("global_model.json", "rb") as f:
            global_model = f.read()
    except FileNotFoundError:
        global_model = None

    fraction_fit = context.run_config.get("fraction-fit", 1.0)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 1.0)

    strategy = FedXgbBagging(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=config_func,
        on_fit_config_fn=config_func,
        initial_parameters=Parameters(tensor_type="", tensors=[global_model]),
    )

    config = ServerConfig(num_rounds=context.run_config.get("num-server-rounds", 30))

    return ServerAppComponents(strategy=strategy, config=config)

#_______________________________________

app = ServerApp(server_fn=server_fn)
