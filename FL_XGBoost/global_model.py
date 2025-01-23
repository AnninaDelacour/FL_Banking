import wandb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt


train_df = pd.read_csv("/Users/annina/Documents/FH/FHStPoelten/FML/assignment1/assignment_1/adult_train.csv")
test_df = pd.read_csv("/Users/annina/Documents/FH/FHStPoelten/FML/assignment1/assignment_1/adult_test.csv")

X_train, y_train = train_df.drop("income", axis=1), train_df["income"]
X_test, y_test = test_df.drop("income", axis=1), test_df["income"]


wandb.init(
    project="global-model-xgboost",
    name="global_model_training",
    config={
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 8,
        "learning_rate": 0.1004,
        "subsample": 1.0,
        "num_boost_round": 200,
        "early_stopping_rounds": 10,
    },
)

config = wandb.config


dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(X_test.columns))


bst = xgb.train(
    {
        "objective": config.objective,
        "eval_metric": config.eval_metric,
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "subsample": config.subsample,
    },
    dtrain,
    num_boost_round=config.num_boost_round,
    evals=[(dtrain, "train"), (dtest, "test")],
    early_stopping_rounds=config.early_stopping_rounds,
    verbose_eval=1,
)


y_test_preds = bst.predict(dtest)
auc_test = roc_auc_score(y_test, y_test_preds)
print("Test AUC:", auc_test)
wandb.log({"Test AUC": auc_test})


xgb.plot_importance(bst, max_num_features=20)
plt.savefig("feature_importance.png")
wandb.log({"Feature Importance": wandb.Image("feature_importance.png")})


bst.save_model("global_model.json")
global_model_bytes = bst.save_raw("json")

with open("global_model.json", "wb") as f:
    f.write(global_model_bytes)

wandb.save("global_model.json")

wandb.finish()
