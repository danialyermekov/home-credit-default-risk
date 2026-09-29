# src/home_credit/core/model_config.py

from typing import Any

FINAL_CB_PARAMS: dict[str, Any] = {
    "iterations": 4397,
    "depth": 7,
    "learning_rate": 0.02011158504,
    "l2_leaf_reg": 6.61868763,
    "random_strength": 0.07185154408,
    "border_count": 254,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": 42,
    "task_type": "GPU",
    "devices": "0",
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "grow_policy": "SymmetricTree",
    "leaf_estimation_method": "Newton",
    "leaf_estimation_iterations": 10,
    "max_ctr_complexity": 4,
    "one_hot_max_size": 2,
    "use_best_model": False,
    "verbose": 200,
}
