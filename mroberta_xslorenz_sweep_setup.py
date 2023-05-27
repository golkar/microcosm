# %%
import wandb, pprint, os

# Setting the wandb notebook name environment variable
os.environ["WANDB_NOTEBOOK_NAME"] = "mroberta_xslorenz_sweep_setup.py"
wandb.login()

sweep_config = {"method": "random"}

metric = {"name": "loss", "goal": "minimize"}

parameters_dict = {
    "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
    "max_position_embeddings": {"value": 3000},
    "hidden_size": {"value": 120},  # needs to be divisible by num_attention_heads
    "num_hidden_layers": {"value": 12},
    "num_attention_heads": {"value": 6},
    "hidden_act": {"value": "gelu"},
    "hidden_dropout_prob": {"value": 0.1},
    "attention_probs_dropout_prob": {"value": 0.1},
    "mlm_probability": {"value": 0.2},
    "num_train_epochs": {"value": 1},
    "warmup_steps": {"value": 0},
    "weight_decay": {"value": 0},
    "fp16": {"value": True},
}

parameters_dict["intermediate_size"] = {
    "value": 4 * parameters_dict["hidden_size"]["value"]
}

sweep_config["parameters"] = parameters_dict
sweep_config["metric"] = metric

pprint.pprint(sweep_config)
# %%

sweep_id = wandb.sweep(sweep_config, project="xslorenz_mroberta")
# sweep_id = "bm2wbfb1"
# %%
