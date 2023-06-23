# %%
import wandb, pprint, os

# Setting the wandb notebook name environment variable
os.environ["WANDB_NOTEBOOK_NAME"] = "KCroberta_xslorenz_sweep_setup.py"
wandb.login()

sweep_config = {"method": "grid"}

metric = {"name": "loss", "goal": "minimize"}

parameters_dict = {
    "learning_rate": {"values": [2 * 1.5**n * 1e-5 for n in range(10)]},
    "max_position_embeddings": {"value": 597},  # has to be max length + 2
    "hidden_size": {"value": 640},  # needs to be divisible by num_attention_heads
    "num_hidden_layers": {"value": 10},
    "num_attention_heads": {"value": 10},
    "hidden_act": {"value": "gelu"},
    "hidden_dropout_prob": {"value": 0},
    "attention_probs_dropout_prob": {"value": 0},
    "mlm_probability": {"value": 0.25},
    "num_train_epochs": {"value": 40},
    "warmup_steps": {"value": 5000},
    "weight_decay": {"value": 0.0001},
    "fp16": {"value": True},
    "batch_size_total": {"value": 256},
}

parameters_dict["intermediate_size"] = {
    "value": 4 * parameters_dict["hidden_size"]["value"]
}

sweep_config["parameters"] = parameters_dict
sweep_config["metric"] = metric

pprint.pprint(sweep_config)
# %%

sweep_id = wandb.sweep(sweep_config, project="xslorenz_kcroberta_shortbrack")


# width 640 sweep
# sweep_id = "de9u7s35" grid [2*1.5**n * 1e-5 for n in range(10)]

# %%
