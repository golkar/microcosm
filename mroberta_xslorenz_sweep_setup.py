# %%
import wandb, pprint, os

# Setting the wandb notebook name environment variable
os.environ["WANDB_NOTEBOOK_NAME"] = "mroberta_xslorenz_sweep_setup.py"
wandb.login()

sweep_config = {"method": "grid"}

metric = {"name": "loss", "goal": "minimize"}

parameters_dict = {
    "learning_rate": {"values": [1.5**n * 1e-4 for n in range(1, 8)]},
    "max_position_embeddings": {"value": 3000},
    "hidden_size": {"value": 240},  # needs to be divisible by num_attention_heads
    "num_hidden_layers": {"value": 12},
    "num_attention_heads": {"value": 6},
    "hidden_act": {"value": "gelu"},
    "hidden_dropout_prob": {"value": 0.1},
    "attention_probs_dropout_prob": {"value": 0.1},
    "mlm_probability": {"value": 0.2},
    "num_train_epochs": {"value": 2},
    "warmup_steps": {"value": 2000},
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

# width 120 sweeps
# sweep_id = "bm2wbfb1" random {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
# sweep_id = "vo2h3dxu" grid [0.0016, 0.0025600000000000006, 0.004096000000000001, 0.0065536000000000014, 0.010485760000000004]

# width 240 sweeps
# sweep_id = "da7pd9yg" grid [1.5**n*1E-4 for n in range(1,8)]
# %%
