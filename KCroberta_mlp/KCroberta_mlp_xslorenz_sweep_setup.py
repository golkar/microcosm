# %%
import wandb, pprint, os

# Setting the wandb notebook name environment variable
os.environ["WANDB_NOTEBOOK_NAME"] = "KCroberta_xslorenz_sweep_setup.py"
wandb.login()

sweep_config = {"method": "grid"}

metric = {"name": "loss", "goal": "minimize"}

parameters_dict = {
    "learning_rate": {"values": [2 * 1.5**n * 1e-5 for n in range(-1, 7)]},
    "max_position_embeddings": {"value": 1150},
    "hidden_size": {"value": 720},  # needs to be divisible by num_attention_heads
    "num_hidden_layers": {"value": 12},
    "num_attention_heads": {"value": 8},
    "hidden_act": {"value": "gelu"},
    "hidden_dropout_prob": {"value": 0},
    "attention_probs_dropout_prob": {"value": 0},
    "mlm_probability": {"value": 0.2},
    "num_train_epochs": {"value": 20},
    "warmup_steps": {"value": 5000},
    "weight_decay": {"value": 0.0001},
    "fp16": {"value": True},
    "power_num": {"value": 1 / 2},
    "number_embed_size": {"value": 32},
    "zero_others": {"value": True},
    "multiply_num_embedding": {"value": False},
}

parameters_dict["intermediate_size"] = {
    "value": 4 * parameters_dict["hidden_size"]["value"]
}

sweep_config["parameters"] = parameters_dict
sweep_config["metric"] = metric

pprint.pprint(sweep_config)
# %%

sweep_id = wandb.sweep(sweep_config, project="xslorenz_kcroberta_mlp")

# width 720 sweeps, power_num 1/2
# sweep_id = "5tmp2oca", grid: lr [2 * 1.5**n * 1e-5 for n in range(10)]

# width 720 sweeps, power_num 1/2, number_embed_size = 32
# sweep_id = "004vcml3", grid: lr [2 * 1.5**n * 1e-5 for n in range(10)]

# width 720 sweeps, power_num 1/2, number_embed_size = 32, zero_others = True, multiply_num_embedding = False
# sweep_id = "ezcuv7si", grid: lr [2 * 1.5**n * 1e-5 for n in range(-1, 7)]

# %%
