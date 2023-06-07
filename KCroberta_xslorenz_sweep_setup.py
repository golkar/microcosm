# %%
import wandb, pprint, os

# Setting the wandb notebook name environment variable
os.environ["WANDB_NOTEBOOK_NAME"] = "KCroberta_xslorenz_sweep_setup.py"
wandb.login()

sweep_config = {"method": "grid"}

metric = {"name": "loss", "goal": "minimize"}

parameters_dict = {
    "learning_rate": {"values": [2 * 1.5**n * 1e-5 for n in range(10)]},
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
}

parameters_dict["intermediate_size"] = {
    "value": 4 * parameters_dict["hidden_size"]["value"]
}

sweep_config["parameters"] = parameters_dict
sweep_config["metric"] = metric

pprint.pprint(sweep_config)
# %%

sweep_id = wandb.sweep(sweep_config, project="xslorenz_kcroberta")

# width 720 sweeps, power_num 1
# sweep_id = "tj9wjro8" grid [2*1.5**n * 1e-5 for n in range(10)]

# width 720 sweeps, power_num 1/3
# sweep_id = "zbzzry4f" grid [2*1.5**n * 1e-5 for n in range(10)]
# new sweep with torch save at the end
# sweep_id = "wqeq85eb" grid [2*1.5**n * 1e-5 for n in range(10)]

# width 720 sweeps, power_num 1/2, 20 epochs
# sweep_id = "cz111509" grid [2*1.5**n * 1e-5 for n in range(10)]

# %%
