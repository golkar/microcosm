# %%
import wandb, os, time, json

# Setting the wandb notebook name environment variable
os.environ["WANDB_NOTEBOOK_NAME"] = "mroberta_xslorenz_sweep_agent.py"
wandb.login()

local_rank = int(os.environ.get("LOCAL_RANK", 0))
global_rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

# get the slurm job id (used later for saving the config file)
slurm_job_id = os.getenv("SLURM_JOB_ID")

# if running locally, set the slur_job_id to current time
if slurm_job_id is None:
    slurm_job_id = int(time.time())

save_path = "/mnt/home/sgolkar/ceph/saves/xslorenz/mroberta/"
config_file = save_path + str(slurm_job_id)

is_master = global_rank == 0

print("local_rank: ", local_rank)
print("global_rank: ", global_rank)
print("world_size: ", world_size)
print("slurm_job_id: ", slurm_job_id)

# %%
from transformers import (
    RobertaForMaskedLM,
    RobertaConfig,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import DatasetDict

# Loading the tokenizer and setting the vocab size
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer_lorenz.json",
    bos_token="[END]",
    eos_token="[END]",
    mask_token="?",
    pad_token="[PAD]",
)
vocab_size = len(wrapped_tokenizer.vocab)

# Loading the saved tokenized dataset
path = "/mnt/home/sgolkar/ceph/datasets/microcosm/lorenz_world_xsmall/clean/"
tokenized_ds = DatasetDict.load_from_disk(path + "tokenized_ds")

train_size = 800_000
test_size = 5000

downsampled_dataset = tokenized_ds["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)


# %%


# defining the train function with wandb config as input
def train(run_id, chkpt_path):
    # pausing to read wandb config if not master
    if not is_master:
        # pause 10 seconds to wait for master to finish setting up
        print("Waiting for master to finish setting up...")
        time.sleep(10)

        # read wandb config
        with open(config_file, "r") as f:
            config = dict(json.load(f))

    # Initialize a new wandb run with the received config

    if is_master:
        wandb.init(id=run_id, resume="must", project="xslorenz_mroberta", dir=save_path)
        wandb_config = wandb.config
        # write wandb config file as a dict to a file in tmp
        print("Writing wandb config to " + config_file + "...")
        with open(config_file, "w") as f:
            json.dump(dict(wandb_config), f)
    else:
        wandb.init(config=config, mode="disabled")
        wandb_config = wandb.config

    # collating, padding and random masking
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=wrapped_tokenizer, mlm_probability=wandb_config.mlm_probability
    )

    # Defining the model config from the wandb_config
    model_config = RobertaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=wandb_config.max_position_embeddings,
        hidden_size=wandb_config.hidden_size,
        num_hidden_layers=wandb_config.num_hidden_layers,
        num_attention_heads=wandb_config.num_attention_heads,
        intermediate_size=wandb_config.intermediate_size,
        hidden_act=wandb_config.hidden_act,
        hidden_dropout_prob=wandb_config.hidden_dropout_prob,
        attention_probs_dropout_prob=wandb_config.attention_probs_dropout_prob,
    )
    model = RobertaForMaskedLM.from_pretrained(chkpt_path)
    # model = RobertaForMaskedLM(config=model_config)

    # defining the training args
    training_args = TrainingArguments(
        # output_dir=wandb.run.dir + "/model",
        output_dir="/mnt/home/sgolkar/ceph/saves/xslorenz/mroberta/wandb/run-20230531_162250-rw7ta38d/files/model/",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=8 // world_size,
        save_total_limit=3,
        logging_steps=200,
        report_to="wandb",
        evaluation_strategy="steps",
        save_steps=5000,
        eval_steps=5000,
        load_best_model_at_end=True,
        learning_rate=2e-5,
        warmup_steps=15_000,
        weight_decay=wandb_config.weight_decay,
        fp16=wandb_config.fp16,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # An empty compute_metrics function to just log val loss
    def compute_metrics(eval_preds):
        return {}

    # defining the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=downsampled_dataset["train"],
        eval_dataset=downsampled_dataset["test"],
        tokenizer=wrapped_tokenizer,
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    # trainer.create_optimizer_and_scheduler(num_training_steps=500_000)
    # import torch

    # trainer.optimizer.load_state_dict(torch.load(chkpt_path + "/optimizer.pt"))
    # trainer.scaler.load_state_dict(torch.load(chkpt_path + "/scaler.pt"))

    # trainer.train()
    trainer.train(resume_from_checkpoint=True)


# %%

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # add sweepid as text argument
    parser.add_argument(
        "--run_id",
        type=str,
        help="The sweep id for the wandb sweep",
    )

    parser.add_argument(
        "--chkpt_path",
        type=str,
        help="Path to the checkpoint to resume from",
    )

    args = parser.parse_args()

    train(run_id=args.run_id, chkpt_path=args.chkpt_path)
# %%
