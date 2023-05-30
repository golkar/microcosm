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
def train(config=None):
    # pausing to read wandb config if not master
    if not is_master:
        # pause 10 seconds to wait for master to finish setting up
        print("Waiting for master to finish setting up...")
        time.sleep(10)

        # read wandb config
        with open(config_file, "r") as f:
            config = dict(json.load(f))

    # Initialize a new wandb run with the received config
    with wandb.init(config=config, dir=save_path + "wandb"):
        wandb_config = wandb.config

        if is_master:
            # write wandb config file as a dict to a file in tmp
            print("Writing wandb config to /tmp/wandb_config.json")
            with open(config_file, "w") as f:
                json.dump(dict(wandb_config), f)

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
        model = RobertaForMaskedLM(config=model_config)

        # defining the training args
        training_args = TrainingArguments(
            output_dir=wandb.run.dir + "/model",
            overwrite_output_dir=True,
            num_train_epochs=wandb_config.num_train_epochs,
            per_device_train_batch_size=8 // world_size,
            save_total_limit=2,
            logging_steps=200,
            report_to="wandb",
            evaluation_strategy="steps",
            save_steps=5000,
            eval_steps=5000,
            load_best_model_at_end=True,
            learning_rate=wandb_config.learning_rate,
            warmup_steps=wandb_config.warmup_steps,
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
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        trainer.train()


# %%

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # add sweepid as text argument
    parser.add_argument(
        "--sweepid",
        type=str,
        help="The sweep id for the wandb sweep",
    )

    # adding the sweep count argument with default value 1
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="The number of times to run the sweep",
    )

    args = parser.parse_args()

    if is_master:
        wandb.agent(args.sweepid, train, count=args.count, project="xslorenz_mroberta")
    else:
        train(config={})
# %%
