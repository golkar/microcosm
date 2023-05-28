# %%
import wandb, os

# Setting the wandb notebook name environment variable
os.environ["WANDB_NOTEBOOK_NAME"] = "mroberta_xslorenz_sweep_agent.py"
wandb.login()


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

# Loading the tokenizer

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
# defining the model initialization function


def train(run_id):
    # Getting the path to the last checkpoint

    if not args.noresume:
        path_last_run = (
            "wandb/"
            + sorted([folder for folder in os.listdir("wandb") if run_id in folder])[-1]
        )
        path_last_chkpt = (
            path_last_run
            + "/files/model/"
            + sorted(
                [
                    folder
                    for folder in os.listdir(path_last_run + "/files/model")
                    if "checkpoint" in folder
                ]
            )[-1]
        )

        print("Will resume from checkpoint:", path_last_chkpt)

    # Initialize a new wandb run
    wandb.init(
        id=run_id,
        resume="must",
        project="xslorenz_mroberta",
    )
    # If called by wandb.agent, as below,
    # this config will be set by Sweep Controller
    wandb_config = wandb.config

    if args.noresume:
        print("finishing previous run and starting new one...")
        try:
            wandb.finish()
        except:
            print("Error in finishing previous run but continuing...")

        wandb.init(config=dict_conf, project="xslorenz_mroberta")

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
        per_device_train_batch_size=8,
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
        deepspeed="./ds_config.json",
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

    trainer.train(resume_from_checkpoint=None if args.noresume else path_last_chkpt)

    # Saving the trainer state
    trainer.save_state()


# %%

if __name__ == "__main__":
    import argparse

    # Get args
    parser = argparse.ArgumentParser(description="My resume script.")

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )

    parser.add_argument(
        "--run_id",
        type=str,
        help="The run id of the wandb run to resume",
    )

    # add boolean arg for resuming from checkpoint
    parser.add_argument(
        "--noresume",
        action="store_true",
        help="Whether to resume from checkpoint",
    )

    args = parser.parse_args()

    # Run the resuemd training
    train(run_id=args.run_id)
