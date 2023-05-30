# %%
import wandb, argparse, deepspeed, time, json, os


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


def train(config=None):
    # log wandb created time

    is_master = cmd_args.local_rank == 0

    if not is_master:
        # pause 5 seconds to wait for master to finish setting up
        print("Waiting for master to finish setting up...")
        time.sleep(20)

        # read wandb config
        with open("/tmp/wandb_config.json", "r") as f:
            config = dict(json.load(f))

        # delete wandb config file
        os.remove("/tmp/wandb_config.json")

    # Initialize a new wandb run
    with wandb.init(config=config, mode=None if is_master else "disabled"):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        wandb_config = wandb.config

        if is_master:
            # write wandb config file as a dict to a file in tmp
            print("Writing wandb config to /tmp/wandb_config.json")
            with open("/tmp/wandb_config.json", "w") as f:
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
            per_device_train_batch_size=4,
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
            # deepspeed="./ds_config.json",
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
    # Get args
    parser = argparse.ArgumentParser(description="My training script.")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )

    # add sweepid as text argument

    parser.add_argument(
        "--sweepid",
        type=str,
        help="The run id of the wandb run to resume",
    )

    # # Include DeepSpeed configuration arguments
    # parser = deepspeed.add_config_arguments(parser)

    cmd_args = parser.parse_args()

    if cmd_args.local_rank == 0:  # only on main process
        # Initialize wandb run
        wandb.agent(cmd_args.sweepid, train, count=1, project="xslorenz_mroberta")
    else:
        train(config={})


# %%
