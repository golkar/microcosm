# %%
import wandb, os, time, json, torch

# Setting the wandb notebook name environment variable
os.environ["WANDB_NOTEBOOK_NAME"] = "KCroberta_mlp_xslorenz_sweep_agent.py"
wandb.login()

local_rank = int(os.environ.get("LOCAL_RANK", 0))
global_rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

# get the slurm job id (used later for saving the config file)
slurm_job_id = os.getenv("SLURM_JOB_ID")

# if running locally, set the slur_job_id to current time
if slurm_job_id is None:
    slurm_job_id = int(time.time())

save_path = "/mnt/home/sgolkar/ceph/saves/xslorenz/KCroberta_mlp/"
config_file = save_path + str(slurm_job_id)

is_master = global_rank == 0


# %%
from transformers import (
    RobertaConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import DatasetDict

from KCroberta_mlp import KC_mlm_collator, KCRobertaForMaskedLMMLP

# Loading the tokenizer and setting the vocab size
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="../toKCenizer_lorenz.json",
    bos_token="[END]",
    eos_token="[END]",
    mask_token="?",
    pad_token="[PAD]",
)
number_token = tokenizer("#")["input_ids"][0]
vocab_size = len(tokenizer.vocab)

# Loading the saved tokenized dataset
ds_path = "/mnt/home/sgolkar/ceph/datasets/microcosm/lorenz_world_xsmall/clean/"
tokenized_ds = DatasetDict.load_from_disk(ds_path + "toKCenized_xslorenz_ds")

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
        # pause 30 seconds to wait for master to finish setting up
        print("Waiting for master to finish setting up...")
        time.sleep(30)

        # read wandb config
        with open(config_file, "r") as f:
            config = dict(json.load(f))

    # Initialize a new wandb run with the received config
    with wandb.init(
        config=config, dir=save_path, mode=None if is_master else "disabled"
    ):
        wandb_config = wandb.config

        if is_master:
            # write wandb config file as a dict to a file in tmp
            print("Writing wandb config to " + config_file + "...")
            with open(config_file, "w") as f:
                json.dump(dict(wandb_config), f)

        # collating, padding and random masking
        data_collator = KC_mlm_collator(
            tokenizer=tokenizer, mlm_probability=wandb_config.mlm_probability
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
        model = KCRobertaForMaskedLMMLP(
            config=model_config,
            power_num=wandb_config.power_num,
            number_embed_size=wandb_config.number_embed_size,
            zero_others=wandb_config.zero_others,
            multiply_num_embedding=wandb_config.multiply_num_embedding,
            number_token=number_token,
        )

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

        if is_master:
            print("Model save path: " + wandb.run.dir + "/model")

        # Reporting the mlm loss and the numbers loss
        def compute_metrics(eval_preds):
            return {
                "loss_mlm": eval_preds[0][0].mean(),
                "loss_numbers": eval_preds[0][1].mean(),
            }

        # defining the trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=downsampled_dataset["train"],
            eval_dataset=downsampled_dataset["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        )

        trainer.train()

        if is_master:
            print("done training...")

            torch.save(model, wandb.run.dir + "/model/model_torch_save.pkl")
            print(
                "saved to file {}".format(wandb.run.dir + "/model/model_torch_save.pkl")
            )


# %%

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # add sweep_id as text argument
    parser.add_argument(
        "--sweep_id",
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
        wandb.agent(
            args.sweep_id, train, count=args.count, project="xslorenz_kcroberta_mlp"
        )
    else:
        train(config={})
# %%
