# %%
import huggingface_hub, wandb, os

token = "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"
huggingface_hub.login(token=token, add_to_git_credential=True)

os.environ["WANDB_NOTEBOOK_NAME"] = "train_KCroberta.py"
wandb.login()
# %%

# loading the tokenizer and the tokenized dataset with numbers extracted
from transformers import PreTrainedTokenizerFast
from datasets import DatasetDict

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="toKCenizer_lorenz.json",
    bos_token="[END]",
    eos_token="[END]",
    mask_token="?",
    pad_token="[PAD]",
)

vocab_size = len(tokenizer.vocab)

ds_path = "/mnt/home/sgolkar/ceph/datasets/microcosm/lorenz_world_xsmall/clean/"
tokenized_ds = DatasetDict.load_from_disk(ds_path + "toKCenized_xslorenz_ds")

# %%
# defining the new collator type with numbers

from KCroberta import KC_mlm_collator

KC_coll = KC_mlm_collator(tokenizer=tokenizer, mlm_probability=0.2)

# %%
# defining the roberta derived model

from transformers import RobertaConfig
from KCroberta import KCRobertaForMaskedLM

hidden_size = 360

config = RobertaConfig(
    vocab_size=vocab_size,
    max_position_embeddings=1150,
    num_attention_heads=6,
    num_hidden_layers=12,
    type_vocab_size=2,
    hidden_size=hidden_size,
    intermediate_size=4 * hidden_size,
)

model = KCRobertaForMaskedLM(config=config, power_num=1 / 3)

print(hidden_size, f"{model.num_parameters():,}")
# %%

#  defining a small dataset for testing the model

train_size = 800_000
test_size = 5000

downsampled_dataset = tokenized_ds["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)

# %%

# defining the trainer

from transformers import Trainer, TrainingArguments


training_args = TrainingArguments(
    output_dir="./KCroberta_xslorenz",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_total_limit=2,
    evaluation_strategy="steps",
    save_steps=5000,
    eval_steps=500,
    logging_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    learning_rate=0.00002,
    warmup_steps=2000,
    weight_decay=0.0001,
)


def compute_metrics(eval_preds):
    return {
        "loss_mlm": eval_preds[0][0].mean(),
        "loss_numbers": eval_preds[0][1].mean(),
    }


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=KC_coll,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# %%

trainer.train()

# %%
