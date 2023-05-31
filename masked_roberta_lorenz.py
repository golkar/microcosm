# %%

import huggingface_hub

# import wandb

# wandb.login()

token = "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"
huggingface_hub.login(token=token, add_to_git_credential=True)

# # %%

# from transformers import (
#     RobertaForMaskedLM,
#     RobertaConfig,
#     PreTrainedTokenizerFast,
#     DataCollatorForLanguageModeling,
# )
# from datasets import DatasetDict


# # %%
# # Loading the datasets into a datasetdict
# path = "/mnt/home/sgolkar/ceph/datasets/microcosm/lorenz_world_xsmall/clean/"
# ds = DatasetDict.from_text(
#     {"train": path + "train_set", "test": path + "test_set", "val": path + "val_set"}
# )


# # %%
# # Loading the tokenizer

# wrapped_tokenizer = PreTrainedTokenizerFast(
#     tokenizer_file="tokenizer_lorenz.json",
#     bos_token="[END]",
#     eos_token="[END]",
#     mask_token="?",
#     pad_token="[PAD]",
# )

# vocab_size = len(wrapped_tokenizer.vocab)

# # %%
# # Efficient parallel tokenization and saving by splitting the dataset into chunks

# def tokenize_fnc(sample):
#     return wrapped_tokenizer(sample["text"])


# tokenized_ds = ds.map(
#     tokenize_fnc,
#     batched=True,
#     num_proc=31,
#     remove_columns="text",
# )

# tokenized_ds.save_to_disk(path + "tokenized_ds")
# %%
from transformers import (
    RobertaForMaskedLM,
    RobertaConfig,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
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


# %%
# collating, padding and random masking
data_collator = DataCollatorForLanguageModeling(
    tokenizer=wrapped_tokenizer, mlm_probability=0.2
)

# an example of the output of the data_collator
samples = [tokenized_ds["train"][i] for i in range(1)]

# for chunk in data_collator(samples)["input_ids"]:
#     print(wrapped_tokenizer.decode(chunk))


# %%

hidden_size = 1440

config = RobertaConfig(
    vocab_size=vocab_size,
    max_position_embeddings=3000,
    num_attention_heads=6,
    num_hidden_layers=12,
    type_vocab_size=2,
    hidden_size=hidden_size,
    intermediate_size=4 * hidden_size,
)

model = RobertaForMaskedLM(config=config)
print(hidden_size, f"{model.num_parameters():,}")

# %%

train_size = 800_000
test_size = 5_000

# train_size = 10_000
# test_size = 1000

downsampled_dataset = tokenized_ds["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
# %%

from transformers import Trainer, TrainingArguments


training_args = TrainingArguments(
    output_dir="./roberta_lorenz_xsmall",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    save_total_limit=2,
    evaluation_strategy="steps",
    save_steps=5000,
    eval_steps=5000,
    # prediction_loss_only=True,
    # report_to="wandb",
    load_best_model_at_end=True,
)


def compute_metrics(eval_preds):
    return {}


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)


# # %%
# import math

# eval_results = trainer.evaluate()
# print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# %%

trainer.train()
# %%
# eval_results = trainer.evaluate()
# print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
# %%
# wandb.finish()
trainer.push_to_hub()
# %%

from transformers import pipeline

mask_filler = pipeline("fill-mask", model=model.cpu(), tokenizer=wrapped_tokenizer)

text = wrapped_tokenizer.decode(tokenized_ds["val"][0]["input_ids"])
split_text = text.split("11.41")
text_masked = split_text[0] + "?.41" + split_text[1]

preds = mask_filler(text_masked)

for pred in preds:
    print(pred["score"])
    print(f">>> {pred['sequence']}")
# %%
text = wrapped_tokenizer.decode(tokenized_ds["val"][0]["input_ids"])
split_text = text.split("'num_sys':1")
text_masked = split_text[0] + "'num_sys':?" + split_text[1]

preds = mask_filler(text_masked)

for pred in preds:
    print(pred["score"])
    print(f">>> {pred['sequence']}")
