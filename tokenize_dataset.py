from transformers import (
    RobertaForMaskedLM,
    RobertaConfig,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
)
from datasets import DatasetDict


# %%
# Loading the datasets into a datasetdict
path = "/mnt/home/sgolkar/ceph/datasets/microcosm/lorenz_world_xsmall/clean/"
ds = DatasetDict.from_text(
    {"train": path + "train_set", "test": path + "test_set", "val": path + "val_set"}
)


# %%
# Loading the tokenizer

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer_lorenz.json",
    bos_token="[END]",
    eos_token="[END]",
    mask_token="?",
    pad_token="[PAD]",
)

vocab_size = len(wrapped_tokenizer.vocab)

# %%
# Efficient parallel tokenization and saving by splitting the dataset into chunks

def tokenize_fnc(sample):
    return wrapped_tokenizer(sample["text"])


tokenized_ds = ds.map(
    tokenize_fnc,
    batched=True,
    num_proc=31,
    remove_columns="text",
)

tokenized_ds.save_to_disk(path + "tokenized_ds")
