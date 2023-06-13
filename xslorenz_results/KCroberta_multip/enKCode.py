# %%
# %load_ext autoreload
# %autoreload 2

# check number replacement and tokenization give the correct original text
check_again = False
# how many samples to check
check_number = 20000


from datasets import DatasetDict
import numpy as np
from KCroberta import *

from transformers import RobertaConfig

path = "/mnt/home/sgolkar/ceph/datasets/microcosm/lorenz_world_xsmall/clean/"
ds = DatasetDict.from_text(
    {"train": path + "train_set", "test": path + "test_set", "val": path + "val_set"}
)
# %%

# Defining the regular expression replacement to replace numbers with #s
# and add the numbers to a list
import re

reg_ex = r"(-\d+\.\d+e-\d+)|(-\d+\.\d+e\d+)|(\d+\.\d+e-\d+)|(\d+\.\d+e\d+)|(-\d+e-\d+)|(-\d+e\d+)|(\d+e-\d+)|(\d+e\d+)|(-\d+\.\d+)|(\d+\.\d+)|((?<!sys)\d+)"


def replace_numbers(text):
    text = re.sub(reg_ex, "#", text)
    return text


def find_numbers(text):
    return [eval("".join(el)) for el in re.findall(reg_ex, text)]


def reconstruct_numbers(text, numbers):
    for number in numbers:
        text = text.replace("#", str(number), 1)
    return text


# %%

# verifying that the functions work


if check_again:
    check_ok = True
    for i, sample in enumerate(ds["train"]):
        try:
            numbers = find_numbers(sample["text"])
            replaced_sample = replace_numbers(sample["text"])
            reconstructed_sample = reconstruct_numbers(replaced_sample, numbers)
            assert sample["text"] == reconstructed_sample
            if i > check_number:
                break
        except:
            check_ok = False
            print("Exception at sample", i)
            break
    if check_ok:
        print("Number replacement check passed!")

# %%

# # Efficient parallel tokenization and saving by splitting the dataset into chunks

from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="toKCenizer_lorenz.json",
    bos_token="[END]",
    eos_token="[END]",
    mask_token="?",
    pad_token="[PAD]",
)

vocab_size = len(tokenizer.vocab)

num_token = tokenizer.encode("#")[0]


# max length in is 1150
max_len = 1150


def tokenize_fnc(sample):
    text = sample["text"].replace(" ", "")
    replaced_text = replace_numbers(text)
    out = tokenizer(replaced_text)
    ids = np.array(out["input_ids"])
    ids = np.pad(ids, (0, max_len - len(ids)), "constant", constant_values=-1)
    locs = ids == num_token
    num_embed = np.ones(max_len).astype(np.float16)
    num_embed[locs] = find_numbers(text)
    out["numbers"] = num_embed
    return out


# Tokenizing the dataset and saving it
tokenize_again = False

# because of variations in the length, batched tokenization is not possible (I think)
if tokenize_again:
    tokenized_ds = ds.map(
        tokenize_fnc,
        # batched=True,
        num_proc=31,
        remove_columns="text",
    )

    tokenized_ds.save_to_disk(path + "toKCenized_xslorenz_ds")

tokenized_ds = DatasetDict.load_from_disk(path + "toKCenized_xslorenz_ds")


# %%

# Checking things are still good after tokenization


def reconstruct_sample(sample):
    num_ = np.array(sample["numbers"])
    ids_ = np.array(sample["input_ids"])
    ids_ = np.pad(ids_, (0, max_len - len(ids_)), "constant", constant_values=-1)
    loc_ = ids_ == num_token
    numbers = list(num_[loc_])

    return reconstruct_numbers(sample["replaced_text"], numbers)


if check_again:
    check_ok = True
    lens = []
    for i, (sample_new, sample_old) in enumerate(
        zip(tokenized_ds["train"], ds["train"])
    ):
        try:
            # text1 is the reconstructed text from the tokenized sample
            sample_new["replaced_text"] = tokenizer.decode(sample_new["input_ids"])
            text1 = (
                reconstruct_sample(sample_new)
                .replace(".0,", ",")
                .replace(".0]", "]")
                .replace(".0}", "}")
            )
            # text2 is the original text from the dataset
            text2 = (
                sample_old["text"]
                .replace(" ", "")
                .replace(".0,", ",")
                .replace(".0]", "]")
                .replace(".0}", "}")
            )

            lens.append(len(sample_new["input_ids"]))
            assert text1 == text2
            if i > check_number:
                break
        except:
            check_ok = False
            print("Exception at sample", i)
            break

    if check_ok:
        print("Tokenization reconstruction check passed!")

    import seaborn as sns

    sns.histplot(lens, discrete=True)


# %%

# putting everything together, new collator and model
KC_coll = KC_mlm_collator(tokenizer=tokenizer, mlm_probability=0.2)
masked_sample = KC_coll([tokenized_ds["train"][3], tokenized_ds["train"][4]])
hidden_size = 360

config = RobertaConfig(
    vocab_size=vocab_size,
    max_position_embeddings=3000,
    num_attention_heads=6,
    num_hidden_layers=12,
    type_vocab_size=2,
    hidden_size=hidden_size,
    intermediate_size=4 * hidden_size,
)

model = KCRobertaForMaskedLM(config=config, power_num=1 / 3)
out = model(**masked_sample)
print(out)
# %%
# the backward pass seems to be fine
out.loss.backward()

# %%
