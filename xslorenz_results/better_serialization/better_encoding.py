# %%
# %load_ext autoreload
# %autoreload 2

# check number replacement and tokenization give the correct original text
check_again = True
# how many samples to check
check_number = 10000


from datasets import DatasetDict
import numpy as np

# from KCroberta import *

from transformers import RobertaConfig

path = "/mnt/home/sgolkar/ceph/datasets/microcosm/lorenz_world_xsmall/clean/"
ds = DatasetDict.from_text(
    {"train": path + "train_set", "test": path + "test_set", "val": path + "val_set"}
)
# %%

# Defining the regular expression replacement to replace numbers with #s
# and add the numbers to a list
import re


def shorten_brackets(text):
    text = (
        text.replace("]]]", "thr_cl")
        .replace("[[[", "thr_op")
        .replace("[[", "two_op")
        .replace("]]", "two_cl")
        .replace("[", "one_op")
        .replace("]", "one_cl")
    )
    text = (
        text.replace("thr_cl ,thr_op", "thr_comma")
        .replace("two_cl ,twop", "two_comma")
        .replace("one_cl, one_op", "one_comma")
    )
    return text


def restore_brackets(text):
    text = (
        text.replace("thr_comma", "thr_cl ,thr_op")
        .replace("two_comma", "two_cl ,twop")
        .replace("one_comma", "one_cl, one_op")
    )
    text = (
        text.replace("thr_cl", "]]]")
        .replace("thr_op", "[[[")
        .replace("two_op", "[[")
        .replace("two_cl", "]]")
        .replace("one_op", "[")
        .replace("one_cl", "]")
    )
    return text


# Defining the regular expression replacement to replace numbers with #s
# and add the numbers to a list
import re

reg_ex = r"(-\d+\.\d+e-\d+)|(-\d+\.\d+e\d+)|(\d+\.\d+e-\d+)|(\d+\.\d+e\d+)|(-\d+e-\d+)|(-\d+e\d+)|(\d+e-\d+)|(\d+e\d+)|(-\d+\.\d+)|(\d+\.\d+)|((?<!sys)\d+)"


def replace_numbers(text):
    text = re.sub(reg_ex, "#", text).replace("#, #", "##").replace("#, #", "##")
    return shorten_brackets(text)


def find_numbers(text):
    return [eval("".join(el)) for el in re.findall(reg_ex, text)]


def reconstruct_numbers(text, numbers):
    text = text.replace("##", "#, #").replace("##", "#, #")
    for number in numbers:
        text = text.replace("#", str(number), 1)
    return restore_brackets(text)


# %%

# verifying that the functions work


if check_again:
    fix_text = lambda x: x.replace(".0,", ",").replace(".0}", "}").replace(".0]", "]")
    check_ok = True
    for i, samp in enumerate(ds["train"]):
        try:
            numbers = find_numbers(samp["text"])
            replaced_sample = replace_numbers(samp["text"])
            reconstructed_sample = reconstruct_numbers(replaced_sample, numbers)
            assert fix_text(samp["text"]) == fix_text(reconstructed_sample)
            if i > check_number:
                break
        except AssertionError:
            check_ok = False
            print("Exception at sample", i)
            print("Original text:\n", fix_text(samp["text"]))
            print("Reconstructed text:\n", fix_text(reconstructed_sample))
            break
    if check_ok:
        print("Number replacement check passed!")
# %%
# an example of the processed text
samp = ds["train"][0]
print("Original text:\n", samp["text"])
numbers = find_numbers(samp["text"])
replaced_sample = replace_numbers(samp["text"])
print("Replaced text:\n", replaced_sample)
reconstructed_sample = reconstruct_numbers(replaced_sample, numbers)
print("Reconstructed text:\n", reconstructed_sample)
# %%

# # Efficient parallel tokenization and saving by splitting the dataset into chunks

from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer.json",
    bos_token="[END]",
    eos_token="[END]",
    mask_token="?",
    pad_token="[PAD]",
)

vocab_size = len(tokenizer.vocab)
num_token = tokenizer.encode("#")[0]

# max length in is 595
max_len = 596


def tokenize_fncn(sample):
    text = sample["text"]
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
tokenize_again = True

test_mode = False
if test_mode:
    ds_to_tokenize = ds["train"].train_test_split(test_size=1, train_size=1000, seed=42)
else:
    ds_to_tokenize = ds


# because of variations in the length, batched tokenization is not possible (I think)
if tokenize_again:
    tokenized_ds = ds_to_tokenize.map(
        tokenize_fncn,
        batched=False,
        num_proc=30,
        remove_columns="text",
    )

    tokenized_ds.save_to_disk(path + "numenc_shortbrack_xslorenz_ds")
    tokenize_again = False

tokenized_ds = DatasetDict.load_from_disk(path + "numenc_shortbrack_xslorenz_ds")


# %%

# Checking things are still good after tokenization


if check_again:
    fix_text = (
        lambda x: x.replace(".0,", ",")
        .replace(".0}", "}")
        .replace(".0]", "]")
        .replace(" ", "")
    )
    check_ok = True
    lens = []
    for i, (sample_new, sample_old) in enumerate(
        zip(tokenized_ds["train"], ds_to_tokenize["train"])
    ):
        try:
            # text1 is the reconstructed text from the tokenized sample
            text1 = tokenizer.decode(sample_new["input_ids"])
            locs = np.array(sample_new["input_ids"]) == num_token
            text1 = fix_text(
                reconstruct_numbers(
                    text1, np.array(sample_new["numbers"][: len(locs)])[locs]
                )
            )
            # text2 is the original text from the dataset
            text2 = fix_text(sample_old["text"])

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
