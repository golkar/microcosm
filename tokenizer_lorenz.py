# This script follow the huggingface tutorial here:
# https://huggingface.co/docs/tokenizers/python/latest/quicktour.html
# https://huggingface.co/learn/nlp-course/chapter6/8?fw=pt

# %%
from tokenizers import (
    decoders,
    models,
    processors,
    Tokenizer,
)


tokenizer = Tokenizer(models.BPE())

vocab = ["{}".format(el) for el in range(10)] + [
    "{:02d}".format(el) for el in range(100)
]
vocab = [
    "'num_sys':",
    "'params':",
    "'data':",
    "'init_point':",
    "'step_size':",
    "'step_multip':",
    "{",
    "}",
    "[",
    "]",
    ",",
    ".",
    "-",
    "'name':",
    "'lorenz'",
    "'normalization':",
    "'embedding':",
    "000",
    "0000",
    "e-05",
    "e-06",
    "e-07",
    "e-08",
] + vocab
vocab = ["'sys{}':".format(el) for el in range(5)] + vocab

tokenizer.add_tokens(vocab)
tokenizer.add_special_tokens(["[END]", "[MASK]"])

tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()

tokenizer.save("tokenizer_lorenz.json")

# %%
from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer_lorenz.json",
    bos_token="[END]",
    eos_token="[END]",
    mask_token="[MASK]",
)
# %%
# Testing after restarting kernel

file = "/mnt/home/sgolkar/ceph/datasets/microcosm/lorenz_world_small/clean/0000"
# load this to see the results of the tokenizer
with open(
    file,
    "r",
) as f:
    out = f.read()

dataset = out.split("\n")[:-1]
sample = dataset[6]

encoding = wrapped_tokenizer.encode(sample)
print(wrapped_tokenizer.decode(encoding) == sample.replace(" ", ""))
# %%

encoding = wrapped_tokenizer.__call__(dataset)
# %%
