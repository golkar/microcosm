# This script follow the huggingface tutorial here:
# https://huggingface.co/docs/tokenizers/python/latest/quicktour.html
# https://huggingface.co/learn/nlp-course/chapter6/8?fw=pt

# %%
from tokenizers import (
    decoders,
    models,
    processors,
    Tokenizer,
    pre_tokenizers,
)


vocab = ["{}".format(el) for el in range(10)] + [
    "{",
    "}",
    "[",
    "]",
    ",",
    ".",
    "-",
]

vocab_words = (
    ["{:02d}".format(el) for el in range(100)]
    + ["'sys{}':".format(el) for el in range(5)]
    + [
        "'num_sys':",
        "'params':",
        "'data':",
        "'description':",
        "'step_size':",
        "'init_point':",
        "'step_multip':",
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
    ]
)

tokenizer = Tokenizer(
    models.BPE(vocab={el: i for i, el in enumerate(vocab)}, merges=[])
)


tokenizer.add_special_tokens(["[END]", "?", "[PAD]"])
tokenizer.add_tokens(vocab_words)

file = "/mnt/home/sgolkar/ceph/datasets/microcosm/lorenz_world_xsmall/clean/0000"
# load this to see the results of the tokenizer
with open(
    file,
    "r",
) as f:
    out = f.read()

dataset = out.split("\n")[:-1]

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()

tokenizer.save("tokenizer_lorenz.json")


# %%
from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer_lorenz.json",
    bos_token="[END]",
    eos_token="[END]",
    mask_token="?",
    pad_token="[PAD]",
)
# %%
# Testing after restarting kernel

file = "/mnt/home/sgolkar/ceph/datasets/microcosm/lorenz_world_xsmall/clean/0000"
# load this to see the results of the tokenizer
with open(
    file,
    "r",
) as f:
    out = f.read()

dataset = out.split("\n")[:-1]
sample = dataset[6]

encoding = wrapped_tokenizer(sample)
print(wrapped_tokenizer.decode(encoding.input_ids) == sample.replace(" ", ""))
print(encoding.tokens())

# %%
