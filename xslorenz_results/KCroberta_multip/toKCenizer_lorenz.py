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


vocab = [
    "{",
    "}",
    "[",
    "]",
    ",",
    ".",
    "-",
    "#",
]

vocab_words = ["'sys{}':".format(el) for el in range(5)] + [
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
]

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


def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]


tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()

tokenizer.save("toKCenizer_lorenz.json")

# %%
