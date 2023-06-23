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
    "thr_cl",
    "thr_op",
    "two_cl",
    "two_op",
    "one_cl",
    "one_op",
    "thr_comma",
    "two_comma",
    "one_comma",
]

tokenizer = Tokenizer(
    models.BPE(vocab={el: i for i, el in enumerate(vocab)}, merges=[])
)


tokenizer.add_special_tokens(["[END]", "?", "[PAD]"])
tokenizer.add_tokens(vocab_words)


tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()

tokenizer.save("tokenizer.json")

# %%
