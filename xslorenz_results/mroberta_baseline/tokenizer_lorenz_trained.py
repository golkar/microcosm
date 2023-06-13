# This script follow the huggingface tutorial here:
# https://huggingface.co/docs/tokenizers/python/latest/quicktour.html
# https://huggingface.co/learn/nlp-course/chapter6/8?fw=pt

# %%
from tokenizers import decoders, models, processors, Tokenizer, pre_tokenizers, trainers

tokenizer = Tokenizer(models.BPE())

file = "/mnt/home/sgolkar/ceph/datasets/microcosm/lorenz_world_xsmall/clean/0000"
# load this to see the results of the tokenizer
with open(
    file,
    "r",
) as f:
    out = f.read()

dataset = out.split("\n")[:-1]


tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(vocab_size=500, special_tokens=["[END]", "?", "[PAD]"])
tokenizer.train([file], trainer=trainer)

tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()

tokenizer.save("tokenizer_lorenz_trained.json")


# %%
from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer_lorenz_trained.json",
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
print(
    wrapped_tokenizer.decode(encoding.input_ids).replace(" ", "")
    == sample.replace(" ", "")
)
print(encoding.tokens())

# %%
