# %%

import torch
from transformers import (
    RobertaForMaskedLM,
    RobertaConfig,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
)


# Dataset for code testing purposes

file = "/mnt/home/sgolkar/ceph/datasets/microcosm/lorenz_world_small/clean/0000"
with open(
    file,
    "r",
) as f:
    out = f.read()

dataset = out.split("\n")[:-1]
sample = dataset[6]

# %%

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer_lorenz.json",
    bos_token="[END]",
    eos_token="[END]",
    mask_token="[MASK]",
    pad_token="[PAD]",
)

vocab_size = len(wrapped_tokenizer.vocab)

encoding = wrapped_tokenizer(dataset[:1], return_special_tokens_mask=True)

input = torch.tensor(encoding.input_ids).cuda()
# %%

config = RobertaConfig(
    vocab_size=vocab_size,
    max_position_embeddings=10000,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

model = RobertaForMaskedLM(config=config).cuda()
print(f"{model.num_parameters():,}")

output = model(input)

# %%

data_collator = DataCollatorForLanguageModeling(
    tokenizer=wrapped_tokenizer, mlm_probability=0.15
)

# %%

# for sample in dataset[:4]:
#     _ = sample.pop("word_ids")

for chunk in data_collator(encoding)["input_ids"]:
    print(f"\n' {wrapped_tokenizer.decode(chunk)}'")
# %%
