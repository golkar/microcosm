# %%
# token = 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

from huggingface_hub import interpreter_login

interpreter_login()

# %%
from transformers import (
    RobertaForMaskedLM,
    RobertaConfig,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
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


# %%

config = RobertaConfig(
    vocab_size=vocab_size,
    max_position_embeddings=3000,
    num_attention_heads=6,
    num_hidden_layers=9,
    type_vocab_size=1,
)

model = RobertaForMaskedLM(config=config)
print(f"{model.num_parameters():,}")

# %%

train_size = 400_000
test_size = 5_000

downsampled_dataset = tokenized_ds["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
# %%

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./roberta_lorenz_xsmall",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    deepspeed="./ds_config.json",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
)
# %%
import math

eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# %%

trainer.train()
# %%
eval_results = trainer.evaluate()
import math

print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
# %%


# %%
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
