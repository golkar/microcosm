# %%

import wandb, os, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# set seaborn style
sns.set_style("ticks")

os.environ["WANDB_NOTEBOOK_NAME"] = "xslorenz_mroberta_results.py"
wandb.login()

# %%
api = wandb.Api()
runs = api.runs("xslorenz_mroberta")

results = []

num_params = {
    120: 2484742,
    240: 9145342,
    360: 19981942,
    480: 34994542,
    720: 77547742,
    1440: 305431342,
}

for run in runs:
    if run.state not in ["finished", "running"]:
        # if run.state == "running":
        #     print("running ", run.config["hidden_size"])
        #     print(run.summary['eval/loss'])
        continue

    run_dict = {
        "hidden_size": run.config["hidden_size"],
        "id": run.id,
        "config": run.config,
        "summary": run.summary,
        "eval/loss": run.summary["eval/loss"],
        "num_params": num_params[run.config["hidden_size"]] // 100000 / 10,
    }

    results.append(run_dict)


df = pd.DataFrame(results)
# %%

# plotting the top 2

# select the top two eval/loss for each hidden size of the dataframe
df_bests = df.groupby("hidden_size", as_index=False).apply(
    lambda x: x.nsmallest(2, "eval/loss")
)

ax = df_bests.boxplot(
    by="hidden_size",
    column=["eval/loss"],
    showfliers=True,
    showmeans=False,
    figsize=(8, 5),
)
ax.set_title("")
plt.suptitle("Validation loss vs hidden size")
ax.set_ylabel("Validation loss")
sns.despine(offset=15)
ax.grid(axis="x")

ax = df_bests.boxplot(
    by="num_params",
    column=["eval/loss"],
    showfliers=True,
    showmeans=False,
    figsize=(8, 5),
)
ax.set_title("")
plt.suptitle("Validation loss vs hidden size")
ax.set_ylabel("Validation loss")

ax.set_xticklabels([item.get_text() + "M" for item in ax.get_xticklabels()])

ax.grid(axis="x")
sns.despine(offset=15)

# %%
df_mean_n_std = df_bests.groupby("hidden_size", as_index=False)["eval/loss"].aggregate(
    ["mean", "std"]
)
ax = df_bests.plot.scatter(x="hidden_size", y="eval/loss", figsize=(8, 5))
df_mean_n_std.plot(y="mean", yerr="std", kind="line", ax=ax)


ax.set_title("")
plt.suptitle("Validation loss vs hidden size")
ax.set_ylabel("Validation loss")

# turn the legend off
ax.legend().set_visible(False)

ax.grid(axis="y")
sns.despine(offset=15)

ax = df_bests.plot.scatter(x="hidden_size", y="eval/loss", figsize=(8, 5))
df_mean_n_std.plot(y="mean", yerr="std", kind="line", ax=ax)


ax.set_title("")
plt.suptitle("Validation loss vs hidden size")
ax.set_ylabel("Validation loss")
hidden_sizes = list(set(df_bests["hidden_size"]))
ax.set_xticks(
    hidden_sizes, [str(num_params[el] // 1000000) + "M" for el in hidden_sizes]
)
ax.set_xlabel("Number of parameters")
ax.legend().set_visible(False)

ax.grid(axis="y")
sns.despine(offset=15)

# %%

# get the lowest eval/loss in the dataframe
best_run = df.loc[df["eval/loss"].idxmin()]

save_path = "/mnt/home/sgolkar/ceph/saves/xslorenz/mroberta/wandb/"

# get the folder in save_path that includes the best_run id
# best_run_path = (
#     save_path
#     + [dir for dir in os.listdir(save_path) if best_run["id"] in dir][0]
#     + "/files/model/checkpoint-200000"
# )

best_run_path = "/mnt/home/sgolkar/ceph/saves/xslorenz/mroberta/wandb/run-20230529_104249-rw7ta38d/files/model/checkpoint-200000"

# %%

from transformers import RobertaForMaskedLM, PreTrainedTokenizerFast, pipeline
from datasets import DatasetDict

model = RobertaForMaskedLM.from_pretrained(best_run_path).cuda()


wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer_lorenz.json",
    bos_token="[END]",
    eos_token="[END]",
    mask_token="?",
    pad_token="[PAD]",
)

mask_filler = pipeline("fill-mask", model=model, tokenizer=wrapped_tokenizer, device=0)


# Loading the saved tokenized dataset
path = "/mnt/home/sgolkar/ceph/datasets/microcosm/lorenz_world_xsmall/clean/"
tokenized_ds = DatasetDict.load_from_disk(path + "tokenized_ds")
# %%

path = "/mnt/home/sgolkar/ceph/datasets/microcosm/lorenz_world_xsmall/clean/"
ds = DatasetDict.from_text(
    {"train": path + "train_set", "test": path + "test_set", "val": path + "val_set"}
)

# %%


def remove_params(sample):
    text = sample["text"].replace(" ", "")
    split1 = text.partition("params':[")
    split2 = [split1[0], "params':[?.", split1[2].partition(".")[2]]
    ans = int(split1[2].partition(".")[0])
    masked_text = "".join(split2)
    return {"masked_text": masked_text, "answer": ans}


ds_with_ans = ds.map(remove_params, batched=False, num_proc=30)

# %%

sample = ds_with_ans["val"][0]
preds = mask_filler(sample["masked_text"])
print(sample["masked_text"])

print("correct answer:", sample["answer"])
for pred in preds:
    print(pred["score"], pred["token_str"])

# %%
import numpy as np

ans_dist_train = np.array(ds_with_ans["train"]["answer"])
ans_mean_train = int(np.mean(ans_dist_train))
sns.histplot(ans_dist_train, stat="probability")
plt.axvline(ans_mean_train, color="red", linestyle="--")
plt.show()

ans_dist_test = np.array(ds_with_ans["test"]["answer"])
print(
    "RMSE of the baseline (mean) on the test set:",
    np.sqrt(np.mean((ans_dist_test - ans_mean_train) ** 2)),
)

# %%

# preds = []
# anss = []
# probs = []
# for sample in tqdm(ds_with_ans["test"]):
#     out = mask_filler(sample["masked_text"])

#     pred = []
#     score = []
#     miss_count = 0
#     for el in out:
#         # continue if the token is not a number
#         if el["token_str"].isnumeric():
#             pred.append(int(el["token_str"]))
#             score.append(el["score"])
#         else:
#             miss_count += 1

#     prob = np.zeros(100)
#     prob[pred] = score

#     ans = sample["answer"]

#     preds.append(pred+[99]*miss_count)
#     anss.append(ans)
#     probs.append(prob)

# preds = np.array(preds)
# anss = np.array(anss)
# probs = np.array(probs)

# # save preds, anss, probs to numpy file
# np.savez(
#     "./preds_anss_probs.npz",
#     preds=preds,
#     anss=anss,
#     probs=probs,
# )

# load preds, anss, probs from numpy file
preds_anss_probs = np.load("./preds_anss_probs.npz")

anss = preds_anss_probs["anss"]
preds = preds_anss_probs["preds"]
probs = preds_anss_probs["probs"]

top_digits = np.argsort([(ans_dist_train == el).mean() for el in range(23)])[::-1]

# %%
print("baseline RMSE: ", anss.std())
print("learned RMSE:", np.sqrt(np.mean((anss - preds[:, 0]) ** 2)))
# %%
# comparing baseline and learned top 1 accuracy
print("baseline top 1 accuracy: {:.1%}".format((anss == ans_mean_train).mean()))
print("learned top 1 accuracy: {:.1%}".format((anss == preds[:, 0]).mean()))

# %%
# comparing baseline and learned top 2 accuracy
baseline_top2 = [ans in top_digits[:2] for ans in anss]
print("baseline top 2 accuracy: {:.1%}".format(np.mean(baseline_top2)))
learned_top2 = [ans in pred[:2] for ans, pred in zip(anss, preds)]
print("learned top 2 accuracy: {:.1%}".format(np.mean(learned_top2)))

# %%
# comparing baseline and learned top 3 accuracy
baseline_top3 = [ans in top_digits[:3] for ans in anss]
print("baseline top 3 accuracy: {:.1%}".format(np.mean(baseline_top3)))
learned_top3 = [ans in pred[:3] for ans, pred in zip(anss, preds)]
print("learned top 3 accuracy: {:.1%}".format(np.mean(learned_top3)))

# %%
# comparing baseline and learned top 5 accuracy
baseline_top5 = [ans in top_digits[:5] for ans in anss]
print("baseline top 5 accuracy: {:.1%}".format(np.mean(baseline_top5)))
learned_top5 = [ans in pred[:5] for ans, pred in zip(anss, preds)]
print("learned top 5 accuracy: {:.1%}".format(np.mean(learned_top5)))

# %%

# sns histogram of the difference between true answer and the predicted answer
sns.histplot(anss - preds[:, 0], stat="probability")
# %%
# computing the cross entropy
baseline_probs = np.array([(ans_dist_train == el).mean() for el in range(23)])
print("baseline cross entropy:", -np.log(baseline_probs[anss]).mean())
print("learned cross entropy:", -np.log(probs[range(len(anss)), anss]).mean())

# %%
