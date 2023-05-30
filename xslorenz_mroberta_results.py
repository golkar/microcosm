# %%

import wandb, os, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

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
    if run.state != "finished":
        if run.state == "running":
            print("running ", run.config["hidden_size"])
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

ax.legend().set_visible(False)

ax.grid(axis="y")
sns.despine(offset=15)
