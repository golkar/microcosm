# %%

from os import listdir
from os.path import isfile, join

mypath = r"/mnt/home/sgolkar/ceph/datasets/microcosm/lorenz_world_xsmall"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# %%
aggr = []
for i, file in enumerate(onlyfiles):
    print("{}: Loading file {}".format(i, file))

    with open(
        r"{}/{}".format(mypath, file),
        "r",
    ) as f:
        out = f.read()

    print("Separating and dropping empty rows".format(file))

    out2 = [el for el in out.split("\n")[:-1] if el != "{}"]
    aggr.extend(out2)

    print("Writing to file.")

    with open(
        r"{}/clean/{:04d}".format(mypath, i),
        "w",
    ) as f:
        for el in out2:
            # write each item on a new line
            f.write("{}\n".format(el))
# %%

train_set = aggr[: int(len(aggr) * 0.8)]
val_set = aggr[int(len(aggr) * 0.8) : int(len(aggr) * 0.9)]
test_set = aggr[int(len(aggr) * 0.9) :]

# %%
with open(
    r"{}/clean/train_set".format(mypath),
    "w",
) as f:
    for el in train_set:
        # write each item on a new line
        f.write("{}\n".format(el))

with open(
    r"{}/clean/val_set".format(mypath),
    "w",
) as f:
    for el in val_set:
        # write each item on a new line
        f.write("{}\n".format(el))

with open(
    r"{}/clean/test_set".format(mypath),
    "w",
) as f:
    for el in test_set:
        # write each item on a new line
        f.write("{}\n".format(el))
