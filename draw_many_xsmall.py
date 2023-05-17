from microcosm import sample_world
import uuid

path = "/mnt/home/sgolkar/ceph/datasets/microcosm/lorenz_world_xsmall/"


if __name__ == "__main__":
    import sys

    num_draws = int(sys.argv[1])

    filename = str(uuid.uuid4())

    samples = []
    for _ in range(num_draws):
        el = sample_world(
            to_list=True,
            graceful_fail=True,
            seed=None,
            num_steps=40,
            max_dim=10,
            max_sys=2,
        )
        if "data" in el.keys():
            samples.append(el)

    with open(
        r"{}{}".format(path, filename),
        "w",
    ) as f:
        for sample in samples:
            # write each item on a new line
            f.write("{}\n".format(el))
