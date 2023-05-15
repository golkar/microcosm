from microcosm import sample_world
import uuid


if __name__ == "__main__":
    import sys

    num_draws = int(sys.argv[1])

    filename = str(uuid.uuid4())

    with open(
        r"/mnt/home/sgolkar/ceph/datasets/microcosm/lorenz_world/{}".format(filename),
        "w",
    ) as f:
        for _ in range(num_draws):
            el = sample_world(to_list=True, graceful_fail=True, seed=None)

            # write each item on a new line
            f.write("{}\n".format(el))
