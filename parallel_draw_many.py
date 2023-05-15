from microcosm import sample_world
from multiprocessing import Pool
import os
import numpy as np


def f(seed):
    return sample_world(seed=seed, to_list=True, graceful_fail=True)


def draw_many(num_draws, start_seed, num_cpus=None):
    # get number of available cpus
    if num_cpus is None:
        num_cpus = os.cpu_count()

    # create a list of seeds
    seeds = np.arange(start_seed, start_seed + num_draws)

    if num_cpus > 1:
        # create a pool of workers
        with Pool(num_cpus) as pool:
            # run sample_world() in parallel using all available cpus
            # with seed given from a list starting from start_seed
            # and return the results
            return pool.starmap(f, [(seed,) for seed in seeds])
    else:
        return [f(seed) for seed in seeds]


# Take bash inputs num_draws and start_seed and run draw_many()
if __name__ == "__main__":
    import sys

    start_seed = int(sys.argv[1])
    num_draws = int(sys.argv[2])
    out = draw_many(num_draws, start_seed)
    with open(
        r"/mnt/home/sgolkar/ceph/datasets/microcosm/lorenz_world/{}-{}".format(
            start_seed, start_seed + num_draws
        ),
        "w",
    ) as f:
        for el in out:
            # write each item on a new line
            f.write("{}\n".format(el))
