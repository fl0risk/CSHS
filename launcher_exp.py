import random

# import openml
import numpy as np

import run_experiment
from utils_exp import generate_base_command, generate_run_commands



suites = [334, 335, 336, 337]
# random.seed(42)
# seeds = random.sample(range(1000, 1000000), 20)
# seeds.sort()
seeds = [27225, 32244, 34326, 92161, 99246, 108473, 117739, 147316, 235053, 257787, 
        89389, 443417, 572858, 620176, 671487, 710570, 773246, 777646, 778572, 936518]


def main():
    command_list = []

    for seed in seeds:
        for suite_id in suites:
            tasks = np.load(f"task_indices/{suite_id}_task_indices.npy")

            # benchmark_suite = openml.study.get_suite(suite_id)
            # tasks = benchmark_suite.tasks

            for task_id in tasks:
                cmd = generate_base_command(run_experiment, flags=dict(suite_id=suite_id, task_id=task_id, seed=seed, result_folder='result_folder'))
                command_list.append(cmd)

    # Submit jobs
    generate_run_commands(command_list, promt=False
                          , num_cpus=1
                          , mem= 32 * 1024
                          , duration= '119:59:00'
                          , mode= 'euler' # 'local'
                          )


if __name__ == '__main__':
    main()