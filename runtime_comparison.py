import argparse
import os

#import openml
import pandas as pd
import numpy as np

from methods import ParameterOptimization
import time

def main(args):
    os.makedirs(args.result_folder, exist_ok=True)
    seed_folder = f"seed_{args.seed}"
    os.makedirs(os.path.join(args.result_folder, seed_folder), exist_ok=True) 
    if int(args.suite_id) == 334:
        tasks = [361282,361286]
    elif int(args.suite_id) == 335:
        tasks = [361289,361293]
    for task_id in tasks:
        file_path = f"seed_{args.seed}/{args.suite_id}_{task_id}.csv"
        path = f"data/{args.suite_id}_{task_id}"
        # Read the data from local files
        X = pd.read_csv(os.path.join(path, f"{args.suite_id}_{task_id}_X.csv"))
        y = pd.read_csv(os.path.join(path, f"{args.suite_id}_{task_id}_y.csv"))
        categorical_indicator = np.load(os.path.join(path, f"{args.suite_id}_{task_id}_categorical_indicator.npy"))

        suite_id = int(args.suite_id)
        task_id = int(task_id)
        seed = int(args.seed)

        # # Run the experiment for the current task
        runtime_file = os.path.join(args.result_folder, seed_folder, f"{args.suite_id}_{task_id}_runtimes.txt")
        # Measure runtime for Hyperband and Random Search, and store with task info
        obj = ParameterOptimization(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite_id, try_num_leaves=True,try_max_depth=False, joint_tuning_depth_leaves=False,seed=seed)
        start_hyperband = time.time()
        final_results_hyperband = obj.run_hyperband()
        end_hyperband = time.time()
        hyperband_runtime = end_hyperband - start_hyperband
        with open(runtime_file, "w") as f:
            f.write(f"hyperband_runtime,{hyperband_runtime/3600}\n")
        # Measure runtime for Random Search
        obj = ParameterOptimization(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite_id, try_num_leaves=True,try_max_depth=False, joint_tuning_depth_leaves=False,seed=seed)
        start_random_search = time.time()
        final_results_random_search = obj.run_random_search()
        end_random_search = time.time()
        random_search_runtime = end_random_search - start_random_search
        with open(runtime_file, "a") as f:
            f.write(f"random_search_runtime,{random_search_runtime/3600}\n")
        # Format the DataFrame
        final_results = pd.concat([final_results_hyperband, final_results_random_search], ignore_index=True)
        final_results["task_id"] = task_id
        final_results["classification"] = 1 if suite_id in [334, 337] else 0

        # Save the results
        final_results.to_csv(os.path.join(args.result_folder, file_path), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite_id')
    parser.add_argument('--seed')
    parser.add_argument('--result_folder')
    args = parser.parse_args()
    main(args)