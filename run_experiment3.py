import argparse
import os

import openml
import pandas as pd
import numpy as np

from methods import ParameterOptimization
def main(args):
    os.makedirs(args.result_folder, exist_ok=True)
    seed_folder = f"seed_{args.seed}"
    os.makedirs(os.path.join(args.result_folder, seed_folder), exist_ok=True)
    tasks = open(f"Seed_Tasks/{args.suite_id}_{args.seed}.txt","r")
    for task_id in tasks:
        task_id = str(task_id).strip()
        file_path = f"seed_{args.seed}/{args.suite_id}_{task_id}.csv"
        path = f"data/{args.suite_id}_{task_id}"
        print(os.path.join(path, f"{args.suite_id}_{task_id}_X.csv"))
        # Read the data from local files
        X = pd.read_csv(os.path.join(path, f"{args.suite_id}_{task_id}_X.csv"))
        y = pd.read_csv(os.path.join(path, f"{args.suite_id}_{task_id}_y.csv"))
        categorical_indicator = np.load(os.path.join(path, f"{args.suite_id}_{task_id}_categorical_indicator.npy"))
        suite_id = int(args.suite_id)
        task_id = int(task_id)
        seed = int(args.seed)

        # # Run the experiment for the current task
        obj = ParameterOptimization(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite_id, try_num_leaves=False, joint_tuning_depth_leaves=False, seed=seed)
        final_results_md = obj.run_methods()

        obj = ParameterOptimization(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite_id, try_num_leaves=True, joint_tuning_depth_leaves=False,seed=seed)
        final_results_nl = obj.run_methods()

        obj = ParameterOptimization(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite_id, try_num_leaves=False, joint_tuning_depth_leaves=True, seed=seed)
        final_results_joint = obj.run_methods()
        # Format the DataFrame
        final_results = pd.concat([final_results_md, final_results_nl,final_results_joint], ignore_index=True)
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