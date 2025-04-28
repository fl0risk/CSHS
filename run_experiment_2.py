import argparse
import os

import openml
import pandas as pd
import numpy as np

from methods import ParameterOptimization


#python run_experiment_2.py  --suite_id=334 --task_id=361110 --seed=27225 --result_folder=Deleted
def main(args):
    os.makedirs(args.result_folder, exist_ok=True)
    seed_folder = f"seed_{args.seed}"
    os.makedirs(os.path.join(args.result_folder, seed_folder), exist_ok=True)
    file_path = f"seed_{args.seed}/{args.suite_id}_{args.task_id}.csv"
    path = f"data/{args.suite_id}_{args.task_id}"
    print(os.path.join(path, f"{args.suite_id}_{args.task_id}_X.csv"))
    # Read the data from local files
    X = pd.read_csv(os.path.join(path, f"{args.suite_id}_{args.task_id}_X.csv"))
    y = pd.read_csv(os.path.join(path, f"{args.suite_id}_{args.task_id}_y.csv"))
    categorical_indicator = np.load(os.path.join(path, f"{args.suite_id}_{args.task_id}_categorical_indicator.npy"))
    suite_id = int(args.suite_id)
    task_id = int(args.task_id)
    seed = int(args.seed)
    obj = ParameterOptimization(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite_id, try_num_leaves=False,try_max_depth = False, joint_tuning_depth_leaves=True,try_num_iter=True, seed=seed)
    final_results = obj.run_methods()
    # Format the DataFrame
    final_results["task_id"] = task_id
    final_results["classification"] = 1 if suite_id in [334, 337] else 0

    # Save the results
    final_results.to_csv(os.path.join(args.result_folder, file_path), index=False,header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite_id')
    parser.add_argument('--task_id')
    parser.add_argument('--seed')
    parser.add_argument('--result_folder')
    args = parser.parse_args()
    main(args)
