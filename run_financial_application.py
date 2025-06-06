import argparse
import os
import time
import openml
import pandas as pd
import numpy as np

from methods import ParameterOptimization 
from SMAC_method import ParameterOptimizationSMAC


def main(args):
    task_id = int(args.task)
    os.makedirs(args.result_folder, exist_ok=True)
    seed_folder = f"seed_{args.seed}"
    os.makedirs(os.path.join(args.result_folder, seed_folder), exist_ok=True)
    file_path = f"seed_{args.seed}/{task_id}.csv"
    path = f"Financial_DATA/"
    # Read the data from local files
    X = pd.read_csv(os.path.join(path, f"{task_id}_X.csv"))
    y = pd.read_csv(os.path.join(path, f"{task_id}_y.csv"))
    categorical_indicator = np.load(os.path.join(path, f"{task_id}_categorical_indicator.npy"))
    seed = int(args.seed)

    ## Run the experiment for the current financial task for RandomSearch, SMAC, TPE, GP-Boost tuning num_leaves because of classification
    obj = ParameterOptimizationSMAC(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=335, try_num_leaves=True,try_max_depth = False, joint_tuning_depth_leaves=False,try_num_iter=False, seed=seed) #Using suite 334 because of classification and 335 for regression
    final_results_SMAC = obj.method_smac()
    # Format the DataFrame
    final_results_SMAC["task_id"] = task_id
    final_results_SMAC["classification"] = 0
    # Save the results
    obj = ParameterOptimization(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=335, try_num_leaves=True,try_max_depth = False, joint_tuning_depth_leaves=False,try_num_iter=False, seed=seed) #Using suite 334 because of classification
    final_results = obj.run_methods()
    # Format the DataFrame
    final_results["task_id"] = task_id
    final_results["classification"] = 0 
    # Save the results
    final_results = pd.concat([final_results, final_results_SMAC],axis=0, ignore_index=True)
    final_results.to_csv(os.path.join(args.result_folder, file_path), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task')
    parser.add_argument('--seed')
    parser.add_argument('--result_folder')
    args = parser.parse_args()
    main(args)