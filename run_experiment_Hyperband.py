import argparse
import os

import openml
import pandas as pd
import numpy as np

from methods import ParameterOptimization
from SMAC_method import ParameterOptimizationSMAC
#suite_id = 335 task_id = 361102 seed = 27225 result_folder = "OneTaskTest"
#python run_experiment_Hyperband.py  --suite_id=334 --task_id=361110 --seed=27225 --result_folder=SMAC
#python run_experiment_Hyperband.py  --suite_id=335 --task_id=361102 --seed=27225 --result_folder=SMAC
def main(args):
    os.makedirs(args.result_folder, exist_ok=True)
    seed_folder = f"seed_{args.seed}"
    os.makedirs(os.path.join(args.result_folder, seed_folder), exist_ok=True)
    file_path = f"seed_{args.seed}/{args.suite_id}_{args.task_id}.csv"
    seed = int(args.seed)
    suite_id = int(args.suite_id)
    task_id = int(args.task_id)
    # benchmark_suite = openml.study.get_suite(suite_id)  # obtain the benchmark suite
    # task = openml.tasks.get_task(task_id)  # download the OpenML task
    # dataset = task.get_dataset()
    # X, y, categorical_indicator, attribute_names = dataset.get_data(
    #      dataset_format="dataframe", target=dataset.default_target_attribute
    #  )
    path = f'/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/data/{suite_id}_{task_id}'
    
    X = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_X.csv"))
    y = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_y.csv"))
    categorical_indicator = np.load(os.path.join(path, f"{suite_id}_{task_id}_categorical_indicator.npy"))
    # # Run the experiment for the current task
    obj = ParameterOptimizationSMAC(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite_id, 
                                try_num_leaves=True,try_max_depth = False, joint_tuning_depth_leaves=False,try_num_iter=False, seed=seed)
    final_results = obj.method_smac()
    obj = ParameterOptimizationSMAC(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite_id, try_num_leaves=False,try_num_iter=True, seed=seed)
    final_results_nl = obj.method_smac()
    #final_results = obj.method_smac()#.run_hyperband()
    # Format the DataFrame
    final_results = pd.concat([final_results,final_results_nl], ignore_index=True) # add [final_results_md,final_results_nl,final_results_joint]
    final_results["task_id"] = task_id
    final_results["classification"] = 1 if suite_id in [334, 337] else 0

    # Save the results
    final_results.to_csv(os.path.join(args.result_folder, file_path), index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite_id')
    parser.add_argument('--task_id')
    parser.add_argument('--seed')
    parser.add_argument('--result_folder')
    args = parser.parse_args()
    main(args)