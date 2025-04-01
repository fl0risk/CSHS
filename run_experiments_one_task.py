import argparse
import os
import time
import openml
import pandas as pd
import numpy as np

from methods import ParameterOptimization

#suite_id = 335 task_id = 361102 seed = 27225 result_folder = "OneTaskTest"
#python run_experiments_one_task.py  --suite_id=335 --task_id=361102 --seed=27225 --result_folder=OneTaskTest
def main(args):
    start_time = time.time()
    os.makedirs(args.result_folder, exist_ok=True)
    seed_folder = f"seed_{args.seed}"
    os.makedirs(os.path.join(args.result_folder, seed_folder), exist_ok=True)
    file_path = f"seed_{args.seed}/{args.suite_id}_{args.task_id}.csv"
    seed = int(args.seed)
    suite_id = int(args.suite_id)
    task_id = int(args.task_id)
    benchmark_suite = openml.study.get_suite(suite_id)  # obtain the benchmark suite
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
         dataset_format="dataframe", target=dataset.default_target_attribute
     )

    # # Run the experiment for the current task
    # obj = ParameterOptimization(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite_id, try_num_leaves=False,joint_tuning_depth_leaves=False, seed=seed)
    # final_results_md = obj.run_methods()
    
    # obj = ParameterOptimization(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite_id, try_num_leaves=True,joint_tuning_depth_leaves=False, seed=seed)
    # final_results_nl = obj.run_methods()

    obj = ParameterOptimization(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite_id, try_num_leaves=False,joint_tuning_depth_leaves=True, seed=seed)
    final_results_joint = obj.run_methods()

    # Format the DataFrame
    final_results = pd.concat([final_results_joint], ignore_index=True) # add [final_results_md,final_results_nl,final_results_joint]
    final_results["task_id"] = task_id
    final_results["classification"] = 1 if suite_id in [334, 337] else 0

    # Save the results
    final_results.to_csv(os.path.join(args.result_folder, file_path), index=False)
    
    end_time = time.time()
    elapsed_time = end_time - start_time    
    with open("timing_results.txt", "a") as f:
        f.write(f"Elapsed time: {elapsed_time:.4f} seconds\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite_id')
    parser.add_argument('--task_id')
    parser.add_argument('--seed')
    parser.add_argument('--result_folder')
    args = parser.parse_args()
    main(args)