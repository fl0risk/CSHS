import argparse
import os

# import openml
import pandas as pd
import numpy as np

from methods import ParameterOptimization



def main(args):
    task_id = int(args.task_id)
    if task_id < 7:
        tasks = np.loadtxt(f"suite_334_tasks.txt")
        task_id = int(tasks[task_id])
        suite_id = 334
    elif task_id < 24:
        tasks = np.loadtxt(f"suite_335_tasks.txt")
        task_id = int(tasks[task_id])
        suite_id = 335
    elif task_id < 43:
        tasks = np.loadtxt(f"suite_336_tasks.txt")
        task_id = int(tasks[task_id])
        suite_id = 336
    else:
        tasks = np.loadtxt(f"suite_337_tasks.txt")
        task_id = int(tasks[task_id])
        suite_id = 337
    os.makedirs(args.result_folder, exist_ok=True)
    
    file_path = f"{suite_id}_{task_id}.csv"
    path = f"/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/data/{suite_id}_{task_id}" #TODO: delete path before data

    # Read the data from local files
    X = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_X.csv"))
    y = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_y.csv"))
    categorical_indicator = np.load(os.path.join(path, f"{suite_id}_{task_id}_categorical_indicator.npy"))

    

    # Run the experiment for the current task
    obj = ParameterOptimization(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite_id)
    final_results = obj.run_default()
    final_results["task_id"] = task_id
    final_results["classification"] = 1 if suite_id in [334, 337] else 0

    # Save the results
    final_results.to_csv(os.path.join(args.result_folder, file_path), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id')
    parser.add_argument('--result_folder')
    args = parser.parse_args()
    main(args)
