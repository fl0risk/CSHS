"""Author: Ioana Iacobici https://github.com/iiacobici modified by Floris Koster https://github.com/fl0risk"""
import json
import os
import numpy as np
import openml

def main():    
    tasks = [361055,361066]

    for task_id in tasks:
        task = openml.tasks.get_task(task_id)  # download the OpenML task
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format="dataframe", target=dataset.default_target_attribute
            )
        os.makedirs("Financial_DATA", exist_ok=True)
        X.to_csv(f"Financial_DATA/{task_id}_X.csv", index=False)
        y.to_csv(f"Financial_DATA/{task_id}_y.csv", index=False)
        np.save(f"Financial_DATA/{task_id}_categorical_indicator.npy", categorical_indicator)

if __name__ == '__main__':
    main()