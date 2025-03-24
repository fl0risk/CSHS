import os

import numpy as np
import pandas as pd



suites = [334, 335, 336, 337]
tasks_to_transform = {334: [], 335: [361099], 336: [361082, 361088], 337: []}

def main():
    for suite_id in suites:
        tasks = np.load(f"task_indices/{suite_id}_task_indices.npy")

        for task_id in tasks:
            path = f"original_data/{suite_id}_{task_id}"
            X = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_X.csv"))
            y = pd.read_csv(os.path.join(path, f"{suite_id}_{task_id}_y.csv")).iloc[:, 0]
            categorical_indicator = np.load(os.path.join(path, f"{suite_id}_{task_id}_categorical_indicator.npy"))

            if task_id in tasks_to_transform[suite_id]:
                y = np.log(y)

            path = f"data/{suite_id}_{task_id}"
            os.makedirs(path, exist_ok=True)
            X.to_csv(os.path.join(path, f"{suite_id}_{task_id}_X.csv"), index=False)
            y.to_csv(os.path.join(path, f"{suite_id}_{task_id}_y.csv"), index=False)
            np.save(os.path.join(path, f"{suite_id}_{task_id}_categorical_indicator.npy"), categorical_indicator) 


if __name__ == '__main__':
    main()