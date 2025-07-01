import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from functools import reduce
import pandas as pd
from methods import ParameterOptimization
suite = [334,335,336,337]
seeds = [27225, 34326,92161, 99246, 108473, 117739,  235053, 257787, 
        89389, 443417, 572858, 620176, 671487, 710570, 773246, 936518,32244,147316, 777646, 778572]
FOLDS = [0,1,2,3,4]
NUM_ITERS = 135
NUM_SEEDS = len(seeds)
NUM_FOLDS = len(FOLDS)
def get_scores_for_one_task(task,  seed, suite=335,rmse = False):
    scores = np.zeros((NUM_ITERS, NUM_FOLDS))
    data = pd.read_csv(f"/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/Results/seed_{seed}/{suite}_{task}.csv")
    if rmse:
        try_max_depth = data.loc[(data['try_num_leaves'] == False) & (data['joint_tuning_depth_leaves'] == False), 'test_rmse'].reset_index(drop=True)
        df = pd.DataFrame({'try_max_depth_rmse': try_max_depth})
        df['method'] = data['method']
        df['fold'] = data['fold']
        for m in FOLDS:
                scores[:,m] = df.loc[(df['method'] == 'tpe') & (df['fold'] == m), 'try_max_depth_rmse'].values
        #scores = np.mean(scores, -1)
    else:
        try_max_depth = data.loc[(data['try_num_leaves'] == False) & (data['joint_tuning_depth_leaves'] == False), 'test_score'].reset_index(drop=True)
        df = pd.DataFrame({'try_max_depth_score': try_max_depth})
        df['method'] = data['method']
        df['fold'] = data['fold']
        for m in FOLDS:
                scores[:,m] = df.loc[(df['method'] == 'tpe') & (df['fold'] == m), 'try_max_depth_score'].values
        #scores = np.mean(scores, -1)
    return scores

def get_std(task,suite =335):
    path = f'/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/data/{suite}_{task}'
    y = pd.read_csv(os.path.join(path, f"{suite}_{task}_y.csv"))
    std =y.std(ddof=0, axis=0)
    return std

def get_data(task, suite=335):
     path = f'/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/data/{suite}_{task}'
     y = pd.read_csv(os.path.join(path, f"{suite}_{task}_y.csv"))
     X = pd.read_csv(os.path.join(path, f"{suite}_{task}_X.csv"))
     cat_indicator = np.load(os.path.join(path, f"{suite}_{task}_categorical_indicator.npy"))
     return y,X,cat_indicator


def get_metrics(task, suite, seed = 27225):
    y, X, categorical_indicator = get_data(int(task), int(suite))
    # Ensure y is numeric, coerce errors to NaN and drop them
    obj = ParameterOptimization(X=X, y=y, categorical_indicator=categorical_indicator, suite_id=suite, try_num_leaves=False, try_max_depth=True, joint_tuning_depth_leaves=False, try_num_iter=False, seed=seed)
    std_y = []
    for fold, (full_train_index, test_index) in enumerate(obj.splits):
        y_test = (obj.y).iloc[test_index]
        std_y.append(y_test.std(axis=0, ddof=0))
    weights = [std_y[j]**(-2) for j in range(len(std_y))]
    weights = np.array(weights)
    weights = weights[np.newaxis, :]
    # weights_deviation = np.mean(np.abs(weights - np.mean(weights)))
    # range_weights = np.max(weights)-np.min(weights)
    diff_min = np.median(weights)-np.min(weights)
    diff_max = np.max(weights)-np.median(weights)
    return weights, diff_min, diff_max  #weights_deviation / np.mean(weights), range_weights/np.mean(weights), range_weights,

def print_coeff(suite):
    names = []
    with open(f"task_indices/{suite}_task_names.json", 'r') as f:
            _names = json.load(f)
    names.extend(_names)
    tasks = np.load(f"task_indices/{suite}_task_indices.npy")
    # normalized_mad = np.zeros((len(tasks)))
    # relative_range = np.zeros((len(tasks)))
    # range_weights = np.zeros((len(tasks)))
    weights = np.zeros((5,len(tasks)))
    diff_min = np.zeros((len(tasks)))
    diff_max = np.zeros((len(tasks)))
    RMSE =  np.zeros((len(tasks),135,5))
    Rsquared = np.zeros((len(tasks),135,5))
    for j, task in enumerate(tasks):
        weights[:,j],diff_min[j], diff_max[j]= get_metrics(int(task),int(suite)) # normalized_mad[j], relative_range[j], range_weights[j],
        Rsquared[j,:,:] = get_scores_for_one_task(task = task,suite = suite,rmse = False,seed =27225)
        RMSE[j,:,:] = get_scores_for_one_task(task = task,suite = suite,rmse =True,seed =27225)
    upper_bound_task = np.mean(RMSE**2,axis = -1)*diff_min[:,np.newaxis]
    lower_bound_task = -np.mean(RMSE**2,axis = -1)*diff_max[:,np.newaxis]
    # print(np.max(np.mean(RMSE**2,axis = (2)),axis=1))
    # upper_bound = np.mean(RMSE**2,axis = (0,2))*(np.median(weights, axis = (0,1))-np.min(weights, axis=(0,1)))
    # lower_bound = -np.mean(RMSE**2,axis = (0,2))*(np.max(weights, axis=(0,1))-np.median(weights, axis = (0,1)))
    # # print(f'Normalized MAD for each task of Suite {suite}', normalized_mad)
    # # print(f'Relative Range for each task of Suite {suite}', relative_range)
    # print(f'Absolute Range for each task of Suite {suite}', range_weights)
    # print(f'The differnce to the min is {diff_min} and from the mean to the max {diff_max}')
    print(f'Taskwise: Upper bound for difference when approximating {upper_bound_task} and the lower bound {lower_bound_task}, shape is {lower_bound_task.shape}')
    max_upper_idx = np.unravel_index(np.argmax(upper_bound_task), upper_bound_task.shape)[0]
    min_lower_idx = np.unravel_index(np.argmin(lower_bound_task), lower_bound_task.shape)[0]
    print(f'The maximum value is {np.max(upper_bound_task)} for the upper bound found in task {max_upper_idx}')
    print(f'The minimum value is {np.min(lower_bound_task)} for the lower bound found in task {min_lower_idx}')
    #print(f'Overall: Upper bound for difference when approximating {upper_bound} and the lower bound {lower_bound}, shape is {lower_bound.shape}')