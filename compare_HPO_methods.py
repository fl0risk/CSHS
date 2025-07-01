"""Some functions copied from Ioana Iacobici https://github.com/iiacobici and modified by Floris Koster https://github.com/fl0risk"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from functools import reduce

seeds = [27225, 34326,92161, 99246, 108473, 117739,  235053, 257787, 
        89389, 443417, 572858, 620176, 671487, 710570, 773246, 936518,32244,147316, 777646, 778572]


NUM_SEEDS = len(seeds)
NUM_TASKS = 59
NUM_REGR_TASKS = 36
NUM_CLASS_TASKS = 23
NUM_ITERS = 135
NUM_ITERS_H = -1

HPO_METHODS_SUBS = ['random_search', 'tpe', 'gp_bo','SMAC']
HPO_METHODS = ['random_search', 'tpe', 'gp_bo','hyperband', 'SMAC']
HPO_METHODS_NAMES = ['Random Grid Search', 'TPE', 'GP-BO','Hyperband', 'SMAC']
HPO_METHODS_SUBS_NAMES = ['Random Grid Search', 'TPE', 'GP-BO','SMAC']
TUNING_STRAT = ['Num Leaves','Max Depth','Joint']
TUNING_STRAT_TASK2 = ['Num Leaves','Max Depth','Joint','Num Iter']
NUM_TUNING_STRAT = len(TUNING_STRAT)
NUM_TUNING_STRAT_TASK2 = len(TUNING_STRAT_TASK2)
NUM_METHODS = len(HPO_METHODS)
NUM_METHODS_SUBS = len(HPO_METHODS_SUBS)
RANDOMNESS = ['both','seeds','tasks']
FOLDS = [0, 1, 2, 3, 4]
NUM_FOLDS = len(FOLDS)
MARKERS = ["o", "*", "^", "s","d"] 
def get_nested_shape(arr, level=0):
    """Recursively print the shape of a nested list or numpy array."""
    if isinstance(arr, np.ndarray):
        print("  " * level + f"ndarray shape: {arr.shape}")
        return arr.shape
    elif isinstance(arr, list):
        print("  " * level + f"list of length: {len(arr)}")
        if len(arr) > 0:
            return (len(arr),) + tuple(get_nested_shape(arr[0], level + 1))
        else:
            return (0,)
    else:
        print("  " * level + f"type: {type(arr)}")
        return ()
def set_plot_theme():
    # Set Seaborn theme
    sns.set_theme(context="paper", style="white")
    palette = ['lime', 'darkorange', 'fuchsia', 'deepskyblue','crimson']

    return palette

def create_scores_dict(classification=False, rmse=False):
    if classification:
        NUM_TASKS = NUM_CLASS_TASKS
        suites = [334, 337]
    
    else:
        NUM_TASKS = NUM_REGR_TASKS
        suites = [335, 336] 
    #get here the number of iterations for hyperband
    data_H = pd.read_csv("/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/Results_H/seed_27225/334_361110.csv")    
    NUM_ITERS_H = len(data_H.loc[(data_H['fold'] == 0) & (data_H['joint_tuning_depth_leaves'] == True)])
    if NUM_ITERS_H > NUM_ITERS:
                    ValueError('There are too much iterations in the Hyperband method.')
    scores_NL = [np.zeros((NUM_ITERS, NUM_TASKS, NUM_SEEDS, NUM_FOLDS)) if HPO_METHODS[i] != 'hyperband' else np.zeros((NUM_ITERS_H, NUM_TASKS, NUM_SEEDS, NUM_FOLDS)) for i in range(NUM_METHODS)]
    for i in range(NUM_METHODS):
        print('The following shape for',i, scores_NL[i].shape)
    scores_MD = [np.zeros((NUM_ITERS, NUM_TASKS, NUM_SEEDS, NUM_FOLDS)) if HPO_METHODS[i] != 'hyperband' else np.zeros((NUM_ITERS_H, NUM_TASKS, NUM_SEEDS, NUM_FOLDS)) for i in range(NUM_METHODS)]
    scores_J = [np.zeros((NUM_ITERS, NUM_TASKS, NUM_SEEDS, NUM_FOLDS)) if HPO_METHODS[i] != 'hyperband' else np.zeros((NUM_ITERS_H, NUM_TASKS, NUM_SEEDS, NUM_FOLDS)) for i in range(NUM_METHODS)]
    names = []
    k=0
    for suite_id in suites:
        with open(f"task_indices/{suite_id}_task_names.json", 'r') as f:
            _names = json.load(f)
        names.extend(_names)

        tasks = np.load(f"task_indices/{suite_id}_task_indices.npy")

        for task_id in tasks:
            for l, seed in enumerate(seeds):
                #print('Seed',seed, 'Suite_id', suite_id, 'Task_id', task_id)
                data = pd.read_csv(f"/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/Results/seed_{seed}/{suite_id}_{task_id}.csv")
                data_H = pd.read_csv(f"/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/Results_H/seed_{seed}/{suite_id}_{task_id}.csv")
                data_H.drop(columns=['param_ind','try_num_iter'], inplace=True)
                data_SMAC = pd.read_csv(f"/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/Results_SMAC/seed_{seed}/{suite_id}_{task_id}.csv")
                data_SMAC.drop(columns=['try_num_iter'], inplace=True)
                #get the len of the data of all folds for one TUNING_STRAT strategy
                LEN_DATA = len(data.loc[(data['try_num_leaves'])])
                LEN_DATA_SMAC = len(data_SMAC.loc[(data_SMAC['try_num_leaves'])])
                LEN_DATA_H = len(data_H.loc[(data_H['try_num_leaves'])])
                data = pd.concat([data, data_H, data_SMAC], ignore_index=True)
                data = data.loc[data['method'] != 'grid_search']
                #data.to_csv('TEST3.csv', index = True)
                if rmse:
                    try_max_depth = data.loc[(data['try_num_leaves'] == False) & (data['joint_tuning_depth_leaves'] == False), 'current_best_test_rmse'].reset_index(drop=True)
                    try_num_leaves = data.loc[data['try_num_leaves'] == True, 'current_best_test_rmse'].reset_index(drop=True)
                    try_joint = data.loc[data['joint_tuning_depth_leaves'] == True, 'current_best_test_rmse'].reset_index(drop=True)
                    df = pd.DataFrame({'try_max_depth_rmse': try_max_depth, 'try_num_leaves_rmse': try_num_leaves,'try_joint_rmse': try_joint})
                    df['method'] = pd.concat([data.loc[0:LEN_DATA-1, 'method'],data_H.loc[0:LEN_DATA_H-1, 'method'],data_SMAC.loc[0:LEN_DATA_SMAC-1, 'method']], ignore_index = True)#data.loc[0:data.shape[0]-1, 'method']
                    df['fold'] = pd.concat([data.loc[0:LEN_DATA-1, 'fold'],data_H.loc[0:LEN_DATA_H-1, 'fold'],data_SMAC.loc[0:LEN_DATA_SMAC-1, 'fold']], ignore_index = True)
                    for i, method in enumerate(HPO_METHODS):
                        for m in FOLDS:
                            print(method, df.loc[(df['method'] == method) & (df['fold'] == m), 'try_num_leaves_rmse'].shape)
                            scores_NL[i][:, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'try_num_leaves_rmse'].values
                            scores_MD[i][:, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'try_max_depth_rmse'].values
                            scores_J[i][:, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'try_joint_rmse'].values
                else:
                    try_max_depth = data.loc[(data['try_num_leaves'] == False) & (data['joint_tuning_depth_leaves'] == False), 'current_best_test_score'].reset_index(drop=True)
                    try_num_leaves = data.loc[data['try_num_leaves'] == True, 'current_best_test_score'].reset_index(drop=True)
                    try_joint = data.loc[data['joint_tuning_depth_leaves'] == True, 'current_best_test_score'].reset_index(drop=True)
                    df = pd.DataFrame({'try_max_depth_score': try_max_depth, 'try_num_leaves_score': try_num_leaves,'try_joint_score': try_joint})
                    df['method'] = pd.concat([data.loc[0:LEN_DATA-1, 'method'],data_H.loc[0:LEN_DATA_H-1, 'method'],data_SMAC.loc[0:LEN_DATA_SMAC-1, 'method']], ignore_index = True)#data.loc[0:data.shape[0]-1, 'method']
                    df['fold'] = pd.concat([data.loc[0:LEN_DATA-1, 'fold'],data_H.loc[0:LEN_DATA_H-1, 'fold'],data_SMAC.loc[0:LEN_DATA_SMAC-1, 'fold']], ignore_index = True)
                    for i, method in enumerate(HPO_METHODS):
                        for m in FOLDS:
                            #print(scores_NL[i][:, k, l, m].shape, method)
                            scores_NL[i][:, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'try_num_leaves_score'].values
                            scores_MD[i][:, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'try_max_depth_score'].values
                            scores_J[i][:, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'try_joint_score'].values
            k += 1
    aggregated_scores_NL = []
    aggregated_scores_MD = []
    aggregated_scores_J  = []
    for i in range(NUM_METHODS):
        aggregated_scores_NL.append(np.mean(scores_NL[i], axis=-1)) #take mean over folds
        aggregated_scores_MD.append(np.mean(scores_MD[i], axis=-1)) #take mean over folds
        aggregated_scores_J.append(np.mean(scores_J[i], axis=-1)) #take mean over folds
    return aggregated_scores_NL, aggregated_scores_MD, aggregated_scores_J, names

def normalize_scores(scores, adtm=False, task_2 = False):
    if task_2:
        num_methods = NUM_METHODS_SUBS
        num_tuning_strat = NUM_TUNING_STRAT_TASK2
    else:
        num_methods = NUM_METHODS
        num_tuning_strat = NUM_TUNING_STRAT
    norm_scores = []
    max = float('-inf')
    if adtm: 
        for i in range(num_tuning_strat):
            scores_max = [np.max(s, axis=(0, 2)) for s in scores[i]] #get maximum across task for each method, 0-axis iterations and 2-axis seeds 
            temp_max = np.maximum.reduce(scores_max)
            max = np.maximum(temp_max,max)
        for i in range(num_tuning_strat):
            for j in range(num_methods):
                if i != 0 and j!=0:
                    all_scores = np.concatenate((all_scores, scores[i][j]),axis = 0)
                else:
                    all_scores = scores[i][j]
        min = np.percentile(all_scores, q=10, axis=(0,2))    
            
        for i in range(num_tuning_strat):            
            norm_scores.append([(s - min[np.newaxis, :, np.newaxis]) / (max[np.newaxis, :, np.newaxis] - min[np.newaxis, :, np.newaxis]) for s in scores[i]])
        affine_map = np.array([min/(max-min),max-min])

    else:
        for i in range(num_tuning_strat):
            for j in range(num_methods):
                if i != 0 and j!=0:
                    all_scores = np.concatenate((all_scores, scores[i][j]),axis = 0)
                else:
                    all_scores = scores[i][j]
        mean_for_norm = np.mean(all_scores, axis = (0,2))
        std_for_norm = np.std(all_scores, axis = (0,2))
        for i in range(num_tuning_strat):
            norm_scores.append([(s - mean_for_norm[np.newaxis, :, np.newaxis]) / std_for_norm[np.newaxis, :, np.newaxis] for s in scores[i]])
        affine_map = np.array([mean_for_norm/std_for_norm,std_for_norm])
    return norm_scores, affine_map

def compare_method(scores,classification = False, RMSE = False, confidence_interval = False, task2 = False, normalization = True):
    'Scores need to be of the form [NL, MD, JOINT]'
    if task2:
        num_methods = NUM_METHODS_SUBS
        num_tuning_strat = NUM_TUNING_STRAT_TASK2
        method_names = HPO_METHODS_SUBS_NAMES 
        method_names = HPO_METHODS_SUBS
    else:
        num_methods = NUM_METHODS
        num_tuning_strat = NUM_TUNING_STRAT
        method_names = HPO_METHODS_NAMES
        methods = HPO_METHODS 
    if classification and RMSE:
        ValueError('Classification and RMSE not possible')
    palette = set_plot_theme()
    #using list because of the different number of iterations 
    norm_scores = []
    mean_norm_scores =[]
    std_norm_scores = []
    lower_lim = []
    upper_lim =  []
    mean_seeds = []
    mean_tasks = []
    avg_var_across_tasks = []
    avg_var_across_seeds = []
    if normalization:
        norm_scores,_= normalize_scores(scores,not RMSE,task2)
    else:
        norm_scores = scores
    for i in range(num_tuning_strat):
        if RMSE:
            mean_norm_scores.append([np.mean(norm_scores_method, axis=(1,2)) for norm_scores_method in norm_scores[i]])
        else:
            mean_norm_scores.append([np.mean(norm_scores_method, axis=(1,2)) for norm_scores_method in norm_scores[i]])
        std_norm_scores.append([np.std(norm_scores_method, axis=(1,2)) for norm_scores_method in norm_scores[i]])
        if confidence_interval:
            lower_lim.append([np.percentile(norm_scores_method, 5, axis=(1,2))for norm_scores_method in norm_scores[i]])
            upper_lim.append([np.percentile(norm_scores_method, 95, axis=(1,2))for norm_scores_method in norm_scores[i]])
    
    fig, axes = plt.subplots(len(RANDOMNESS),num_tuning_strat, figsize=(20,num_tuning_strat*5))
    axes = axes.flatten()
    if not task2:
        NUM_ITERS_H = norm_scores[0][HPO_METHODS.index('hyperband')].shape[0]
    for randomness in range(len(RANDOMNESS)):
        for i in range(num_tuning_strat):
                ax = axes[randomness * (3+task2) + i]  # Select the appropriate subplot
                for method_ind in range(num_methods):
                    #make a case distinction because there are only NUM_ITERS_H datapoints for Hyperband
                    if task2 or methods[method_ind] != 'hyperband':
                        iterations = np.arange(NUM_ITERS)
                    elif methods[method_ind] == 'hyperband':
                        iterations = np.linspace(0,NUM_ITERS-1,NUM_ITERS_H)
                    ax.plot(
                        iterations,
                        np.clip(mean_norm_scores[i][method_ind],-1 if RMSE else 0,1),
                        color=palette[method_ind],
                        marker=MARKERS[method_ind],
                        label=method_names[method_ind],
                        markersize=14,
                        linewidth=2.5,
                        markevery=15
                    )
                    if randomness == 1:  # Randomness due to the seeds
                        mean_tasks = np.mean(norm_scores[i][method_ind], axis=1)
                        avg_var_across_tasks = np.std(mean_tasks, axis = -1)
                        ax.plot(
                            iterations,
                            np.clip(mean_norm_scores[i][method_ind] - avg_var_across_tasks,-1 if RMSE else 0,1),
                            linestyle='--',
                            color=palette[method_ind],
                            alpha=0.6,
                            linewidth=2.5
                        )
                        ax.plot(
                            iterations,
                            np.clip(mean_norm_scores[i][method_ind] + avg_var_across_tasks,-1 if RMSE else 0,1),
                            linestyle='--',
                            color=palette[method_ind],
                            alpha=0.6,
                            linewidth=2.5
                        )

                    elif randomness == 2:  # Randomness due to the tasks
                        mean_seeds = np.mean(norm_scores[i][method_ind], axis=-1)
                        avg_var_across_seeds = np.std(mean_seeds, axis = -1)

                        ax.plot(
                            iterations,
                            np.clip(mean_norm_scores[i][method_ind] - avg_var_across_seeds,-1 if RMSE else 0,1),
                            linestyle='--',
                            color=palette[method_ind],
                            alpha=0.6,
                            linewidth=2.5
                        )
                        ax.plot(
                            iterations,
                            np.clip(mean_norm_scores[i][method_ind] + avg_var_across_seeds,-1 if RMSE else 0,1),
                            linestyle='--',
                            color=palette[method_ind],
                            alpha=0.6,
                            linewidth=2.5
                        )
                    else:
                        if confidence_interval:
                            ax.plot(
                                iterations,
                                lower_lim[i][method_ind],
                                linestyle='--',
                                color=palette[method_ind],
                                alpha=0.6,
                                linewidth=2.5
                            )
                            ax.plot(
                                iterations,
                                upper_lim[i][method_ind],
                                linestyle='--',
                                color=palette[method_ind],
                                alpha=0.6,
                                linewidth=2.5
                            )
                        else:
                            ax.plot(
                                iterations,
                                np.clip(mean_norm_scores[i][method_ind] - std_norm_scores[i][method_ind],-1 if RMSE else 0,1),
                                linestyle='--',
                                color=palette[method_ind],
                                alpha=0.6,
                                linewidth=2.5
                            )
                            ax.plot(
                                iterations,
                                np.clip(mean_norm_scores[i][method_ind] + std_norm_scores[i][method_ind],-1 if RMSE else 0,1),
                                linestyle='--',
                                color=palette[method_ind],
                                alpha=0.6,
                                linewidth=2.5
                            )
                
                for spine in ax.spines.values():
                    spine.set_edgecolor('dimgray')  # Set the desired color here
                    spine.set_linewidth(1)  # Optionally, adjust the thickness

                # Customize grid
                ax.grid(True, color='lightgray', linewidth=0.5)

                # Set the y-axis to have at most 5 ticks
                ax.tick_params(axis='both', which='major', labelsize=28)
                ax.xaxis.set_major_locator(MaxNLocator(6))
                ax.yaxis.set_major_locator(MaxNLocator(5))
                
                title = 'Number Leaves' if i == 0 else 'Max Depth' if i == 1 else 'Joint' if i==2 else 'Num Iter & Joint'
                title += f' and Randomness: '
                title += 'Total' if randomness == 0 else 'Seeds' if randomness == 1 else 'Task'
                ax.set_title(title, fontsize=18)
                ax.set_xlim(0, NUM_ITERS - 1)
                if not RMSE:
                    ax.set_ylim(0, 1)
                else:
                    ax.set_ylim(-0.8, 1)
                    
                for i, ax in enumerate(axes):
                    if i % 3 == 0:  
                        ax.set_ylabel('Average aggregated test score', fontsize=14)
                    if i // 3 == 2: 
                        ax.set_xlabel('Iteration', fontsize=14)
    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=len(method_names), fontsize=16, bbox_to_anchor=(0.5, -0.025)) if classification else fig.legend(lines, labels, loc='lower center', ncol=len(method_names), fontsize=16, bbox_to_anchor=(0.5, -0.025))
    bigtitle = 'Comparison of Methods'
    if not classification:
        bigtitle += ' for Regression Task'
    if RMSE:
        bigtitle += ' using RMSE'
    if classification:
        bigtitle += ' for Classification Tasks'
    if confidence_interval:
        bigtitle += ' with Confindence Interval'
    plt.suptitle(bigtitle, fontsize=30, y=1.001)
    plt.tight_layout()
    file = 'compare_methods'
    if RMSE:
        file += '_RMSE'
    if classification:
        file += '_classficiation'
    if confidence_interval:
        file += '_confindence_interval'
    if task2:
        file +="_Task2"
    plt.savefig(f'plots_pub/{file}.png', bbox_inches='tight')
    plt.show()


def plot_scores_aggregated_tasks_per_tuning_method(scores, METHOD, classification=False, adtm=False, confidence_interval=False, randomness=0, rmse=False, task2 = False,normalization=True):
    if task2 and (METHOD == 'hyperband' or METHOD == 'SMAC'):
        ValueError('In Task 2 we do not consider SMAC or hyperband')
    if task2:
        num_tuning_strat = NUM_TUNING_STRAT_TASK2
        tuning_strat = TUNING_STRAT_TASK2
        method_names = HPO_METHODS_SUBS_NAMES 
        methods = HPO_METHODS_SUBS
    else:
        num_tuning_strat = NUM_TUNING_STRAT
        tuning_strat = TUNING_STRAT
        method_names = HPO_METHODS_NAMES
        methods = HPO_METHODS 
    if classification:
        num_tasks = NUM_CLASS_TASKS
    else: 
        num_tasks = NUM_REGR_TASKS
    if rmse:
        title = 'Aver. RMSE '

    else:
        title = 'Aggr. score '
    if classification:
        title += 'for Classification Tasks '
    else:
        title += 'for Regression Tasks '
    title += f'using {method_names[methods.index(METHOD)]}'
    # if adtm:
    #     title += ' (ADTM)'
    
    # if classification:
    #     title += ' for classification tasks'

    # else:
    #     title += ' for regression tasks'
    # title += f' using the hyperparameter selction method: {METHOD}'
    palette = set_plot_theme()
    plt.figure(figsize=(12, 8))
    method_index = methods.index(METHOD)
    if task2 or methods[method_index] !='hyperband':
        num_iters = NUM_ITERS
    elif methods[method_index] =='hyperband':
        num_iters = scores[0][method_index].shape[0]
    norm_scores = np.zeros((num_tuning_strat,num_iters,num_tasks, NUM_SEEDS)) 
    mean_seeds = np.zeros((num_tuning_strat,num_iters,num_tasks)) 
    mean_tasks = np.zeros((num_tuning_strat,num_iters, NUM_SEEDS)) 
    avg_var_across_tasks = np.zeros((num_tuning_strat,num_iters,num_tasks, NUM_SEEDS)) 
    avg_var_across_seeds = np.zeros((num_tuning_strat,num_iters,num_tasks, NUM_SEEDS)) 
    mean_norm_scores = np.zeros((num_tuning_strat,num_iters)) 
    std_norm_scores  = np.zeros((num_tuning_strat,num_iters)) 
    lower_lim  = np.zeros((num_tuning_strat,num_iters)) 
    upper_lim  = np.zeros((num_tuning_strat,num_iters)) 
  
    for i in range(num_tuning_strat):
        if normalization:
            norm_scores[i,:,:,:],_ = normalize_scores(scores, adtm,task2)[i][method_index]
        else:
            norm_scores[i,:,:,:] = scores[i][method_index]
        mean_norm_scores[i,:] = np.mean(norm_scores[i,:,:,:], axis=(1,2))
        std_norm_scores[i,:] = np.std(norm_scores[i,:,:,:], axis=(1,2))
        if confidence_interval:
            lower_lim[i,:] = np.percentile(norm_scores[i,:,:,:], 5, axis=(1,2))
            upper_lim[i,:] = np.percentile(norm_scores[i,:,:,:], 95, axis=(1,2))
    #print(mean_norm_scores[0,:]-mean_norm_scores[1,:])
    if HPO_METHODS[method_index] != 'hyperband':
        iterations = np.arange(NUM_ITERS)
    else:
        iterations = np.linspace(0,NUM_ITERS-1,num_iters)
    for i in range(num_tuning_strat):
        plt.plot(iterations, np.clip(mean_norm_scores[i,:],-1 if rmse else 0,1), label=tuning_strat[i], color=palette[i], marker=MARKERS[i], markersize=14, linewidth=2.5, markevery=15)
        #print(i, method_index , NAME_TUNING[i])
        if randomness == 1:       # Randomness due to the seeds
            mean_tasks[i,:,:] = np.mean(norm_scores[i,:,:,:], axis=1)
            avg_var_across_tasks[i,:] = np.std(mean_tasks[i,:,:], axis=-1)

            plt.fill_between(iterations, np.clip(mean_norm_scores[i , :] - avg_var_across_tasks[i , :],-1 if rmse else 0,1), np.clip(mean_norm_scores[i][method_index , :] + avg_var_across_tasks[i][method_index , :],-1 if rmse else 0,1), alpha=0.2, color=palette[i])
            plt.plot(iterations, np.clip(mean_norm_scores[i, :] - avg_var_across_tasks[i, :],-1 if rmse else 0,1), linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
            plt.plot(iterations, np.clip(mean_norm_scores[i, :] + avg_var_across_tasks[i, :],-1 if rmse else 0,1), linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
        
        # Randomness due to the tasks
        elif randomness == 2:
            mean_seeds[i,:,:] = np.mean(norm_scores[i,:,:,:], axis=-1)
            avg_var_across_seeds[i] = np.std(mean_seeds[i], axis=-1)
            plt.fill_between(iterations, np.clip(mean_norm_scores[i, :] - avg_var_across_seeds[i, :],-1 if rmse else 0,1), np.clip(mean_norm_scores[i, :] + avg_var_across_seeds[i, :],-1 if rmse else 0,1), alpha=0.2, color=palette[i])
            plt.plot(iterations, np.clip(mean_norm_scores[i, :] - avg_var_across_seeds[i, :],-1 if rmse else 0,1), linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
            plt.plot(iterations, np.clip(mean_norm_scores[i, :] + avg_var_across_seeds[i, :],-1 if rmse else 0,1), linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)

        else:
            if confidence_interval:
                plt.fill_between(iterations, lower_lim[i, :], upper_lim[i, :], hatch='/', alpha=0.2, color=palette[i])
                plt.plot(iterations, lower_lim[i, :], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
                plt.plot(iterations, upper_lim[i, :], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
            
            else:
                plt.fill_between(iterations, np.clip(mean_norm_scores[i, :] - std_norm_scores[i, :],-1 if rmse else 0,1), np.clip(mean_norm_scores[i, :] + std_norm_scores[i, :],-1 if rmse else 0,1), alpha=0.2, color=palette[i])
                plt.plot(iterations, np.clip(mean_norm_scores[i, :] - std_norm_scores[i, :],-1 if rmse else 0,1), linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
                plt.plot(iterations, np.clip(mean_norm_scores[i, :] + std_norm_scores[i, :],-1 if rmse else 0,1), linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)

    for spine in plt.gca().spines.values():
        spine.set_edgecolor('dimgray')  # Set the desired color here
        spine.set_linewidth(1)      # Optionally, adjust the thickness
    
    # Customize grid
    plt.gca().grid(True, color='lightgray', linewidth=0.5)

    # Set the y-axis to have at most 5 ticks
    plt.gca().tick_params(axis='both', which='major', labelsize=28)
    plt.gca().xaxis.set_major_locator(MaxNLocator(6))
    plt.gca().yaxis.set_major_locator(MaxNLocator(5))

    plt.title(title, fontsize=24)
    plt.xlabel('Iteration', fontsize=30)
    plt.ylabel('Average aggregated test score', fontsize=30)
    if rmse or METHOD == 'hyperband':
        plt.legend(loc="center left", ncol=1, bbox_to_anchor=(0.413, 0.8), fontsize=30)

    else:
        plt.legend(loc="center left", ncol=1, bbox_to_anchor=(0.413, 0.2), fontsize=30)
    plt.xlim(0, NUM_ITERS - 1)
    if adtm:
        plt.ylim(0, 1)
    else:
        plt.ylim(-0.5, 1)
    file = "agg_scores"
    file += "_classification" if classification else "_regression"
    if adtm:
        file += "_adtm"

    if confidence_interval:
        file += "_confidence_interval"

    if randomness == 0:
        file += "_total_randomness"

    elif randomness == 1:
        file += "_randomness_seeds"

    elif randomness == 2:
        file += "_randomness_tasks"

    if rmse:
        file += "_rmse"
    file += f"_{METHOD}"
    if task2:
        file +="_Task2"
    plt.savefig(f"plots_pub/{file}.png")
    plt.show()

def get_scores_num_iter():
    scores_regr = [np.zeros((NUM_ITERS, NUM_REGR_TASKS, NUM_SEEDS, NUM_FOLDS)) for i in range(NUM_METHODS_SUBS)]
    scores_class = [np.zeros((NUM_ITERS, NUM_CLASS_TASKS, NUM_SEEDS, NUM_FOLDS)) for i in range(NUM_METHODS_SUBS)]
    rmse_regr = [np.zeros((NUM_ITERS, NUM_REGR_TASKS, NUM_SEEDS, NUM_FOLDS)) for i in range(NUM_METHODS_SUBS)]
    suites = [334,335,336,337]
    k1 = 0
    k2 = 0
    for suite_id in suites:
        tasks = np.load(f"task_indices/{suite_id}_task_indices.npy")

        for task_id in tasks:
            for l, seed in enumerate(seeds):
                #print('Seed',seed, 'Suite_id', suite_id, 'Task_id', task_id)
                data = pd.read_csv(f"/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/Result_Task_2/seed_{seed}/{suite_id}_{task_id}.csv")
                #get the len of the data of all folds for one TUNING_STRAT strategy
                LEN_DATA = len(data.loc[(data['iter'])])
                #print(LEN_DATA)
                if (suite_id == 335) or (suite_id == 336):
                    try_num_iter_score = data.loc[(data['joint_tuning_depth_leaves'] == True) & (data['try_num_iter'] == True), 'current_best_test_score'].reset_index(drop=True)
                    try_num_iter_rmse = data.loc[(data['joint_tuning_depth_leaves'] == True) & (data['try_num_iter'] == True), 'current_best_test_rmse'].reset_index(drop=True)
                    df_rmse = pd.DataFrame({'try_num_iter': try_num_iter_rmse})
                    df_score = pd.DataFrame({'try_num_iter': try_num_iter_score})
                    df_rmse['method'] = data.loc[0:LEN_DATA-1, 'method']
                    df_score['method'] = data.loc[0:LEN_DATA-1, 'method']
                    df_rmse['fold'] = data.loc[0:LEN_DATA-1, 'fold']
                    df_score['fold'] = data.loc[0:LEN_DATA-1, 'fold']
                    for i, method in enumerate(HPO_METHODS_SUBS):
                        for m in FOLDS:
                            scores_regr[i][:,k2,l,m] = df_score.loc[(df_score['method'] == method) & (df_score['fold'] == m), 'try_num_iter'].values
                            rmse_regr[i][:,k2,l,m] = df_rmse.loc[(df_rmse['method'] == method) & (df_rmse['fold'] == m), 'try_num_iter'].values
                else:
                    try_num_iter_score = data.loc[(data['joint_tuning_depth_leaves'] == True) & (data['try_num_iter'] == True), 'current_best_test_score'].reset_index(drop=True)
                    df_score = pd.DataFrame({'try_num_iter': try_num_iter_score})
                    df_score['method'] = data.loc[0:LEN_DATA-1, 'method']
                    df_score['fold'] = data.loc[0:LEN_DATA-1, 'fold']
                    for i, method in enumerate(HPO_METHODS_SUBS):
                        for m in FOLDS:
                            #print(df_score)
                            scores_class[i][:,k1,l,m] = df_score.loc[(df_score['method'] == method) & (df_score['fold'] == m), 'try_num_iter'].values
            if (suite_id == 335) or (suite_id == 336):
                k2 += 1
            else:
                k1 += 1
    aggregated_scores_regr = []
    aggregated_scores_class = []
    aggregated_rmse  = []
    for i in range(NUM_METHODS_SUBS):
        aggregated_scores_regr.append(np.mean(scores_regr[i], axis=-1)) #take mean over folds
        aggregated_scores_class.append(np.mean(scores_class[i], axis=-1)) #take mean over folds
        aggregated_rmse.append(np.mean(rmse_regr[i], axis=-1)) #take mean over folds
    return aggregated_scores_regr, aggregated_scores_class, aggregated_rmse

def plot_scores_per_task_tuning_methods(scores, METHOD,names, classification=False, adtm=False, confidence_interval=False, rmse=False,log_loss = False, normalization = True, Task2 = False, Borders = False):
    if Task2:
        name_tuning = ['Num Leaves','Max Depth','Joint','Num Iter']
    elif not Task2:
        name_tuning = ['Num Leaves','Max Depth','Joint']
    elif METHOD == 'grid_search':
        name_tuning = ['num_leaves','max_depth']
    if rmse:
        title = 'Average RMSE'
    elif log_loss:
        title = 'Average Log Loss'
    else:
        title = 'Average score'
    if not normalization:
        title += ' (not normalized)'
    if adtm:
        title += ' (ADTM)'

    if classification:
        num_tasks = NUM_CLASS_TASKS
        title += ' per iteration for each classification task'

    else:
        num_tasks = NUM_REGR_TASKS
        title += ' per iteration for each regression task'
    title += f' using the hyperparameter selction method: {METHOD}'
    num_cols = 4
    num_rows = (num_tasks + num_cols - 1) // num_cols
    method_index = HPO_METHODS.index(METHOD)
    palette = set_plot_theme()
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4))
    axes = axes.flatten()

    mean_norm_scores =[]# [np.zeros((NUM_METHODS,NUM_ITERS,num_tasks)),np.zeros((NUM_METHODS,NUM_ITERS,num_tasks))
                       # ,np.zeros((NUM_METHODS_SUBS,NUM_ITERS,num_tasks))]
    std_norm_scores  = []#[np.zeros((NUM_METHODS,NUM_ITERS,num_tasks)),np.zeros((NUM_METHODS,NUM_ITERS,num_tasks))
                        #,np.zeros((NUM_METHODS_SUBS,NUM_ITERS,num_tasks))]
    lower_lim  = []#[np.zeros((NUM_METHODS,NUM_ITERS,num_tasks)),np.zeros((NUM_METHODS,NUM_ITERS,num_tasks))
                    #    ,np.zeros((NUM_METHODS_SUBS,NUM_ITERS,num_tasks))]
    upper_lim  = []#[np.zeros((NUM_METHODS,NUM_ITERS,num_tasks)),np.zeros((NUM_METHODS,NUM_ITERS,num_tasks))
                   #     ,np.zeros((NUM_METHODS_SUBS,NUM_ITERS,num_tasks))]
    if normalization:
        scores,_ = normalize_scores(scores, adtm) 
    for i in range(len(name_tuning)):
        get_nested_shape(scores[i])
        mean_norm_scores.append(np.mean(scores[i][method_index], axis=-1))
        std_norm_scores.append( np.std(scores[i][method_index], axis=-1))#[i] = np.std(scores[i], axis=-1)
        if confidence_interval:
            lower_lim.append(np.percentile(scores[i][method_index], 95, axis=-1))#[i] = np.percentile(scores[i], 5, axis=-1)
            upper_lim.append(np.percentile(scores[i][method_index], 95, axis=-1))#[i] = np.percentile(scores[i], 95, axis=-1)
    for i in range(len(name_tuning)):
        for k in range(num_tasks):
            ax = axes[k]
            ax.plot(
                np.arange(NUM_ITERS),
                np.clip(mean_norm_scores[i][:, k],0-rmse - log_loss,1),
                label=name_tuning[i],
                color=palette[i],
                marker=MARKERS[i],
                markersize=14,
                linewidth=2.5,
                markevery=20
            )
            if Borders:
                if confidence_interval:
                    ax.plot(np.arange(NUM_ITERS), np.clip(lower_lim[i][:, k],0-rmse-log_loss,1), linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
                    ax.plot(np.arange(NUM_ITERS), np.clip(upper_lim[i][:, k],0-rmse-log_loss,1), linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)

                else:
                    ax.plot(np.arange(NUM_ITERS), np.clip(mean_norm_scores[i][ :, k]
                                - std_norm_scores[i][:, k],0-rmse-log_loss,1), linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
                    ax.plot(np.arange(NUM_ITERS), np.clip(mean_norm_scores[i][:, k]
                                + std_norm_scores[i][:, k],0-rmse-log_loss,1), linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)

            if len(names[k]) > 24:
                names[k] = names[k][:24]

            if not classification:
                if k == 3:
                    names[k] = names[k][:21]
                elif k == 9:
                    names[k] = names[k][:16]

            ax.set_title(names[k], fontsize=28)

            # Customize grid
            ax.grid(True, color='lightgray', linewidth=0.5)

            # Hide x labels and tick labels for all but the bottom row
            if k < (num_rows - 1) * num_cols:
                if not (classification and k == 19):
                    ax.set_xticklabels([])

            # Hide y labels and tick labels for all but the leftmost column
            if k % num_cols != 0:
                    ax.set_yticklabels([])

            for spine in ax.spines.values():
                spine.set_edgecolor('dimgray')  # Set the desired color here
                spine.set_linewidth(1)      # Optionally, adjust the thickness

            # Increase the size of remaining tick labels
            ax.tick_params(axis='both', which='major', labelsize=28)
            ax.xaxis.set_major_locator(MaxNLocator(6))
            ax.yaxis.set_major_locator(MaxNLocator(5))

            ax.set_xlim(0, NUM_ITERS - 1)
            if adtm:
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(-1, 1)


    # Make each subplot (axes) quadratic (equal width and height)
    for ax in axes[:num_tasks]:
        pos = ax.get_position()
        width = pos.width
        height = pos.height
        max_dim = max(width, height)
        # Center the axes and set both width and height to max_dim
        new_pos = [pos.x0, pos.y0, max_dim, max_dim]
        ax.set_position(new_pos)

    # Add legend
    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        lines, labels,
        loc='upper center',
        ncol=len(name_tuning),
        fontsize=30,
        bbox_to_anchor=(0.5, 0.959 if classification else 0.968)
    )

    # Set a separate y-axis label for each subplot
    for ax in axes[:num_tasks]:
        ax.set_ylabel('Average test score', fontsize=18)

    # Set global x-label
    fig.text(0.5, 0.045, 'Iteration', ha='center', va='center', fontsize=30)

    # Adjust layout to leave space for legend and labels
    #plt.tight_layout(rect=[0, 0, 1, 0.85])
    if classification:
        plt.subplots_adjust(top=0.894, bottom=0.08, left=0.1, hspace=0.2, wspace=0.2)
    else:
        plt.subplots_adjust(top=0.924, bottom=0.07, left=0.1, hspace=0.2, wspace=0.2)

    # Set the title
    fig.suptitle(title, fontsize=24)
    file = "scores_per_task"
    file += "_classification" if classification else "_regression"
    if adtm:
        file += "_adtm"
    if confidence_interval:
        file += "_confidence_interval"
    if rmse:
        file += "_rmse"
    if log_loss:
        file += "_log_loss"
    file += f"_{METHOD}"

    plt.savefig(f"plots_pub/{file}.png")
    plt.show()


def create_scores_dict_modified(classification=False, rmse=False, current_best = False, log_loss = False):
    if current_best and not (log_loss):
        rmse_value = 'current_best_test_rmse'
        score_value = 'current_best_test_score'
    elif (not current_best) and log_loss:
        rmse_value = 'test_rmse'
        score_value = 'test_log_loss'
    elif current_best and log_loss:
        rmse_value = 'current_best_test_rmse'
        score_value = 'current_best_test_log_loss'
    else:
        rmse_value = 'test_rmse'
        score_value = 'test_score'
    if classification:
        NUM_TASKS = NUM_CLASS_TASKS
        suites = [334, 337]
    
    else:
        NUM_TASKS = NUM_REGR_TASKS
        suites = [335, 336] 
    #get here the number of iterations for hyperband
    data_H = pd.read_csv("/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/Results_H/seed_27225/334_361110.csv")    
    NUM_ITERS_H = len(data_H.loc[(data_H['fold'] == 0) & (data_H['joint_tuning_depth_leaves'] == True)])
    if NUM_ITERS_H > NUM_ITERS:
                    ValueError('There are too much iterations in the Hyperband method.')
    scores_NL = [np.zeros((NUM_ITERS, NUM_TASKS, NUM_SEEDS, NUM_FOLDS)) if HPO_METHODS[i] != 'hyperband' else np.zeros((NUM_ITERS_H, NUM_TASKS, NUM_SEEDS, NUM_FOLDS)) for i in range(NUM_METHODS)]
    for i in range(NUM_METHODS):
        print('The following shape for',i, scores_NL[i].shape)
    scores_MD = [np.zeros((NUM_ITERS, NUM_TASKS, NUM_SEEDS, NUM_FOLDS)) if HPO_METHODS[i] != 'hyperband' else np.zeros((NUM_ITERS_H, NUM_TASKS, NUM_SEEDS, NUM_FOLDS)) for i in range(NUM_METHODS)]
    scores_J = [np.zeros((NUM_ITERS, NUM_TASKS, NUM_SEEDS, NUM_FOLDS)) if HPO_METHODS[i] != 'hyperband' else np.zeros((NUM_ITERS_H, NUM_TASKS, NUM_SEEDS, NUM_FOLDS)) for i in range(NUM_METHODS)]
    names = []
    k=0
    for suite_id in suites:
        with open(f"task_indices/{suite_id}_task_names.json", 'r') as f:
            _names = json.load(f)
        names.extend(_names)

        tasks = np.load(f"task_indices/{suite_id}_task_indices.npy")

        for task_id in tasks:
            for l, seed in enumerate(seeds):
                #print('Seed',seed, 'Suite_id', suite_id, 'Task_id', task_id)
                data = pd.read_csv(f"/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/Results/seed_{seed}/{suite_id}_{task_id}.csv")
                data_H = pd.read_csv(f"/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/Results_H/seed_{seed}/{suite_id}_{task_id}.csv")
                data_H.drop(columns=['param_ind','try_num_iter'], inplace=True)
                data_SMAC = pd.read_csv(f"/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/Results_SMAC/seed_{seed}/{suite_id}_{task_id}.csv")
                data_SMAC.drop(columns=['try_num_iter'], inplace=True)
                #get the len of the data of all folds for one TUNING_STRAT strategy
                LEN_DATA = len(data.loc[(data['try_num_leaves'])])
                LEN_DATA_SMAC = len(data_SMAC.loc[(data_SMAC['try_num_leaves'])])
                LEN_DATA_H = len(data_H.loc[(data_H['try_num_leaves'])])
                data = pd.concat([data, data_H, data_SMAC], ignore_index=True)
                data = data.loc[data['method'] != 'grid_search']
                #data.to_csv('TEST3.csv', index = True)
                if rmse:
                    try_max_depth = data.loc[(data['try_num_leaves'] == False) & (data['joint_tuning_depth_leaves'] == False), rmse_value].reset_index(drop=True)
                    try_num_leaves = data.loc[data['try_num_leaves'] == True, rmse_value].reset_index(drop=True)
                    try_joint = data.loc[data['joint_tuning_depth_leaves'] == True, rmse_value].reset_index(drop=True)
                    df = pd.DataFrame({'try_max_depth_rmse': try_max_depth, 'try_num_leaves_rmse': try_num_leaves,'try_joint_rmse': try_joint})
                    df['method'] = pd.concat([data.loc[0:LEN_DATA-1, 'method'],data_H.loc[0:LEN_DATA_H-1, 'method'],data_SMAC.loc[0:LEN_DATA_SMAC-1, 'method']], ignore_index = True)#data.loc[0:data.shape[0]-1, 'method']
                    df['fold'] = pd.concat([data.loc[0:LEN_DATA-1, 'fold'],data_H.loc[0:LEN_DATA_H-1, 'fold'],data_SMAC.loc[0:LEN_DATA_SMAC-1, 'fold']], ignore_index = True)
                    for i, method in enumerate(HPO_METHODS):
                        for m in FOLDS:
                            print(method, df.loc[(df['method'] == method) & (df['fold'] == m), 'try_num_leaves_rmse'].shape)
                            scores_NL[i][:, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'try_num_leaves_rmse'].values
                            scores_MD[i][:, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'try_max_depth_rmse'].values
                            scores_J[i][:, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'try_joint_rmse'].values
                else:
                    try_max_depth = data.loc[(data['try_num_leaves'] == False) & (data['joint_tuning_depth_leaves'] == False), score_value].reset_index(drop=True)
                    try_num_leaves = data.loc[data['try_num_leaves'] == True, score_value].reset_index(drop=True)
                    try_joint = data.loc[data['joint_tuning_depth_leaves'] == True, score_value].reset_index(drop=True)
                    df = pd.DataFrame({'try_max_depth_score': try_max_depth, 'try_num_leaves_score': try_num_leaves,'try_joint_score': try_joint})
                    df['method'] = pd.concat([data.loc[0:LEN_DATA-1, 'method'],data_H.loc[0:LEN_DATA_H-1, 'method'],data_SMAC.loc[0:LEN_DATA_SMAC-1, 'method']], ignore_index = True)#data.loc[0:data.shape[0]-1, 'method']
                    df['fold'] = pd.concat([data.loc[0:LEN_DATA-1, 'fold'],data_H.loc[0:LEN_DATA_H-1, 'fold'],data_SMAC.loc[0:LEN_DATA_SMAC-1, 'fold']], ignore_index = True)
                    for i, method in enumerate(HPO_METHODS):
                        for m in FOLDS:
                            #print(scores_NL[i][:, k, l, m].shape, method)
                            scores_NL[i][:, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'try_num_leaves_score'].values
                            scores_MD[i][:, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'try_max_depth_score'].values
                            scores_J[i][:, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'try_joint_score'].values
            k += 1
    aggregated_scores_NL = []
    aggregated_scores_MD = []
    aggregated_scores_J  = []
    for i in range(NUM_METHODS):
        aggregated_scores_NL.append(np.mean(scores_NL[i], axis=-1)) #take mean over folds
        aggregated_scores_MD.append(np.mean(scores_MD[i], axis=-1)) #take mean over folds
        aggregated_scores_J.append(np.mean(scores_J[i], axis=-1)) #take mean over folds
    return aggregated_scores_NL, aggregated_scores_MD, aggregated_scores_J, names


def compare_std(scores,TUNING_SETTING,names, classification = False, adtm = False, log_loss = False, rmse = False, normalization = False):
    if rmse:
        title = 'Standard Deviation of RMSE'
    elif log_loss:
        title = 'Standard Deviation of Log Loss'
    else:
        title = 'Standard Deviation of Scores'
    if not normalization:
        title += ' (not normalized)'
    if adtm:
        title += ' (ADTM)'

    if classification:
        num_tasks = NUM_CLASS_TASKS
        title += ' per iteration for each classification task'

    else:
        num_tasks = NUM_REGR_TASKS
        title += ' per iteration for each regression task'
    title += f' using the tuning setting {TUNING_SETTING}'
    num_cols = 4
    num_rows = (num_tasks + num_cols - 1) // num_cols
    tuning_index = TUNING_STRAT.index(TUNING_SETTING)
    palette = set_plot_theme()
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4))
    axes = axes.flatten()

    std_norm_scores  = []#[np.zeros((NUM_METHODS,NUM_ITERS,num_tasks)),np.zeros((NUM_METHODS,NUM_ITERS,num_tasks))
                        #,np.zeros((NUM_METHODS_SUBS,NUM_ITERS,num_tasks))]
    if normalization:
        scores,_ = normalize_scores(scores, adtm) 
    methods = np.array([0,1,2,4])
    for i in methods:
        print(i)
        std_norm_scores.append(np.std(scores[tuning_index][i], axis=-1))#[i] = np.std(scores[i], axis=-1)
    for i in range(NUM_METHODS_SUBS):
        for k in range(num_tasks):
            ax = axes[k]
            ax.plot(
                np.arange(NUM_ITERS),
                std_norm_scores[i][:, k],
                label=HPO_METHODS[i+1 if i ==3 else i],
                color=palette[i],
                marker=MARKERS[i],
                markersize=14,
                linewidth=2.5,
                markevery=20
            )                

            if len(names[k]) > 24:
                names[k] = names[k][:24]

            if not classification:
                if k == 3:
                    names[k] = names[k][:21]
                elif k == 9:
                    names[k] = names[k][:16]

            ax.set_title(names[k], fontsize=28)

            # Customize grid
            ax.grid(True, color='lightgray', linewidth=0.5)

            # Hide x labels and tick labels for all but the bottom row
            if k < (num_rows - 1) * num_cols:
                if not (classification and k == 19):
                    ax.set_xticklabels([])

            # Hide y labels and tick labels for all but the leftmost column
            if k % num_cols != 0:
                    ax.set_yticklabels([])

            for spine in ax.spines.values():
                spine.set_edgecolor('dimgray')  # Set the desired color here
                spine.set_linewidth(1)      # Optionally, adjust the thickness

            # Increase the size of remaining tick labels
            ax.tick_params(axis='both', which='major', labelsize=28)
            ax.xaxis.set_major_locator(MaxNLocator(6))
            ax.yaxis.set_major_locator(MaxNLocator(5))

            ax.set_xlim(0, NUM_ITERS - 1)
            # print(np.max(std_norm_scores[i][:, k]))
            # if np.max(std_norm_scores[i][:, k]) > temp_max[k]:
            #     temp_max[k] = np.max(std_norm_scores[i][:, k])
            # ax.set_ylim(0,temp_max[k])


    # Make each subplot (axes) quadratic (equal width and height)
    for ax in axes[:num_tasks]:
        pos = ax.get_position()
        width = pos.width
        height = pos.height
        max_dim = max(width, height)
        # Center the axes and set both width and height to max_dim
        new_pos = [pos.x0, pos.y0, max_dim, max_dim]
        ax.set_position(new_pos)

    # Add legend
    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        lines, labels,
        loc='upper center',
        ncol=len(HPO_METHODS_SUBS) if classification else len(HPO_METHODS_SUBS),
        fontsize=30,
        bbox_to_anchor=(0.5, 0.959 if classification else 0.968)
    )

    # Set a separate y-axis label for each subplot
    for ax in axes[:num_tasks]:
        ax.set_ylabel('Standard Deviation', fontsize=18)

    # Set global x-label
    fig.text(0.5, 0.045, 'Iteration', ha='center', va='center', fontsize=30)

    # Adjust layout to leave space for legend and labels
    #plt.tight_layout(rect=[0, 0, 1, 0.85])
    if classification:
        plt.subplots_adjust(top=0.894, bottom=0.08, left=0.1, hspace=0.2, wspace=0.2)
    else:
        plt.subplots_adjust(top=0.924, bottom=0.07, left=0.1, hspace=0.2, wspace=0.2)

    # Set the title
    fig.suptitle(title, fontsize=24)
    file = "std_per_task"
    file += "_classification" if classification else "_regression"
    if adtm:
        file += "_adtm"
    if rmse:
        file += "_rmse"
    if log_loss:
        file += "_log_loss"
    file += f"_{TUNING_SETTING}"

    plt.savefig(f"plots_pub/{file}.png")
    plt.show()

