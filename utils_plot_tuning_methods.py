"""Some functions copied from Ioana Iacobici https://github.com/iiacobici and modified by Floris Koster https://github.com/fl0risk"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


seeds = [27225, 34326,92161, 99246, 108473, 117739,  235053, 257787, 
        89389, 443417, 572858, 620176, 671487, 710570, 773246, 936518] #32244,147316, 777646, 778572

NUM_SEEDS = len(seeds)
NUM_TASKS = 59 #can be removed after testing
NUM_REGR_TASKS = 36 #can be removed after testing
NUM_CLASS_TASKS = 23 #can be removed after testing
NUM_ITERS = 135
METHODS = ['grid_search', 'random_search', 'tpe', 'gp_bo']
METHODS_JOINT = ['random_search', 'tpe', 'gp_bo']
NAME_METHODS = ['Fixed grid search', 'Random grid search', 'TPE', 'GP-BO']
NAME_TUNING = ['num_leaves','max_depth','joint']
NUM_TUNING = len(NAME_TUNING)
NUM_METHODS = len(METHODS)
NUM_METHODS_JOINT = len(METHODS_JOINT)
FOLDS = [0, 1, 2, 3, 4]
NUM_FOLDS = len(FOLDS)
MARKERS = ["o", "*", "^", "s"]


def set_plot_theme():
    # Set Seaborn theme
    sns.set_theme(context="paper", style="white")
    palette = ['lime', 'darkorange', 'fuchsia', 'deepskyblue']

    return palette


def create_scores_dict(classification=False, rmse=False):
    if classification:
        NUM_TASKS = NUM_CLASS_TASKS
        suites = [334, 337]
    
    else:
        NUM_TASKS = NUM_REGR_TASKS
        suites = [335, 336] 
    
    scores_NL = np.zeros((NUM_METHODS, NUM_ITERS, NUM_TASKS, NUM_SEEDS, NUM_FOLDS))
    scores_MD = np.zeros((NUM_METHODS, NUM_ITERS, NUM_TASKS, NUM_SEEDS, NUM_FOLDS))
    scores_J = np.zeros((NUM_METHODS_JOINT, NUM_ITERS, NUM_TASKS, NUM_SEEDS, NUM_FOLDS))
    k = 0
    names = []
    
    for suite_id in suites:
        with open(f"task_indices/{suite_id}_task_names.json", 'r') as f:
            _names = json.load(f)
        names.extend(_names)

        tasks = np.load(f"task_indices/{suite_id}_task_indices.npy")

        for task_id in tasks:
            for l, seed in enumerate(seeds):
                data = pd.read_csv(f"/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/Results/seed_{seed}/{suite_id}_{task_id}.csv")
                #data['current_best_test_score'] = data['current_best_test_score'].clip(0, 1)
                if rmse:
                    try_max_depth = data.loc[(data['try_num_leaves'] == False) & (data['joint_tuning_depth_leaves'] == False), 'current_best_test_rmse'].reset_index(drop=True)
                    try_num_leaves = data.loc[data['try_num_leaves'] == True, 'current_best_test_rmse'].reset_index(drop=True)
                    try_joint = data.loc[data['joint_tuning_depth_leaves'] == True, 'current_best_test_rmse'].reset_index(drop=True)
                    df = pd.DataFrame({'try_max_depth_rmse': try_max_depth, 'try_num_leaves_rmse': try_num_leaves})
                    df_joint = pd.DataFrame({'try_joint_rmse': try_joint})
                    df['method'] = data.loc[0:data.shape[0]-1, 'method']
                    df['fold'] = data.loc[0:data.shape[0]-1, 'fold']
                    df_joint['method'] = data.loc[(data['method']!= 'grid_search'), 'method'].reset_index(drop=True)
                    df_joint['fold'] = data.loc[(data['method']!= 'grid_search'), 'fold'].reset_index(drop=True)
                    for i, method in enumerate(METHODS):
                        for m in FOLDS:
                            scores_NL[i, :, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'try_num_leaves_rmse'].values
                            scores_MD[i, :, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'try_max_depth_rmse'].values
                    for i, method in enumerate(METHODS_JOINT):
                        for m in FOLDS:
                            scores_J[i, :, k, l, m] = df_joint.loc[(df_joint['method'] == method) & (df_joint['fold'] == m), 'try_joint_rmse'].values
                else:
                    try_max_depth = data.loc[(data['try_num_leaves'] == False) & (data['joint_tuning_depth_leaves'] == False), 'current_best_test_score'].reset_index(drop=True)
                    try_num_leaves = data.loc[data['try_num_leaves'] == True, 'current_best_test_score'].reset_index(drop=True)
                    try_joint = data.loc[data['joint_tuning_depth_leaves'] == True, 'current_best_test_score'].reset_index(drop=True)
                    df = pd.DataFrame({'try_max_depth_score': try_max_depth, 'try_num_leaves_score': try_num_leaves})
                    df_joint = pd.DataFrame({'try_joint_score': try_joint})
                    df['method'] = data.loc[0:data.shape[0]-1, 'method']
                    df['fold'] = data.loc[0:data.shape[0]-1, 'fold']
                    df_joint['method'] = data.loc[(data['method']!= 'grid_search'), 'method'].reset_index(drop=True)
                    df_joint['fold'] = data.loc[(data['method']!= 'grid_search'), 'fold'].reset_index(drop=True)                    
                    for i, method in enumerate(METHODS):
                        for m in FOLDS:
                            scores_NL[i, :, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'try_num_leaves_score'].values
                            scores_MD[i, :, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'try_max_depth_score'].values
                    for i, method in enumerate(METHODS_JOINT):
                        for m in FOLDS:
                            scores_J[i, :, k, l, m] = df_joint.loc[(df_joint['method'] == method) & (df_joint['fold'] == m), 'try_joint_score'].values
            k += 1
    aggregated_scores_NL = np.mean(scores_NL, axis=-1) #take mean over folds
    aggregated_scores_MD = np.mean(scores_MD, axis=-1) #take mean over folds
    aggregated_scores_J = np.mean(scores_J, axis=-1) #take mean over folds
    if not rmse :
        aggregated_scores_NL = aggregated_scores_NL.clip(0,1)
        aggregated_scores_MD = aggregated_scores_MD.clip(0,1)
        aggregated_scores_J = aggregated_scores_J.clip(0,1)
    return aggregated_scores_NL, aggregated_scores_MD, aggregated_scores_J, names

def normalize_scores(scores, adtm=False):
    if adtm: 
        max = np.max(scores, axis=(0, 1, 3)) #gets maximum for each task
        min = np.percentile(scores, 10, axis=(0, 1, 3))

        norm_scores = (scores - min[np.newaxis, np.newaxis, :, np.newaxis]) / (max[np.newaxis, np.newaxis, :, np.newaxis] - min[np.newaxis, np.newaxis, :, np.newaxis])
        norm_scores = np.clip(norm_scores, 0, 1)

    else:
        mean_for_norm = np.mean(scores, axis=(0, 1, 3))
        std_for_norm = np.std(scores, axis=(0, 1, 3))

        norm_scores = (scores - mean_for_norm[np.newaxis, np.newaxis, :, np.newaxis]) / std_for_norm[np.newaxis, np.newaxis, :, np.newaxis]

    return norm_scores


def plot_scores_per_task_tuning_methods(scores, METHOD,names, classification=False, adtm=False, confidence_interval=False, rmse=False):
    name_tuning = NAME_TUNING
    if METHOD == 'grid_search':
        name_tuning = ['num_leaves','max_depth']
    if rmse:
        title = 'Average RMSE'

    else:
        title = 'Average score'

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
    method_index = METHODS.index(METHOD)
    palette = set_plot_theme()
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4))
    axes = axes.flatten()

    mean_norm_scores = [np.zeros((NUM_METHODS,NUM_ITERS,num_tasks)),np.zeros((NUM_METHODS,NUM_ITERS,num_tasks))
                        ,np.zeros((NUM_METHODS_JOINT,NUM_ITERS,num_tasks))]
    std_norm_scores  = [np.zeros((NUM_METHODS,NUM_ITERS,num_tasks)),np.zeros((NUM_METHODS,NUM_ITERS,num_tasks))
                        ,np.zeros((NUM_METHODS_JOINT,NUM_ITERS,num_tasks))]
    lower_lim  = [np.zeros((NUM_METHODS,NUM_ITERS,num_tasks)),np.zeros((NUM_METHODS,NUM_ITERS,num_tasks))
                        ,np.zeros((NUM_METHODS_JOINT,NUM_ITERS,num_tasks))]
    upper_lim  = [np.zeros((NUM_METHODS,NUM_ITERS,num_tasks)),np.zeros((NUM_METHODS,NUM_ITERS,num_tasks))
                        ,np.zeros((NUM_METHODS_JOINT,NUM_ITERS,num_tasks))]
    helper = [name =='joint' for name in name_tuning] #used since for joint tuning there is one method less
    for i in range(len(name_tuning)):
        scores[i] = normalize_scores(scores[i], adtm)
        mean_norm_scores[i] = np.mean(scores[i], axis=-1)
        std_norm_scores[i] = np.std(scores[i], axis=-1)
        if confidence_interval:
            lower_lim[i] = np.percentile(scores[i], 5, axis=-1)
            upper_lim[i] = np.percentile(scores[i], 95, axis=-1)
    for i in range(len(name_tuning)):
        for k in range(num_tasks):
            ax = axes[k]
            ax.plot(
                np.arange(NUM_ITERS),
                mean_norm_scores[i][method_index -helper[i], :, k],
                label=name_tuning[i],
                color=palette[i],
                marker=MARKERS[i],
                markersize=14,
                linewidth=2.5,
                markevery=20
            )
            if confidence_interval:
                ax.fill_between(np.arange(NUM_ITERS), lower_lim[i][method_index -helper[i], :, k], upper_lim[i][method_index -helper[i], :, k], hatch='/', alpha=0.2, color=palette[i])
                ax.plot(np.arange(NUM_ITERS), lower_lim[i][method_index -helper[i], :, k], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
                ax.plot(np.arange(NUM_ITERS), upper_lim[i][method_index -helper[i], :, k], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)

            else:
                ax.fill_between(np.arange(NUM_ITERS), mean_norm_scores[i][method_index -helper[i], :, k]
                                    - std_norm_scores[i][method_index -helper[i], :, k], mean_norm_scores[i][method_index -helper[i], :, k] + std_norm_scores[i][method_index -helper[i], :, k], alpha=0.2, color=palette[i])
                ax.plot(np.arange(NUM_ITERS), mean_norm_scores[i][method_index -helper[i], :, k]
                            - std_norm_scores[i][method_index -helper[i], :, k], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
                ax.plot(np.arange(NUM_ITERS), mean_norm_scores[i][method_index -helper[i], :, k]
                            + std_norm_scores[i][method_index -helper[i], :, k], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)

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

    # Remove any empty subplots
    for a in range(num_tasks, num_rows * num_cols):
        fig.delaxes(axes[a])

    fig.suptitle(title, fontsize=32)
    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', ncol=len(name_tuning), fontsize=30, bbox_to_anchor=(0.5, 0.959)) if classification else fig.legend(lines, labels, loc='upper center', ncol=len(METHODS), fontsize=30, bbox_to_anchor=(0.5, 0.968))#0.953
    fig.text(0.5, 0.045, 'Iteration', ha='center', va='center', fontsize=30)
    fig.text(0.04, 0.5, 'Average test score', ha='center', va='center', rotation='vertical', fontsize=30)

    plt.tight_layout(rect=[0, 0, 1, 0.85]) 
    plt.subplots_adjust(top=0.894, bottom=0.08, left=0.1, hspace=0.2, wspace=0.2) if classification else plt.subplots_adjust(top=0.924, bottom=0.07, left=0.1, hspace=0.2, wspace=0.2)#top=0.907
    file = "scores_per_task"
    file += "_classification" if classification else "_regression"
    if adtm:
        file += "_adtm"

    if confidence_interval:
        file += "_confidence_interval"

    if rmse:
        file += "_rmse"
    file += f"_{METHOD}"

    plt.savefig(f"plots/{file}.png")
    plt.show()

def plot_scores_aggregated_tasks_per_tuning_method(scores, METHOD, classification=False, adtm=False, confidence_interval=False, randomness=0, rmse=False):
    name_tuning = NAME_TUNING
    if METHOD == 'grid_search':
        name_tuning = ['num_leaves','max_depth']
    if rmse:
        title = 'Average RMSE'

    else:
        title = 'Aggregated score'

    if adtm:
        title += ' (ADTM)'
    
    if classification:
        title += ' for classification tasks'

    else:
        title += ' for regression tasks'
    title += f' using the hyperparameter selction method: {METHOD}'
    palette = set_plot_theme()
    plt.figure(figsize=(12, 8))
    method_index = METHODS.index(METHOD)
    helper = [name =='joint' for name in name_tuning] #used since for joint tuning there is one method less
    
    norm_scores = [np.zeros((NUM_METHODS,NUM_ITERS)),np.zeros((NUM_METHODS,NUM_ITERS))
                        ,np.zeros((NUM_METHODS_JOINT,NUM_ITERS))]
    mean_seeds = [np.zeros((NUM_METHODS,NUM_ITERS,NUM_SEEDS)),np.zeros((NUM_METHODS,NUM_ITERS,NUM_SEEDS))
                        ,np.zeros((NUM_METHODS_JOINT,NUM_ITERS,NUM_SEEDS))]
    mean_tasks = [np.zeros((NUM_METHODS,NUM_ITERS,NUM_SEEDS)),np.zeros((NUM_METHODS,NUM_ITERS,NUM_SEEDS))
                        ,np.zeros((NUM_METHODS_JOINT,NUM_ITERS,NUM_SEEDS))]
    avg_var_across_tasks = [np.zeros((NUM_METHODS,NUM_ITERS)),np.zeros((NUM_METHODS,NUM_ITERS))
                        ,np.zeros((NUM_METHODS_JOINT,NUM_ITERS))]
    avg_var_across_seeds = [np.zeros((NUM_METHODS,NUM_ITERS)),np.zeros((NUM_METHODS,NUM_ITERS))
                        ,np.zeros((NUM_METHODS_JOINT,NUM_ITERS))]
    mean_norm_scores = [np.zeros((NUM_METHODS,NUM_ITERS)),np.zeros((NUM_METHODS,NUM_ITERS))
                        ,np.zeros((NUM_METHODS_JOINT,NUM_ITERS))]
    std_norm_scores  = [np.zeros((NUM_METHODS,NUM_ITERS)),np.zeros((NUM_METHODS,NUM_ITERS))
                        ,np.zeros((NUM_METHODS_JOINT,NUM_ITERS))]
    lower_lim  = [np.zeros((NUM_METHODS,NUM_ITERS)),np.zeros((NUM_METHODS,NUM_ITERS))
                        ,np.zeros((NUM_METHODS_JOINT,NUM_ITERS))]
    upper_lim  = [np.zeros((NUM_METHODS,NUM_ITERS)),np.zeros((NUM_METHODS,NUM_ITERS))
                        ,np.zeros((NUM_METHODS_JOINT,NUM_ITERS))]
    
    for i in range(len(name_tuning)):
        norm_scores[i] = normalize_scores(scores[i], adtm)
        mean_norm_scores[i] = np.mean(norm_scores[i], axis=(2,3))
        std_norm_scores[i] = np.std(norm_scores[i], axis=(2,3))
        if confidence_interval:
            lower_lim[i] = np.percentile(norm_scores[i], 5, axis=(2,3))
            upper_lim[i] = np.percentile(norm_scores[i], 95, axis=(2,3))

    for i in range(len(name_tuning)):
        plt.plot(np.arange(NUM_ITERS), mean_norm_scores[i][method_index -helper[i], :], label=NAME_TUNING[i], color=palette[i], marker=MARKERS[i], markersize=14, linewidth=2.5, markevery=15)

        if randomness == 1:       # Randomness due to the seeds
            mean_tasks[i] = np.mean(norm_scores[i], axis=2)
            avg_var_across_tasks[i] = np.std(mean_tasks[i], axis=-1)

            plt.fill_between(np.arange(NUM_ITERS), mean_norm_scores[i][method_index -helper[i], :] - avg_var_across_tasks[i][method_index -helper[i], :], mean_norm_scores[i][method_index -helper[i], :] + avg_var_across_tasks[i][method_index -helper[i], :], alpha=0.2, color=palette[i])
            plt.plot(np.arange(NUM_ITERS), mean_norm_scores[i][method_index -helper[i], :] - avg_var_across_tasks[i][method_index -helper[i], :], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
            plt.plot(np.arange(NUM_ITERS), mean_norm_scores[i][method_index -helper[i], :] + avg_var_across_tasks[i][method_index -helper[i], :], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
        
        # Randomness due to the tasks
        elif randomness == 2:
            mean_seeds[i] = np.mean(norm_scores[i], axis=-1)
            avg_var_across_seeds[i] = np.std(mean_seeds[i], axis=-1)
            plt.fill_between(np.arange(NUM_ITERS), mean_norm_scores[i][method_index -helper[i], :] - avg_var_across_seeds[i][method_index -helper[i], :], mean_norm_scores[i][method_index -helper[i], :] + avg_var_across_seeds[i][method_index -helper[i], :], alpha=0.2, color=palette[i])
            plt.plot(np.arange(NUM_ITERS), mean_norm_scores[i][method_index -helper[i], :] - avg_var_across_seeds[i][method_index -helper[i], :], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
            plt.plot(np.arange(NUM_ITERS), mean_norm_scores[i][method_index -helper[i], :] + avg_var_across_seeds[i][method_index -helper[i], :], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)

        else:
            if confidence_interval:
                plt.fill_between(np.arange(NUM_ITERS), lower_lim[i][method_index -helper[i], :], upper_lim[i][method_index -helper[i], :], hatch='/', alpha=0.2, color=palette[i])
                plt.plot(np.arange(NUM_ITERS), lower_lim[i][method_index -helper[i], :], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
                plt.plot(np.arange(NUM_ITERS), upper_lim[i][method_index -helper[i], :], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
            
            else:
                plt.fill_between(np.arange(NUM_ITERS), mean_norm_scores[i] - std_norm_scores[i], mean_norm_scores[i] + std_norm_scores[i], alpha=0.2, color=palette[i])
                plt.plot(np.arange(NUM_ITERS), mean_norm_scores[i] - std_norm_scores[i], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
                plt.plot(np.arange(NUM_ITERS), mean_norm_scores[i] + std_norm_scores[i], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)

    for spine in plt.gca().spines.values():
        spine.set_edgecolor('dimgray')  # Set the desired color here
        spine.set_linewidth(1)      # Optionally, adjust the thickness
    
    # Customize grid
    plt.gca().grid(True, color='lightgray', linewidth=0.5)

    # Set the y-axis to have at most 5 ticks
    plt.gca().tick_params(axis='both', which='major', labelsize=28)
    plt.gca().xaxis.set_major_locator(MaxNLocator(6))
    plt.gca().yaxis.set_major_locator(MaxNLocator(5))

    plt.suptitle(title, fontsize=32)
    plt.xlabel('Iteration', fontsize=30)
    plt.ylabel('Average aggregated test score', fontsize=30)
    if rmse:
        plt.legend(loc="center left", ncol=1, bbox_to_anchor=(0.413, 0.8), fontsize=30)

    else:
        plt.legend(loc="center left", ncol=1, bbox_to_anchor=(0.413, 0.2), fontsize=30)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(METHODS), fontsize=30)

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

    plt.savefig(f"plots/{file}.png")
    plt.show()
