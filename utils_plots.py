import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns



# random.seed(42)
# seeds = random.sample(range(1000, 1000000), 20)
# seeds.sort()
seeds = [27225, 32244, 34326, 92161, 99246, 108473, 117739, 147316, 235053, 257787, 
        89389, 443417, 572858, 620176, 671487, 710570, 773246, 777646, 778572, 936518]

NUM_SEEDS = len(seeds)
NUM_TASKS = 59
NUM_REGR_TASKS = 36
NUM_CLASS_TASKS = 23
NUM_ITERS = 135
METHODS = ['grid_search', 'random_search', 'tpe', 'gp_bo']
NAME_METHODS = ['Fixed grid search', 'Random grid search', 'TPE', 'GP-BO']
NUM_METHODS = len(METHODS)
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
        num_tasks = NUM_CLASS_TASKS
        suites = [334, 337]
    
    else:
        num_tasks = NUM_REGR_TASKS
        suites = [335, 336]
    
    scores = np.zeros((NUM_METHODS, NUM_ITERS, num_tasks, NUM_SEEDS, NUM_FOLDS))

    k = 0
    names = []

    for suite_id in suites:
        with open(f"task_indices/{suite_id}_task_names.json", 'r') as f:
            _names = json.load(f)
        names.extend(_names)

        tasks = np.load(f"task_indices/{suite_id}_task_indices.npy")

        for task_id in tasks:
            for l, seed in enumerate(seeds):
                data = pd.read_csv(f"result_folder/seed_{seed}/{suite_id}_{task_id}.csv")
                data['current_best_test_score'] = data['current_best_test_score'].clip(0, 1)

                # Pick the best score between the scores obtained when choosing 'max_depth', respectively 'num_leaves'
                if rmse:
                    try_max_depth = data.loc[data['max_depth'] != -1, 'current_best_test_rmse'].reset_index(drop=True)
                    try_num_leaves = data.loc[data['max_depth'] == -1, 'current_best_test_rmse'].reset_index(drop=True)
                    df = pd.DataFrame({'try_max_depth_score': try_max_depth, 'try_num_leaves_rmse': try_num_leaves})
                    df['current_best_test_rmse'] = df.min(axis=1)
                    df['method'] = data.loc[0:2699, 'method']
                    df['fold'] = data.loc[0:2699, 'fold']

                    for i, method in enumerate(METHODS):
                        for m in FOLDS:
                            scores[i, :, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'current_best_test_rmse'].values

                else:
                    try_max_depth = data.loc[data['max_depth'] != -1, 'current_best_test_score'].reset_index(drop=True)
                    try_num_leaves = data.loc[data['max_depth'] == -1, 'current_best_test_score'].reset_index(drop=True)
                    df = pd.DataFrame({'try_max_depth_score': try_max_depth, 'try_num_leaves_score': try_num_leaves})
                    df['current_best_test_score'] = df.max(axis=1)
                    df['method'] = data.loc[0:2699, 'method']
                    df['fold'] = data.loc[0:2699, 'fold']

                    for i, method in enumerate(METHODS):
                        for m in FOLDS:
                            scores[i, :, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'current_best_test_score'].values

            k += 1

    aggregated_scores = np.mean(scores, axis=-1)

    return aggregated_scores, names


def normalize_scores(scores, adtm=False):
    if adtm: 
        max = np.max(scores, axis=(0, 1, 3))
        min = np.percentile(scores, 10, axis=(0, 1, 3))

        norm_scores = (scores - min[np.newaxis, np.newaxis, :, np.newaxis]) / (max[np.newaxis, np.newaxis, :, np.newaxis] - min[np.newaxis, np.newaxis, :, np.newaxis])
        norm_scores = np.clip(norm_scores, 0, 1)

    else:
        mean_for_norm = np.mean(scores, axis=(0, 1, 3))
        std_for_norm = np.std(scores, axis=(0, 1, 3))

        norm_scores = (scores - mean_for_norm[np.newaxis, np.newaxis, :, np.newaxis]) / std_for_norm[np.newaxis, np.newaxis, :, np.newaxis]

    return norm_scores


def plot_scores_per_task(scores, names, classification=False, adtm=False, confidence_interval=False, rmse=False):
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

    num_cols = 4
    num_rows = (num_tasks + num_cols - 1) // num_cols

    palette = set_plot_theme()
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4))
    axes = axes.flatten()

    norm_scores = normalize_scores(scores, adtm)

    mean_norm_scores = np.mean(norm_scores, axis=-1)
    std_norm_scores = np.std(norm_scores, axis=-1)

    if confidence_interval:
        lower_lim = np.percentile(norm_scores, 5, axis=-1)
        upper_lim = np.percentile(norm_scores, 95, axis=-1)

    for i in range(len(METHODS)):
        for k in range(num_tasks):
            ax = axes[k]
            ax.plot(np.arange(NUM_ITERS), mean_norm_scores[i, :, k], label=NAME_METHODS[i], color=palette[i], marker=MARKERS[i], markersize=14, linewidth=2.5, markevery=20)

            if confidence_interval:
                ax.fill_between(np.arange(NUM_ITERS), lower_lim[i, :, k], upper_lim[i, :, k], hatch='/', alpha=0.2, color=palette[i])
                ax.plot(np.arange(NUM_ITERS), lower_lim[i, :, k], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
                ax.plot(np.arange(NUM_ITERS), upper_lim[i, :, k], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)

            else:
                ax.fill_between(np.arange(NUM_ITERS), mean_norm_scores[i, :, k] - std_norm_scores[i, :, k], mean_norm_scores[i, :, k] + std_norm_scores[i, :, k], alpha=0.2, color=palette[i])
                ax.plot(np.arange(NUM_ITERS), mean_norm_scores[i, :, k] - std_norm_scores[i, :, k], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
                ax.plot(np.arange(NUM_ITERS), mean_norm_scores[i, :, k] + std_norm_scores[i, :, k], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)

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
    fig.legend(lines, labels, loc='upper center', ncol=len(METHODS), fontsize=30, bbox_to_anchor=(0.5, 0.959)) if classification else fig.legend(lines, labels, loc='upper center', ncol=len(METHODS), fontsize=30, bbox_to_anchor=(0.5, 0.968))#0.953
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

    plt.savefig(f"plots/{file}.png")
    plt.show()


def plot_scores_aggregated_tasks(scores, classification=False, adtm=False, confidence_interval=False, randomness=0, rmse=False):
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

    palette = set_plot_theme()
    plt.figure(figsize=(12, 8))

    norm_scores = normalize_scores(scores, adtm)

    mean_norm_scores = np.mean(norm_scores, axis=(2, 3))
    std_norm_scores = np.std(norm_scores, axis=(2, 3))

    if confidence_interval:
        lower_lim = np.percentile(norm_scores, 5, axis=(2, 3))
        upper_lim = np.percentile(norm_scores, 95, axis=(2, 3))

    for i in range(len(METHODS)):
        plt.plot(np.arange(NUM_ITERS), mean_norm_scores[i], label=NAME_METHODS[i], color=palette[i], marker=MARKERS[i], markersize=14, linewidth=2.5, markevery=15)

        # Randomness due to the seeds
        if randomness == 1:
            mean_tasks = np.mean(norm_scores[i], axis=1)
            avg_var_across_tasks = np.std(mean_tasks, axis=-1)

            plt.fill_between(np.arange(NUM_ITERS), mean_norm_scores[i] - avg_var_across_tasks, mean_norm_scores[i] + avg_var_across_tasks, alpha=0.2, color=palette[i])
            plt.plot(np.arange(NUM_ITERS), mean_norm_scores[i] - avg_var_across_tasks, linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
            plt.plot(np.arange(NUM_ITERS), mean_norm_scores[i] + avg_var_across_tasks, linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
        
        # Randomness due to the tasks
        elif randomness == 2:
            mean_seeds = np.mean(norm_scores[i], axis=-1)
            avg_var_across_seeds = np.std(mean_seeds, axis=-1)
            plt.fill_between(np.arange(NUM_ITERS), mean_norm_scores[i] - avg_var_across_seeds, mean_norm_scores[i] + avg_var_across_seeds, alpha=0.2, color=palette[i])
            plt.plot(np.arange(NUM_ITERS), mean_norm_scores[i] - avg_var_across_seeds, linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
            plt.plot(np.arange(NUM_ITERS), mean_norm_scores[i] + avg_var_across_seeds, linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)

        else:
            if confidence_interval:
                plt.fill_between(np.arange(NUM_ITERS), lower_lim[i], upper_lim[i], hatch='/', alpha=0.2, color=palette[i])
                plt.plot(np.arange(NUM_ITERS), lower_lim[i], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
                plt.plot(np.arange(NUM_ITERS), upper_lim[i], linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
            
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

    plt.savefig(f"plots/{file}.png")
    plt.show()
