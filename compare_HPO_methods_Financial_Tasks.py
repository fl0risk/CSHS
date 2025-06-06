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
NUM_TASKS = 3
NUM_CLASS_TASKS = 3
NUM_ITERS = 135

HPO_METHODS = ['random_search', 'tpe','gp_bo', 'SMAC'] #TODO: add 'gp_bo'
HPO_METHODS_NAMES = ['Random Grid Search', 'TPE','GP-BO', 'SMAC'] #TODO: add 'GP-BO'
TUNING_STRAT = ['Num Leaves']
NUM_TUNING_STRAT = len(TUNING_STRAT)
NUM_METHODS = len(HPO_METHODS)
RANDOMNESS = ['both','seeds','tasks']
FOLDS = [0, 1, 2, 3, 4]
NUM_FOLDS = len(FOLDS)
MARKERS = ["o", "*", "^", "s","d"] 

def set_plot_theme():
    # Set Seaborn theme
    sns.set_theme(context="paper", style="white")
    palette = ['lime', 'darkorange', 'fuchsia', 'deepskyblue','crimson']

    return palette

def create_scores_dict():
    
    scores = [np.zeros((NUM_ITERS, NUM_TASKS, NUM_SEEDS, NUM_FOLDS)) for i in range(NUM_METHODS)]
    log_loss = [np.zeros((NUM_ITERS, NUM_TASKS, NUM_SEEDS, NUM_FOLDS)) for i in range(NUM_METHODS)]
    f1_score = [np.zeros((NUM_ITERS, NUM_TASKS, NUM_SEEDS, NUM_FOLDS)) for i in range(NUM_METHODS)]
    k=0
        
    tasks = [361055, 361066, 317599]

    for task_id in tasks:
        for l, seed in enumerate(seeds):
            data = pd.read_csv(f"/Users/floris/Desktop/ETH/ETH_FS25/Semesterarbeit/Results_Financial_Data/seed_{seed}/{task_id}.csv")
            test_score = data.loc[data['try_num_leaves'] == True, 'test_score'].reset_index(drop=True) #TODO: add current_best
            test_log_loss = data.loc[data['try_num_leaves'] == True, 'test_log_loss'].reset_index(drop=True)
            test_f1_score = data.loc[data['try_num_leaves'] == True, 'test_f1_score'].reset_index(drop=True)
            df = pd.DataFrame({'test_score': test_score, 'test_log_loss': test_log_loss,'test_f1_score': test_f1_score})
            
            df['method'] = data['method']
            df['fold'] = data['fold']
            print(df)
            for i, method in enumerate(HPO_METHODS):
                for m in FOLDS:
                    print(i,k,l,m)
                    scores[i][:, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'test_score'].values
                    log_loss[i][:, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'test_log_loss'].values
                    f1_score[i][:, k, l, m] = df.loc[(df['method'] == method) & (df['fold'] == m), 'test_f1_score'].values
        k += 1
    aggregated_scores = []
    aggregated_loss = []
    aggregated_f1_score  = []
    for i in range(NUM_METHODS):
        aggregated_scores.append(np.mean(scores[i], axis=-1)) #take mean over folds
        aggregated_loss.append(np.mean(log_loss[i], axis=-1)) #take mean over folds
        aggregated_f1_score.append(np.mean(f1_score[i], axis=-1)) #take mean over folds
    return aggregated_scores, aggregated_loss, aggregated_f1_score

def normalize_scores(scores,adtm):
    if adtm:
        scores_max = [np.max(s, axis=(0, 2)) for s in scores] #get maximum across task for each method, 0-axis iterations and 2-axis seeds 
        max = np.maximum.reduce(scores_max)
        for j in range(NUM_METHODS):
            if j!=0:
                all_scores = np.concatenate((all_scores, scores[j]),axis = 0)
            else:
                all_scores = scores[j]
        min = np.percentile(all_scores, q=10, axis=(0,2))
        print(min/(max-min))
        norm_scores = [(s - min[np.newaxis, :, np.newaxis]) / (max[np.newaxis, :, np.newaxis] - min[np.newaxis, :, np.newaxis]) for s in scores]
    else:
        for j in range(NUM_METHODS):
            if j!=0:
                all_scores = np.concatenate((all_scores, scores[j]),axis = 0)
            else:
                all_scores = scores[j]
        mean_for_norm = np.mean(all_scores, axis = (0,2))
        std_for_norm = np.std(all_scores, axis = (0,2))
        norm_scores = [(s - mean_for_norm[np.newaxis, :, np.newaxis]) / std_for_norm[np.newaxis, :, np.newaxis] for s in scores]

    return norm_scores
def plot_scores_per_task(norm_score, names, type='score',best = True,confidence_interval=False):
    if best:
        title = f'Average best {type}'
    else:
        title = f'Average {type}'
    title += ' per iteration for each task'

    is_loss = (type == 'log loss')

    num_tasks = NUM_CLASS_TASKS
    num_cols = 2
    num_rows = 2

    palette = set_plot_theme()
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 10))
    axes = axes.flatten()

    
    mean_norm_scores = np.mean(norm_score, axis=-1)
    std_norm_scores = np.std(norm_score, axis=-1)

    for i in range(NUM_METHODS):
        for k in range(num_tasks):
            ax = axes[k]
            ax.plot(np.arange(NUM_ITERS), np.clip(mean_norm_scores[i, :, k],-1 if is_loss else 0,1), label=HPO_METHODS_NAMES[i], color=palette[i], marker=MARKERS[i], markersize=14, linewidth=2.5, markevery=20)
            
            if confidence_interval:
                ax.fill_between(np.arange(NUM_ITERS), np.clip(mean_norm_scores[i, :, k] - std_norm_scores[i, :, k],-1 if is_loss else 0 ,1), np.clip(mean_norm_scores[i, :, k] + std_norm_scores[i, :, k],-1 if is_loss else 0,1), alpha=0.2, color=palette[i])
                ax.plot(np.arange(NUM_ITERS), np.clip(mean_norm_scores[i, :, k] - std_norm_scores[i, :, k],-1 if is_loss else 0,1), linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
                ax.plot(np.arange(NUM_ITERS), np.clip(mean_norm_scores[i, :, k] + std_norm_scores[i, :, k],-1 if is_loss else 0,1), linestyle='--', color=palette[i], alpha=0.6, linewidth=2.5)
                

            if len(names[k]) > 24:
                names[k] = names[k][:24]

            ax.set_title(names[k], fontsize=28)

            # Customize grid
            ax.grid(True, color='lightgray', linewidth=0.5)


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
            if type == 'log loss':
                ax.set_ylim(-0.8, 1)
            else:
                ax.set_ylim(0, 1)

    # Remove any empty subplots
    for a in range(num_tasks, num_rows * num_cols):
        fig.delaxes(axes[a])

    fig.suptitle(title, fontsize=32)
    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', ncol=NUM_METHODS, fontsize=30, bbox_to_anchor=(0.5, 0.959))
    fig.text(0.5, 0.045, 'Iteration', ha='center', va='center', fontsize=30)
    fig.text(0.04, 0.5, 'Average test score', ha='center', va='center', rotation='vertical', fontsize=30)

    plt.tight_layout(rect=[0, 0, 1, 0.85]) 
    plt.subplots_adjust(top=0.894, bottom=0.08, left=0.1, hspace=0.2, wspace=0.2)
    file = f"{type}_per_financial_task"
    file += "_classification"
    if best:
        file+= "_current_best"
    if confidence_interval:
        file+= "_confidence_interval"
    

    plt.savefig(f"plots/{file}.png")
    plt.show()
