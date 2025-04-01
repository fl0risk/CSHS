import os

import gpboost as gpb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error, r2_score

from utils_modelling import read_and_preprocess_dataset, split_dataset, save_gpboost_summary #read_and_preprocess_dataset



PATH = 'results_model/lin_model/'
os.makedirs(PATH, exist_ok=True)

def assess_linear_regression(classification: bool = False) -> None:
    #df = read_and_preprocess_dataset(classification)
    df = read_and_preprocess_dataset(classification)
    X, y, group_data = split_dataset(df)

    # Fit a grouped random effects model
    gp_model = gpb.GPModel(group_data=group_data, likelihood="gaussian")
    gp_model.fit(y=y, X=X, params={"std_dev": True})

    if classification:
        filename = 'classification'

    else:
        filename = 'regression'

    save_gpboost_summary(gp_model, os.path.join(PATH, f'model_summary_{filename}.txt'))

    # Make predictions
    y_pred = gp_model.predict(y=y, group_data_pred=group_data, X_pred=X)
    rmse = root_mean_squared_error(y, y_pred['mu'])
    r2 = r2_score(y, y_pred['mu'])

    # Plot predicted vs actual values
    sns.set_theme(style='dark')
    plt.figure(figsize=(10, 8))

    plt.scatter(y_pred['mu'], y, color='mediumslateblue', alpha=0.3, s=10)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle='--', color='firebrick', linewidth=2)
    plt.xlabel('Predicted values')
    plt.ylabel('Actual values')
    plt.suptitle('Predicted vs Actual values', fontsize=16)
    plt.title(f'$RMSE: {rmse:.4f}, R^2: {r2:.4f}$', fontsize=14)
    
    plt.savefig(os.path.join(PATH, f'predicted_vs_actual_{filename}.png'))


def main():
    for classification in [False, True]:
        assess_linear_regression(classification)


if __name__ == '__main__':
    main()