"""Author: Ioana Iacobici https://github.com/iiacobici modified by Floris Koster https://github.com/fl0risk"""

import random
import re

import gpboost as gpb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from utils import modified_grid_search_tune_parameters



class ParameterOptimization:
    """This class is used to perform hyperparameter tuning using the proposed methods and to evaluate the model obtained using the best hyperparameters."""

    def __init__(self, X, y, categorical_indicator, suite_id, test_size=0.2, val_size=0.2, try_num_leaves=False, seed=42,joint_tuning_depth_leaves = False):

        self.seed = seed
        self.fixed_seeds = self._generate_local_seeds()

        X, y = self._subsample_data(X, y)

        # Check if the target variable is a pandas Series or a numpy array
        if isinstance(y, np.ndarray):    #np.ndarray is the type of an object created with np.array()
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.ravel()  # Convert (n_samples, 1) to (n_samples,)

            y = pd.Series(y)

        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.squeeze()

        self.X = X
        self.y = y
        self.categorical_indicator = categorical_indicator
        self.test_size = test_size
        self.val_size = val_size
        self.suite_id = suite_id
        self.try_num_leaves = try_num_leaves
        self.joint_tuning_depth_leaves = joint_tuning_depth_leaves

        self.max_bin_val = np.min([self.X.shape[0], 10000])
        self.X = self._clean_column_names(self.X)

        self._create_other_params()
        self._preprocess_features()

        self.splits = self._split_data()


    def run_methods(self, ):
        """This function runs all hyperparameter tuning methods on the 5 folds and returns the results in a DataFrame."""
        # Iterate through the 5 folds
        for fold, (full_train_index, test_index) in enumerate(self.splits):
            X_train_full, X_test = self.X.iloc[full_train_index], self.X.iloc[test_index]
            y_train_full, y_test = self.y.iloc[full_train_index], self.y.iloc[test_index]

            # Perform hyperparameter tuning using the different methods
            trials_grid_search = self.grid_search_method(
                X_train_full=X_train_full, y_train_full=y_train_full, 
                X_test=X_test, y_test=y_test
            )
            trials_random_search = self.grid_search_method(
                X_train_full=X_train_full, y_train_full=y_train_full, 
                X_test=X_test, y_test=y_test, 
                num_try_random=135
            )
            trials_tpe = self.tpe_method(
                X_train_full=X_train_full, y_train_full=y_train_full,
                X_test=X_test, y_test=y_test
            )
            trials_gp_bo = self.gp_bo_method(
                X_train_full=X_train_full, y_train_full=y_train_full,
                X_test=X_test, y_test=y_test
            )

            # Concatenate the results from the different methods

            for method in ['grid_search','random_search', 'tpe', 'gp_bo']:#'grid_search','random_search', 'tpe', 'gp_bo']:
                trials = eval(f'trials_{method}')
                trials['fold'] = fold
                trials['method'] = method

                if fold == 0 and method == 'grid_search':
                     final_results = trials
                else:
                     final_results = pd.concat([final_results, trials])
        final_results.reset_index(inplace=True)
        final_results.rename(columns={"index": "iter"}, inplace=True)
        return final_results


    def grid_search_method(self, X_train_full, y_train_full, X_test, y_test, num_try_random=None):
        """This function performs fixed/random grid search on the model."""
        # Define the hyperparameter grid
        param_grid = {
            'learning_rate': [0.01, 0.1, 1],
            'min_data_in_leaf': [10, 100, 1000],
            'max_depth': [1, 2, 3, 5, 10],
            'lambda_l2': [0, 1, 10]
        }

        # Change the grid for random search
        if num_try_random is not None:
            # Complete the current grid with more options for the parameters
            param_grid['learning_rate'].append(0.001)
            param_grid['min_data_in_leaf'].append(1)
            param_grid['lambda_l2'].append(100)

            # Add new parameters to the grid
            param_grid['max_bin'] = [255, 500, 1000, self.max_bin_val]
            param_grid['bagging_fraction'] = [0.5, 0.75, 1]
            param_grid['feature_fraction'] = [0.5, 0.75, 1]

        # Adjust the grid for the case where we tune the 'num_leaves' parameter
        if self.try_num_leaves:
            param_grid.pop('max_depth')
            param_grid['num_leaves'] = [2**1, 2**2, 2**3, 2**5, 2**10]
        if self.joint_tuning_depth_leaves:
            param_grid['num_leaves'] = [2**1, 2**2, 2**3, 2**5, 2**10]    
        # Perform hyperparameter tuning
        train_set = gpb.Dataset(X_train_full, label=y_train_full)

        X_train_full = X_train_full.reset_index(drop=True)
        y_train_full = y_train_full.reset_index(drop=True)

        # Split the full training set into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=self.val_size, random_state=self.fixed_seeds[2]
        )
        
        # Define the folds for the grid search
        train_set_idx = X_train.index.values
        val_set_idx = X_val.index.values
        folds = [(train_set_idx, val_set_idx)]

        opt_params = modified_grid_search_tune_parameters(
            param_grid=param_grid, params=self.other_params, num_try_random=num_try_random, folds=folds, seed=self.seed, 
            train_set=train_set, use_gp_model_for_validation=False, verbose_eval=True,
            num_boost_round=1000, early_stopping_rounds=20
        )

        # Uncomment to evaluate the model on the test set using the best hyperparameters chosen by the algorithm:

        # train_set_full = gpb.Dataset(self.X_train_full, label=self.y_train_full)
        # best_params = opt_params['best_params'].copy()

        # # Get the best number of iterations
        # best_iter = opt_params['best_iter']

        # best_params.update(self.other_params)

        # # Evaluate the model on the unseen test set
        # bst = gpb.train(
        #     params=best_params, train_set=train_set_full, num_boost_round=best_iter,
        #     verbose_eval=False
        # )
        # y_pred = bst.predict(data=self.X_test, pred_latent=False)

        # # Adjust the evaluation metric based on the task
        # if self.suite_id in [334, 337]:
        #     y_pred = (y_pred > 0.5).astype(int)
        #     score = accuracy_score(self.y_test, y_pred)

        # else:
        #     score = root_mean_squared_error(self.y_test, y_pred)

        # Format the DataFrame
        df_trials = self._convert_dict_to_df(opt_params['all_combinations'])
        if num_try_random is None:
            df_trials['max_bin'] = 255
            df_trials['bagging_fraction'] = 1
            df_trials['feature_fraction'] = 1
            df_trials['val_score'] = df_trials.pop('val_score')

        # Compute the test scores
        df_trials = self._compute_test_scores(
            X_train_full=X_train_full, y_train_full=y_train_full, X_test=X_test, y_test=y_test,
            df=df_trials, best_iter=opt_params['best_iter']
        )

        # Add the 'try_num_leaves' and 'joint_tuning_depth_leaves columns to the DataFrame
        df_trials['try_num_leaves'] = self.try_num_leaves
        df_trials['joint_tuning_depth_leaves'] = self.joint_tuning_depth_leaves
        return df_trials # score, opt_params['best_params'], opt_params['best_iter']


    def tpe_method(self, X_train_full, y_train_full, X_test, y_test):
        """This function performs hyperparameter tuning using the TPE method."""

        self.min_score = float('inf')
        
        # Split the full training set into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=self.val_size, random_state=self.fixed_seeds[2]
        )

        # Define the objective function
        def objective_opt(trial):
            """Objective function for the Optuna optimization."""
            # Adjust the grid for the case where we tune the 'num_leaves' parameter
            if self.try_num_leaves:
                param_grid = {
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 1),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 1000),
                    'lambda_l2': trial.suggest_float('lambda_l2', 0, 100),
                    'max_bin': trial.suggest_int('max_bin', 255, self.max_bin_val),
                    'num_leaves': trial.suggest_int('num_leaves', 2, 1024),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1)
                }

            elif self.joint_tuning_depth_leaves: #add option for jointly tuning depth and num_leaves
                param_grid = {
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 1),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 1000),
                    'max_depth': trial.suggest_int('max_depth', 1, 10),
                    'lambda_l2': trial.suggest_float('lambda_l2', 0, 100),
                    'max_bin': trial.suggest_int('max_bin', 255, self.max_bin_val),
                    'num_leaves': trial.suggest_int('num_leaves', 2, 1024),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1)
                }
            else:
                param_grid = {
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 1),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 1000),
                    'max_depth': trial.suggest_int('max_depth', 1, 10),
                    'lambda_l2': trial.suggest_float('lambda_l2', 0, 100),
                    'max_bin': trial.suggest_int('max_bin', 255, self.max_bin_val),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1)
                }
            # Train the model
            score, best_iter = self._train_model_for_validation(
                X_train, y_train, X_val, y_val, 
                param_grid
            )

            # Get the best number of iterations
            if score < self.min_score:
                self.min_score = score
                self.best_iter = best_iter

            return score

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective_opt, n_trials=135)
        df = study.trials_dataframe()

        # Uncomment to evaluate the model on the test set using the best hyperparameters chosen by the algorithm:

        # train_set_full = gpb.Dataset(self.X_train_full, label=self.y_train_full)
        # best_params = study.best_params.copy()
        # best_params.update(self.other_params)

        # # Evaluate the model on the unseen test set
        # bst = gpb.train(
        #     params=best_params, train_set=train_set_full, num_boost_round=self.best_iter,
        #     verbose_eval=False
        # )
        # y_pred = bst.predict(data=self.X_test, pred_latent=False)

        # # Adjust the evaluation metric based on the task
        # if self.suite_id in [334, 337]:
        #     y_pred = (y_pred > 0.5).astype(int)
        #     score = accuracy_score(self.y_test, y_pred)

        # else:
        #     score = root_mean_squared_error(self.y_test, y_pred)

        df_trials = self._modify_df_for_tpe(df)

        # Compute the test scores
        df_trials = self._compute_test_scores(
            X_train_full=X_train_full, y_train_full=y_train_full, X_test=X_test, y_test=y_test,
            df=df_trials, best_iter=self.best_iter
        )

        # Add the 'try_num_leaves' and 'joint_tuning_depth_leaves columns to the DataFrame
        df_trials['try_num_leaves'] = self.try_num_leaves
        df_trials['joint_tuning_depth_leaves'] = self.joint_tuning_depth_leaves
        return df_trials # score, study.best_params, self.best_iter


    def gp_bo_method(self, X_train_full, y_train_full, X_test, y_test):
        """This function performs hyperparameter tuning using the GP-BOOST method."""
        # Define the hyperparameter space
        space = [
            Real(0.001, 1, name='learning_rate'),
            Integer(1, 1000, name='min_data_in_leaf'),
            Integer(1, 10, name='max_depth'),
            Real(0, 100, name='lambda_l2'),
            Integer(255, self.max_bin_val, name='max_bin'),
            Real(0.5, 1, name='bagging_fraction'),
            Real(0.5, 1, name='feature_fraction')
        ]
        self.min_score = float('inf')

        # Adjust the space for the case where we tune the 'num_leaves' parameter
        if self.try_num_leaves:
            space.remove(Integer(1, 10, name='max_depth'))
            space.append(Integer(2, 1024, name='num_leaves'))
        if self.joint_tuning_depth_leaves:
            space.append(Integer(2, 1024, name='num_leaves'))
        # Split the full training set into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=self.val_size, random_state=self.fixed_seeds[2]
        )

        # Temporarily modify the 'numpy' module to avoid deprecation error ('np.int' instead of 'int')
        np.int = int

        # Define the objective function
        @use_named_args(space)
        def objective_gp_bo(**params):
            """Objective function for the GP-BO optimization."""

            score, best_iter = self._train_model_for_validation(
                X_train, y_train, X_val, y_val, 
                params
            )

            # Get the best number of iterations
            if score < self.min_score:
                self.min_score = score
                self.best_iter = best_iter

            return score

        result = gp_minimize(objective_gp_bo, space, n_calls=135, random_state=self.seed)

        # Restore the 'numpy' module to its original state
        delattr(np, 'int')

        # Uncomment to evaluate the model on the test set using the best hyperparameters chosen by the algorithm:

        # best_parameters = dict(zip([p.name for p in space], result.x))

        # train_set_full = gpb.Dataset(self.X_train_full, label=self.y_train_full)
        # best_params = best_parameters.copy()
        # best_params.update(self.other_params)

        # # Evaluate the model on the unseen test set
        # bst = gpb.train(
        #     params=best_params, train_set=train_set_full, num_boost_round=self.best_iter,
        #     verbose_eval=False
        # )
        # y_pred = bst.predict(data=self.X_test, pred_latent=False)

        # # Adjust the evaluation metric based on the task
        # if self.suite_id in [334, 337]:
        #     y_pred = (y_pred > 0.5).astype(int)
        #     score = accuracy_score(self.y_test, y_pred)

        # else:
        #     score = root_mean_squared_error(self.y_test, y_pred)

        df_trials = self._convert_array_to_df(result.x_iters, result.func_vals) #x_iters are the parameter values where the objective function was evaluated

        # Compute the test scores
        df_trials = self._compute_test_scores(
            X_train_full=X_train_full, y_train_full=y_train_full, X_test=X_test, y_test=y_test,
            df=df_trials, best_iter=self.best_iter
        )

       # Add the 'try_num_leaves' and 'joint_tuning_depth_leaves columns to the DataFrame
        df_trials['try_num_leaves'] = self.try_num_leaves
        df_trials['joint_tuning_depth_leaves'] = self.joint_tuning_depth_leaves
        return df_trials # score, best_parameters, self.best_iter
    

    def _generate_local_seeds(self):
        """This function generates the local seeds for the current task."""
        random.seed(42)
        seeds = random.sample(range(1000, 1000000), 3)

        return seeds
    

    def _subsample_data(self, X, y):
        """This function subsamples the data if the dataset is too large."""
        if X.shape[0] > 100000:
            X = X.sample(n=100000, random_state=self.fixed_seeds[0])
            y = y.loc[X.index]

        return X, y
    

    def _clean_column_names(self, df):
        """This function cleans the column names of a DataFrame by replacing special JSON characters with underscores."""
        # Define a regular expression pattern to match special characters
        pattern = re.compile(r'[\{\}\[\]\"\:\\,]')
 
        df.columns = [pattern.sub('_', col) for col in df.columns]

        return df
    

    def _create_other_params(self):
        """This function creates the 'other_params' dictionary based on the task."""

        self.other_params = {'verbose': -1}

        # Adjust the 'other_params' dictionary based on the task
        if self.try_num_leaves:
            self.other_params['max_depth'] = -1     #{'verbose': -1, 'max_depth': -1}
        elif not self.joint_tuning_depth_leaves:
            self.other_params['num_leaves'] = 2**10     #{'verbose': -1, 'num_leaves': 2**10}
        # Set the objective and metric functions based on the given task
        if self.suite_id in [334, 337]:
            self.other_params['objective'] = 'binary_logit'
            self.other_params['metric'] = 'binary_error'

        else:
            self.other_params['objective'] = 'regression'
            self.other_params['metric'] = 'rmse'
    

    def _preprocess_features(self):
        """This function converts the categorical features of the data into numeric types."""
        # Extract the categorical columns in the DataFrame
        categorical_columns = self.X.columns[self.categorical_indicator]

        # Fit the OneHotEncoder
        enc = OneHotEncoder()
        encoded_columns = enc.fit_transform(self.X[categorical_columns])

        # Format the DataFrame with the encoded columns
        encoded_df = pd.DataFrame(encoded_columns.toarray(), columns=enc.get_feature_names_out(categorical_columns))
        self.X.reset_index(drop=True, inplace=True)
        self.X = pd.concat([self.X.drop(columns=categorical_columns), encoded_df], axis=1)

        # Convert the target variable into numeric type
        if self.suite_id in [334, 337]:
            le = LabelEncoder()
            self.y = le.fit_transform(self.y)

            # Ensure the target variable is a pandas Series
            self.y = pd.Series(self.y)


    def _split_data(self):
        """This function splits the data into 5 folds used for training and testing iteratively."""
        kf = KFold(n_splits=5, shuffle=True, random_state=self.fixed_seeds[1])

        return kf.split(self.X)


    def _train_model_for_validation(self, X_train, y_train, X_val, y_val, params, num_boost_round: int = 1000) -> float:
        """This function performs the model training and evaluation and returns the prediction accuracy based on the validation set."""
        params_copy = params.copy()
        params_copy.update(self.other_params)
        train_set = gpb.Dataset(X_train, label=y_train) 
        valid_set = gpb.Dataset(X_val, label=y_val)
        # Train the model
        bst = gpb.train(
            params=params_copy, train_set=train_set, num_boost_round=num_boost_round,
            valid_sets=[valid_set], early_stopping_rounds=20,
            verbose_eval=False
        )
        y_pred = bst.predict(data=X_val, pred_latent=False) #pred_latent = FALSE => response variable is predicted

        # Get the best number of iterations
        best_iter = bst.best_iteration

        # Evaluate the model based on the respective task
        if self.suite_id in [334, 337]:
            y_pred = (y_pred > 0.5).astype(int)
            score = 1 - accuracy_score(y_val, y_pred)

        else:
            score = root_mean_squared_error(y_val, y_pred)

        return score, best_iter
    

    def _compute_test_scores(self, X_train_full, y_train_full, X_test, y_test, df, best_iter: int) -> float:
        """This function trains the model on the full training set and evaluates it on the test set, adding the 'test_score' column to the corresponding DataFrame."""
        test_scores = []
        test_log_loss = []
        test_f1_scores = []
        test_rmse = []
        for _, row in df.iterrows():
            params_copy = row.drop(['val_score']).to_dict()

            # Ensure the correct types for specific parameters
            params_copy = {key: int(value) if key in ['min_data_in_leaf', 'max_depth', 'num_leaves', 'max_bin'] else value for key, value in params_copy.items()}
            train_set_full = gpb.Dataset(X_train_full, label=y_train_full)
    
            # Train the model
            bst = gpb.train(
                params=params_copy, train_set=train_set_full, num_boost_round=best_iter,
                verbose_eval=False
            )
            y_pred = bst.predict(data=X_test, pred_latent=False)

            # Evaluate the model based on the respective task
            if self.suite_id in [334, 337]:
                log_loss_score = log_loss(y_test, y_pred)
                y_pred = (y_pred > 0.5).astype(int)
                score = accuracy_score(y_test, y_pred)
                f_1_score = f1_score(y_test, y_pred)
                rmse = np.nan

            else:
                log_loss_score = np.nan
                f_1_score = np.nan
                score = r2_score(y_test, y_pred, force_finite=True)
                rmse = root_mean_squared_error(y_test, y_pred)


            test_scores.append(score)
            test_log_loss.append(log_loss_score)
            test_f1_scores.append(f_1_score)
            test_rmse.append(rmse)

        # Add the 'test_score' and the 'current_best_test_score' columns to the DataFrame
        df['test_score'] = test_scores
        df['test_log_loss'] = test_log_loss
        df['test_f1_score'] = test_f1_scores
        df['test_rmse'] = test_rmse
        df['current_best_test_score'] = df['test_score'].cummax()
        df['current_best_test_log_loss'] = df['test_log_loss'].cummin()
        df['current_best_test_f1_score'] = df['test_f1_score'].cummax()
        df['current_best_test_rmse'] = df['test_rmse'].cummin()

        return df


    def _convert_dict_to_df(self, dict_params):
        """This function converts the dictionary outputted by the 'modified_grid_search_tune_parameters' function into a DataFrame."""
        # Normalize the dictionary to ensure all parameters are present in each element
        normalized_data = [{**v['params'], 'val_score': v['score']} for v in dict_params.values()]

        # Format the DataFrame
        df = pd.DataFrame(normalized_data)
        if self.try_num_leaves:
            df['max_depth'] = -1
        elif not self.joint_tuning_depth_leaves:
            df['num_leaves'] = 2**10
        return df
    

    def _modify_df_for_tpe(self, df):
        """This function modifies the DataFrame for the TPE method."""
        # Drop unnecessary columns from the DataFrame with the trials
        df.drop(columns=['number', 'datetime_start', 'datetime_complete', 'duration', 'state'], inplace=True)

        if self.try_num_leaves:
            df.columns = ['val_score', 'bagging_fraction', 'feature_fraction', 'lambda_l2', 'learning_rate', 'max_bin', 'min_data_in_leaf', 'num_leaves']
            df['max_depth'] = -1
        elif self.joint_tuning_depth_leaves:
            df.columns = ['val_score', 'bagging_fraction', 'feature_fraction', 'lambda_l2', 'learning_rate', 'max_bin','max_depth', 'min_data_in_leaf', 'num_leaves']
        else:
            df.columns = ['val_score','bagging_fraction', 'feature_fraction', 'lambda_l2', 'learning_rate', 'max_bin', 'max_depth', 'min_data_in_leaf']
            df['num_leaves'] = 2**10

        return df
    

    def _convert_array_to_df(self, x_iters, func_vals):
        """This function converts the arrays from the GP-BO trials into a DataFrame."""

        if self.try_num_leaves:
            df = pd.DataFrame(x_iters, columns=['learning_rate', 'min_data_in_leaf', 'lambda_l2', 'max_bin', 'bagging_fraction', 'feature_fraction', 'num_leaves'])
            df['max_depth'] = -1
        elif self.joint_tuning_depth_leaves:
            df = pd.DataFrame(x_iters, columns=['learning_rate', 'min_data_in_leaf', 'max_depth','lambda_l2', 'max_bin', 'bagging_fraction', 'feature_fraction', 'num_leaves'])
        else:
            df = pd.DataFrame(x_iters, columns=['learning_rate', 'min_data_in_leaf', 'max_depth', 'lambda_l2', 'max_bin', 'bagging_fraction', 'feature_fraction'])
            df['num_leaves'] = 2**10
        
        # Convert the 'score' array into a column for the DataFrame
        df['val_score'] = func_vals

        return df