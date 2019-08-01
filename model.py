# %%
import time
from functools import partial

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, NuSVC

from load_data import load_data
from logger import get_logger
from figure import plot_confusion_matrix

# %%
def euclidean_distance(y_test, y_pred):
    return np.sqrt((y_test[:, 0] - y_pred[:, 0])**2 + (y_test[:, 1] - y_pred[:, 1])**2)

# %%
def train_classifier_model(X, X_test, y, params=None, folds=None, model_type='lgb', model=None, plot_feature_importance=False):
    if folds == None:
        n_fold = 3
        folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=11)
    else:
        n_fold = folds.n_splits
    oof = np.zeros(len(X))
    y_pred = pd.DataFrame(np.zeros((len(X_test), n_fold)))
    scores = []
    feature_importance = pd.DataFrame()
    logger.info('train start')
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        logger.info(f'    Fold {fold_n}, started at {time.ctime()}')
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        
        if model_type == 'lgb':
            if model == None:
                model = lgb.LGBMRegressor(**params, n_estimators=50000, n_jobs=-1)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                    verbose=10000, early_stopping_rounds=200)
            
            y_pred_valid = model.predict(X_valid)
            y_pred.loc[:, fold_n] = model.predict(X_test, num_iteration=model.best_iteration_)

        if model_type == 'sklearn':
            if model == None:
                model = RandomForestClassifier(n_estimators=1000, random_state=1)
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid)
            accuracy = accuracy_score(y_valid, y_pred_valid)
            f1 = f1_score(y_valid, y_pred_valid, average='macro')
            logger.info(f'    Fold {fold_n}. Accuracy: {accuracy:.4f}.')
            logger.info(f'    Fold {fold_n}. Macro-F1: {f1:.4f}.')
            y_pred.loc[:, fold_n] = model.predict(X_test)
            
        if model_type == 'nn':
            model.fit(x=X_train, y=y_train, validation_data=[X_valid, y_valid], **params)
    
            y_pred_valid = model.predict(X_valid)[:, 0]
            y_pred.loc[:, fold_n] = model.predict(X_test)[:, 0]
            
            history = model.history.history
            train_loss = history["loss"]
            valid_loss = history["val_loss"]
            logger.info(f"loss: {train_loss[-1]:.3f} | val_loss: {valid_loss[-1]:.3f} | diff: {train_loss[-1]-valid_loss[-1]:.3f}")
            
            if fold_n == 0:
                plt.figure(figsize=(16, 12))
            p = plt.plot(train_loss, label=f'train_{fold_n}')
            plt.plot(valid_loss, '--', label=f'valid_{fold_n}', color=p[0].get_color())
            plt.grid()
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')

        oof[valid_index] = y_pred_valid.reshape(-1,)
        scores.append(mean_absolute_error(y_valid, y_pred_valid))

        prediction = y_pred.mode(axis=1)[0]
        
        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    logger.info('train end')
    logger.info('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
        
    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
            plt.title('LGB Features (avg over folds)')
        
            return oof, prediction, feature_importance
        return oof, prediction
    
    else:
        return oof, prediction

# %%
def train_regressor_model(X, X_test, y, params=None, folds=None, model_type='lgb', model=None, plot_feature_importance=False):
    if folds == None:
        n_fold = 3
        folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
    else:
        n_fold = folds.n_splits
    oof = np.zeros((len(X), 2))
    prediction = np.zeros((len(X_test), 2))
    scores = []
    feature_importance = pd.DataFrame()
    logger.info('train start')
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        logger.info(f'    Fold {fold_n}, started at {time.ctime()}')
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        
        if model_type == 'lgb':
            if model == None:
                model = lgb.LGBMRegressor(**params, n_estimators=50000, n_jobs=-1)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                    verbose=10000, early_stopping_rounds=200)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        if model_type == 'sklearn':
            if model == None:
                model = RandomForestRegressor(n_estimators=200, random_state=1)
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid)
            euclidean_dist = euclidean_distance(y_valid, y_pred_valid)
            mse = mean_squared_error(y_valid, y_pred_valid)
            logger.info(f'    Fold {fold_n}. Mean of euclidean distance: {np.mean(euclidean_dist):.4f}.')
            logger.info(f'    Fold {fold_n}. MSE: {mse:.4f}.')
            y_pred = model.predict(X_test)
            
        if model_type == 'nn':
            model.fit(x=X_train, y=y_train, validation_data=[X_valid, y_valid], **params)
    
            y_pred_valid = model.predict(X_valid)[:, 0]
            y_pred = model.predict(X_test)[:, 0]
            
            history = model.history.history
            train_loss = history["loss"]
            valid_loss = history["val_loss"]
            logger.info(f"loss: {train_loss[-1]:.3f} | val_loss: {valid_loss[-1]:.3f} | diff: {train_loss[-1]-valid_loss[-1]:.3f}")
            
            if fold_n == 0:
                plt.figure(figsize=(16, 12))
            p = plt.plot(train_loss, label=f'train_{fold_n}')
            plt.plot(valid_loss, '--', label=f'valid_{fold_n}', color=p[0].get_color())
            plt.grid()
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')

        oof[valid_index] = y_pred_valid
        scores.append(np.mean(euclidean_distance(y_valid, y_pred_valid)))
        prediction += y_pred
        
        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold

    logger.info('train end')
    logger.info('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
        
    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
            plt.title('LGB Features (avg over folds)')
        
            return oof, prediction, feature_importance
        return oof, prediction
    
    else:
        return oof, prediction

# %%
def objective(X_train, y_train, model_name, trial):
    if model_name == 'RandomForestClassifier' or model_name == 'RandomForestRegressor':
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 4)
        }
        if model_name == 'RandomForestClassifier':
            params['criterion'] = trial.suggest_categorical('criterion', ['gini', 'entropy'])
            model = RandomForestClassifier(**params, n_estimators=1000, random_state=1)
        if model_name == 'RandomForestRegressor':
            params['criterion'] = trial.suggest_categorical('criterion', ['mse', 'mae'])
            model = RandomForestRegressor(**params, n_estimators=200, random_state=1)

    if model_name == 'SVC':
        params = {
            'C': trial.suggest_loguniform('C', 1e0, 1e3),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
            'gamma': trial.suggest_loguniform('gamma', 1e-3, 1e0),
        }
        if params['kernel'] == 'poly':
            params['degree'] = trial.suggest_int('degree', 3, 5)

        model = Pipeline([('scaler', StandardScaler()), ('model', SVC(**params))])

    if model_name == 'NuSVC':
        params = {
            'nu': trial.suggest_uniform('nu', 0.001, 1.0),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
            'gamma': trial.suggest_loguniform('gamma', 1e-2, 1e0),
        }
        if params['kernel'] == 'poly':
            params['degree'] = trial.suggest_int('degree', 3, 5)

        model = Pipeline([('scaler', StandardScaler()), ('model', NuSVC(**params))])

    logger.debug('current params: ')
    for key, value in params.items():
        logger.debug('    {}: {}'.format(key, value))

    # Cross validation
    if model_name == 'RandomForestRegressor':
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        spl = cv.split(X_train)
    else:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        spl = cv.split(X_train, y_train)
    list_accuracy = []
    list_mse = []
    euclidean_dists = []
    logger.debug('Cross-validation start')
    for train_idx, valid_idx in spl:
        # Extract data for validation
        x_trn = X_train.iloc[train_idx, :]
        x_val = X_train.iloc[valid_idx, :]
        y_trn = y_train[train_idx]
        y_val = y_train[valid_idx]
        logger.debug('    trn shape: {}, val shape: {}'.format(np.shape(x_trn), np.shape(x_val)))
        
        # Count labels
        unique_y_trn = np.unique(y_trn, return_counts=True)
        unique_y_val = np.unique(y_val, return_counts=True)
        logger.debug('    y_trn: {}, y_val: {}'.format(unique_y_trn, unique_y_val))

        # Training
        model.fit(x_trn, y_trn)

        # Predict result
        y_pred = model.predict(x_val)

        # for regressor
        if model_name == 'RandomForestRegressor':
            mse = mean_squared_error(y_val, y_pred)
            list_mse.append(mse)
            euclidean_dist = euclidean_distance(y_val, y_pred)
            euclidean_dists.append(np.mean(euclidean_dist))
        # for classifier
        else:
            accuracy = accuracy_score(y_val, y_pred)
            list_accuracy.append(accuracy)
            logger.debug('    accuracy: {}'.format(accuracy))
    logger.debug('Cross-validation end')

    if model_name == 'RandomForestRegressor':
        mean_mse = np.mean(list_mse)
        logger.debug('mean of mse: {}'.format(mean_mse))
        mean_euclidean_dist = np.mean(euclidean_dists)
        logger.debug('mean of euclidean dist: {}'.format(euclidean_dists))

        return mean_euclidean_dist
    else:
        mean_accuracy = np.mean(list_accuracy)
        logger.debug('mean of accuracy: {}'.format(mean_accuracy))

        return 1 - mean_accuracy

# %%
def model_tuning(X_train, y_train, model_name, n_trials=10):
    # Parameter tuning with optuna
    logger.info('tuning start')
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=11))
    study.optimize(partial(objective, X_train, y_train, model_name), n_trials=n_trials)
    logger.info('tuning end')

    logger.info('Number of finished trials: {}'.format(len(study.trials)))
    logger.info('Best trial:')
    trial = study.best_trial
    logger.info('    Value: {}'.format(trial.value))
    logger.info('    params: ')
    for key, value in trial.params.items():
        logger.info('        {}: {}'.format(key, value))

    # for classifier
    if model_name == 'RandomForestClassifier':
        model = RandomForestClassifier(**trial.params, n_estimators=1000, random_state=1)
    if model_name == 'SVC':
        model = Pipeline([('scaler', StandardScaler()), ('model', SVC(**trial.params))])
    if model_name == 'NuSVC':
        model = Pipeline([('scaler', StandardScaler()), ('model', NuSVC(**trial.params))])
    # for regressor
    if model_name == 'RandomForestRegressor':
        model = RandomForestRegressor(**trial.params, n_estimators=1000, random_state=1)

    return model, trial.params

# %%
def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data(y_cols=['Position_x', 'Position_y'], test_size=0.4, is_stratify=False, random_state=2)

    # Training
    model, _ = model_tuning(X_train, y_train, model_name='RandomForestRegressor', n_trials=50)

    # Best params (for classifier)
    # model = RandomForestClassifier(n_estimators=1000, max_features='sqrt', criterion='gini')
    # params = {
    #     'C': 1.1820939900623273,
    #     'kernel': 'rbf',
    #     'gamma': 0.1977483570545047
    # }
    # model = Pipeline([('scaler', StandardScaler()), ('model', SVC(**params))])
    _, y_pred = train_regressor_model(X=X_train, X_test=X_test, y=y_train, model_type='sklearn', model=model)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    euclidean_dist = euclidean_distance(y_test, y_pred)
    logger.info('RESULT:')
    logger.info('    MSE: {}'.format(mse))
    logger.info('    MAE: {}'.format(mae))
    logger.info('    Median of Euclidean distance: {}'.format(np.median(euclidean_dist)))
    logger.info('    Params: {}'.format(model.get_params()))
    # for key, value in params():
    #     logger.info('        {}: {}'.format(key, value))

# %%
if __name__ == "__main__":
    logger = get_logger(__name__)
    main()
