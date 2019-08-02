# %%
import os
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
from sklearn.externals import joblib

from load_data import load_data
from logger import get_logger
from figure import plot_confusion_matrix, pred_animation, plot_score, plot_score_cdf, plot_feature_importance
from config import TRAINED_CLF_MODEL_PATH, TRAINED_REG_MODEL_PATH

# %%
def euclidean_distance(y_test, y_pred):
    return np.sqrt((y_test[:, 0] - y_pred[:, 0])**2 + (y_test[:, 1] - y_pred[:, 1])**2)

# %%
def objective(X_train, y_train, model_name, trial):
    if model_name == 'RandomForestClassifier' or model_name == 'RandomForestRegressor':
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 5),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
        }
        if model_name == 'RandomForestClassifier':
            params['criterion'] = trial.suggest_categorical('criterion', ['gini', 'entropy'])
            model = RandomForestClassifier(**params, n_estimators=200, random_state=1)
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


class Saver:
    def __init__(self, est_type):
        if est_type == 'classifier':
            self.root = TRAINED_CLF_MODEL_PATH
        elif est_type == 'regressor':
            self.root = TRAINED_REG_MODEL_PATH
        else:
            raise ValueError('est_type must be "classifier" or "regressor".')

    def save_model(self, model, fold_n=None):
        if fold_n is None:
            path = os.path.join(self.root, 'model.pkl')
        else:
            path = os.path.join(self.root, f'model_fold{fold_n}.pkl')

        joblib.dump(model, path)
        
    def restore_model(self, fold_n=None):
        if fold_n is None:
            path = os.path.join(self.root, 'model.pkl')
        else:
            path = os.path.join(self.root, f'model_fold{fold_n}.pkl')

        try:
            model = joblib.load(path)
        except IOError:
            model = None

        return model

    def save_params(self, params):
        path = os.path.join(self.root, 'params.json')
        pd.Series(params).to_json(path)
        
    def restore_params(self):
        path = os.path.join(self.root, 'params.json')
        try:
            params = pd.read_json(path, typ='series').to_dict()
        except ValueError:
            params = None

        return params

    def save_feature_importance(self, feature_importance):
        path = os.path.join(self.root, 'feature_importance.json')
        try:
            feature_importance.to_json(path, orient='records')
        except AttributeError:
            return

    def restore_feature_importance(self):
        path = os.path.join(self.root, 'feature_importance.json')
        try:
            feature_importance = pd.read_json(path)
        except ValueError:
            feature_importance = None

        return feature_importance


# %%
class ClassifierModel:
    def __init__(self, model_type='sklearn', model_name=None, n_folds=3):
        self.est_type = 'classifier'
        self.saver = Saver(self.est_type)
        self.model_type = model_type
        if self.model_type == 'sklearn':
            if model_name is None:
                self.model_name = 'RandomForestClassifier'
        else:
            raise ValueError('model_type supported by current version is "sklearn" only.')

        self.params = self.saver.restore_params()

        if n_folds == 0:
            self.is_ensemble = False
            self.model = self.saver.restore_model()
        else:
            self.is_ensemble = True
            self.n_folds = n_folds

            self.models = []
            for fold_n in range(self.n_folds):
                self.model = self.saver.restore_model(fold_n)
                self.models.append(self.model)
        
        self.feature_importance = self.saver.restore_feature_importance()

    def load_data(self, y_cols=['Position'], isdrop_delay=False, test_size=0.4, is_stratify=True, random_state=2):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            load_data(y_cols=y_cols, test_size=test_size, is_stratify=is_stratify, random_state=random_state)

    def _param_tuning(self):
        _, self.params = model_tuning(self.X_train, self.y_train, self.model_name, n_trials=20)

    def train(self, is_override_model=False, is_override_params=False):
        if self.model is not None and not is_override_model:
            return

        if self.params is None or is_override_params:
            self._param_tuning()
            self.saver.save_params(self.params)

        if self.is_ensemble:
            self.oof = np.zeros(len(self.X_train))
            scores = []
            self.feature_importance = pd.DataFrame()
            folds = KFold(n_splits=self.n_folds, shuffle=True, random_state=11)
            for fold_n, (train_index, valid_index) in enumerate(folds.split(self.X_train)):
                logger.info(f'    Fold {fold_n}, started at {time.ctime()}')
                X_trn, X_val = self.X_train.iloc[train_index], self.X_train.iloc[valid_index]
                y_trn, y_val = self.y_train[train_index], self.y_train[valid_index]

                self._fit(X_trn, y_trn)
                self.models[fold_n] = self.model
                y_pred_val = self.model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred_val)
                score = np.median(accuracy)
                logger.info(f'    Fold {fold_n}. Accuracy: {score:.4f}.')
                scores.append(score)
                self.oof[valid_index] = y_pred_val

                # feature importance
                fold_importance = pd.DataFrame()
                fold_importance["feature"] = X_trn.columns
                fold_importance["importance"] = self.model.feature_importances_
                fold_importance["fold"] = fold_n + 1
                self.feature_importance = pd.concat([self.feature_importance, fold_importance], axis=0)

                self.saver.save_model(self.model, fold_n)

            self.feature_importance["importance"] /= self.n_folds
            self.saver.save_feature_importance(self.feature_importance)

            logger.info('train end')
            logger.info('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
        else:
            self._fit(self.X_train, self.y_train)
            # feature importance
            self.feature_importance = pd.DataFrame()
            self.feature_importance["feature"] = self.X_train.columns
            self.feature_importance["importance"] = self.model.feature_importances_
            
            self.saver.save_model(self.model)
            self.saver.save_feature_importance(self.feature_importance)

    def _fit(self, X_train, y_train):
        if self.model_type == 'sklearn':
            if self.model_name == 'SVC':
                self.model = SVC(random_state=42, **self.params)
            if self.model_name == 'NuSVC':
                self.model = NuSVC(random_state=42, **self.params)
            if self.model_name == 'RandomForestClassifier':
                self.model = RandomForestClassifier(n_estimators=1000, random_state=42, **self.params)
            self.model.fit(X_train, y_train)

    def predict(self):
        if self.is_ensemble:
            y_pred = pd.DataFrame(np.zeros((len(self.X_test), self.n_folds)))
            for fold_n in range(self.n_folds):
                model = self.models[fold_n]
                y_pred.loc[:, fold_n] = model.predict(self.X_test)
            self.prediction = y_pred.mode(axis=1)[0]
        else:
            self.prediction = self.model.predict(self.X_test)

        self.pred_score = accuracy_score(self.y_test, self.prediction)

        return self.prediction

    def run(self, is_get_pred=False):
        self.load_data()
        self.train()
        y_pred = self.predict()
        if is_get_pred:
            return y_pred

    def get_dataset(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_params(self):
        return self.params

    def get_model(self):
        if self.is_ensemble:
            return self.models
        else:
            return self.model

    def get_oof(self):
        return self.oof

    def get_pred(self):
        return self.prediction

    def get_pred_score(self):
        return self.pred_score

    def get_feature_importance(self):
        return self.feature_importance


class RegressorModel:
    def __init__(self, model_type='sklearn', model_name=None, n_folds=3):
        self.est_type = 'regressor'
        self.saver = Saver(self.est_type)
        self.model_type = model_type
        if self.model_type == 'sklearn':
            if model_name is None:
                self.model_name = 'RandomForestRegressor'
            elif model_name != 'RandomForestRegressor':
                raise ValueError('model_name supported by current version is "RandomForestRegressor" only.')
        else:
            raise ValueError('model_type supported by current version is "sklearn" only.')

        self.params = self.saver.restore_params()

        if n_folds == 0:
            self.is_ensemble = False
            self.model = self.saver.restore_model()
        else:
            self.is_ensemble = True
            self.n_folds = n_folds

            self.models = []
            for fold_n in range(self.n_folds):
                self.model = self.saver.restore_model(fold_n)
                self.models.append(self.model)
        
        self.feature_importance = self.saver.restore_feature_importance()

    def load_data(self, y_cols=['Position_x', 'Position_y'], isdrop_delay=False, test_size=0.4, is_stratify=False, random_state=2):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            load_data(y_cols=y_cols, test_size=test_size, is_stratify=is_stratify, random_state=random_state)

    def _param_tuning(self):
        _, self.params = model_tuning(self.X_train, self.y_train, self.model_name, n_trials=20)

    def train(self, is_override_model=False, is_override_params=False):
        if self.model is not None and not is_override_model:
            return

        if self.params is None or is_override_params:
            self._param_tuning()
            self.saver.save_params(self.params)

        if self.is_ensemble:
            self.oof = np.zeros((len(self.X_train), 2))
            scores = []
            self.feature_importance = pd.DataFrame()
            folds = KFold(n_splits=self.n_folds, shuffle=True, random_state=11)
            for fold_n, (train_index, valid_index) in enumerate(folds.split(self.X_train)):
                logger.info(f'    Fold {fold_n}, started at {time.ctime()}')
                X_trn, X_val = self.X_train.iloc[train_index], self.X_train.iloc[valid_index]
                y_trn, y_val = self.y_train[train_index], self.y_train[valid_index]

                self._fit(X_trn, y_trn)
                self.models[fold_n] = self.model
                y_pred_val = self.model.predict(X_val)
                euclidean_dist = euclidean_distance(y_val, y_pred_val)
                score = np.median(euclidean_dist)
                logger.info(f'    Fold {fold_n}. Median of euclidean distance: {score:.4f}.')
                scores.append(score)
                self.oof[valid_index] = y_pred_val

                # feature importance
                fold_importance = pd.DataFrame()
                fold_importance["feature"] = X_trn.columns
                fold_importance["importance"] = self.model.feature_importances_
                fold_importance["fold"] = fold_n + 1
                self.feature_importance = pd.concat([self.feature_importance, fold_importance], axis=0)

                self.saver.save_model(self.model, fold_n)

            self.feature_importance["importance"] /= self.n_folds
            self.saver.save_feature_importance(self.feature_importance)

            logger.info('train end')
            logger.info('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
        else:
            self._fit(self.X_train, self.y_train)
            # feature importance
            self.feature_importance = pd.DataFrame()
            self.feature_importance["feature"] = self.X_train.columns
            self.feature_importance["importance"] = self.model.feature_importances_
            
            self.saver.save_model(self.model)
            self.saver.save_feature_importance(self.feature_importance)

    def _fit(self, X_train, y_train):
        if self.model_type == 'sklearn':
            if self.model_name == 'RandomForestRegressor':
                self.model = RandomForestRegressor(n_estimators=1000, random_state=42, **self.params)
            self.model.fit(X_train, y_train)

    def predict(self):
        if self.is_ensemble:
            self.prediction = np.zeros((len(self.X_test), 2))
            for fold_n in range(self.n_folds):
                model = self.models[fold_n]
                y_pred = model.predict(self.X_test)
                self.prediction += y_pred
            self.prediction /= self.n_folds
        else:
            self.prediction = self.model.predict(self.X_test)

        self.pred_score = euclidean_distance(self.y_test, self.prediction)

        return self.prediction

    def run(self, is_get_pred=False):
        self.load_data()
        self.train()
        y_pred = self.predict()
        if is_get_pred:
            return y_pred

    def get_dataset(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_params(self):
        return self.params

    def get_model(self):
        if self.is_ensemble:
            return self.models
        else:
            return self.model

    def get_oof(self):
        return self.oof

    def get_pred(self):
        return self.prediction

    def get_pred_score(self):
        return self.pred_score

    def get_feature_importance(self):
        return self.feature_importance


# %%
def classification():
    cm = ClassifierModel(n_folds=0)
    cm.load_data()
    X_train, X_test, y_train, y_test = cm.get_dataset()
    cm.train(is_override_model=False, is_override_params=False)
    y_pred = cm.predict()
    feature_importance = cm.get_feature_importance()

    precision = precision_score(y_test, y_pred, average=None)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=None)
    logger.info('RESULT:')
    logger.info('    Precision: {}'.format(precision))
    logger.info('    Mean of precision: {}'.format(np.mean(precision)))
    logger.info('    Accuracy: {}'.format(accuracy))
    logger.info('    F-measure: {}'.format(f1))
    logger.info('    Mean of f-measure: {}'.format(np.mean(f1)))
    logger.info('    Params: {}'.format(cm.get_params()))

    labels = np.unique(y_test)
    labels.sort()
    cmx = confusion_matrix(y_test, y_pred, labels=labels)
    logger.info('Confusion matrix:\n{}'.format(cmx))
    plot_confusion_matrix(cmx, is_save=False)
    plot_feature_importance(feature_importance, is_save=False)

def regression():
    rm = RegressorModel(n_folds=0)
    rm.load_data()
    X_train, X_test, y_train, y_test = rm.get_dataset()
    rm.train(is_override_model=False, is_override_params=False)
    y_pred = rm.predict()
    feature_importance = rm.get_feature_importance()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    euclidean_dist = euclidean_distance(y_test, y_pred)
    logger.info('RESULT:')
    logger.info('    MSE: {}'.format(mse))
    logger.info('    MAE: {}'.format(mae))
    logger.info('    Median of Euclidean distance: {}'.format(np.median(euclidean_dist)))
    logger.info('    Params: {}'.format(rm.get_params()))

    pred_animation(y_test, y_pred, is_save=False)
    plot_score(y_test, euclidean_dist, is_save=False)
    plot_score_cdf(euclidean_dist, is_save=False)
    plot_feature_importance(feature_importance, is_save=False)


# %%
if __name__ == "__main__":
    logger = get_logger(__name__)
    classification()
    # regression()
