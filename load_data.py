import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from logger import get_logger
from config import DATASET_FILE_PATH, DATA_COL_NAMES

logger = get_logger(__name__)

def load_data(y_cols=('Position',), is_drop_delay=False, test_size=0.45, is_stratify=True, random_state=0):
    data_df = load_measured_data(is_drop_delay)

    logger.debug('split for training and testing')
    X_cols = [c for c in DATA_COL_NAMES if c not in y_cols]
    X = data_df.loc[:, X_cols].values
    y = data_df.loc[:, y_cols].values

    if is_stratify:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train_df = pd.DataFrame(X_train, columns=X_cols)
    X_test_df = pd.DataFrame(X_test, columns=X_cols)

    logger.info('train shape: {}, test shape: {}'.format(np.shape(X_train), np.shape(X_test)))
    # Count labels
    unique_y_train = np.unique(y_train, return_counts=True)
    unique_y_test = np.unique(y_test, return_counts=True)
    logger.info('    y_train: {}, y_test: {}'.format(unique_y_train, unique_y_test))

    return X_train_df, X_test_df, y_train, y_test

def load_measured_data(is_drop_delay=False):
    logger.debug('measured data loading')
    data_df = pd.read_csv(DATASET_FILE_PATH, names=DATA_COL_NAMES)
    logger.debug('done')
    
    if is_drop_delay == True:
        return data_df.drop(columns=DATA_COL_NAMES[4:])
    else:
        return data_df
