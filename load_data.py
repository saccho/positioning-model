import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from logger import get_logger
from config import DATA_FILE_PATH, DATA_COL_NAMES

logger = get_logger(__name__)

def load_data(isdrop_delay=False, test_size=0.45, is_stratify=True):
    data_df = load_measured_data(isdrop_delay)

    logger.debug('split for training and testing')
    data = data_df.values
    X = data[:, 1:]
    y = data[:, 0]

    if is_stratify:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    X_train_df = pd.DataFrame(X_train, columns=data_df.columns[1:])
    X_test_df = pd.DataFrame(X_test, columns=data_df.columns[1:])

    logger.info('train shape: {}, test shape: {}'.format(np.shape(X_train), np.shape(X_test)))
    # Count labels
    unique_y_train = np.unique(y_train, return_counts=True)
    unique_y_test = np.unique(y_test, return_counts=True)
    logger.info('    y_train: {}, y_test: {}'.format(unique_y_train, unique_y_test))

    return X_train_df, X_test_df, y_train, y_test

def load_measured_data(isdrop_delay=False):
    logger.debug('measured data loading')
    data_df = pd.read_csv(DATA_FILE_PATH, names=DATA_COL_NAMES)
    logger.debug('done')
    
    if isdrop_delay == True:
        return data_df.drop(columns=DATA_COL_NAMES[4:])
    else:
        return data_df
