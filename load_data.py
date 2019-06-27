import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from logger import get_logger

logfile_path = os.path.join('log', 'model.log')
logger = get_logger(__name__, logfile_path)

def load_data():
    logger.debug('enter')
    measured_day = '20190626'
    datafile_path = os.path.join('data', measured_day, 'dataset.csv')
    names = [
        'Position', 
        'P_-42_1', 'P_0_1', 'P_42_1',
        'P_-42_2', 'P_0_2', 'P_42_2',
    ]
    data_df = pd.read_csv(datafile_path, names=names)

    data = data_df.values
    X = data[:, 1:]
    y = data[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=0)

    X_train_df = pd.DataFrame(X_train, columns=names[1:])
    X_test_df = pd.DataFrame(X_test, columns=names[1:])

    logger.debug('train shape: {}, test shape: {}'.format(np.shape(X_train), np.shape(X_test)))
    logger.debug('exit')
    return X_train_df, X_test_df, y_train, y_test
