import os

MEASURED_DAY = '20190626'
DATA_FILE_PATH = os.path.join('data', MEASURED_DAY, 'dataset.csv')
DATA_COL_NAMES = [
    'Position', 
    r'$\it{P_{-42, 1}}$', r'$\it{P_{0, 1}}$', r'$\it{P_{42, 1}}$',
    r'$\it{P_{-42, 2}}$', r'$\it{P_{0, 2}}$', r'$\it{P_{42, 2,}}$',
]
LOG_FILE_PATH = os.path.join('log', f'use_{MEASURED_DAY}_data.log')
