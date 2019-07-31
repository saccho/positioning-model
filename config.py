import os

MEASURED_DAY = '20190726'
# DATA_FILE_PATH = os.path.join('data', MEASURED_DAY, 'dataset.csv')
# DATA_COL_NAMES = [
#     'Position', 
#     r'$\it{P_{1, 1}}$', r'$\it{P_{2, 1}}$', r'$\it{P_{3, 1}}$',
#     r'$\it{P_{1, 2}}$', r'$\it{P_{2, 2}}$', r'$\it{P_{3, 2}}$',
# ]
DATA_FILE_PATH = os.path.join('data', MEASURED_DAY, 'y_dataset.csv')
DATA_COL_NAMES = [
    'Position_x', 'Position_y', 
    r'$\rm{Re}(\it{y_{1, 1}})$', r'$\rm{Re}(\it{y_{2, 1}})$', r'$\rm{Re}(\it{y_{3, 1}})$',
    r'$\rm{Re}(\it{y_{1, 2}})$', r'$\rm{Re}(\it{y_{2, 2}})$', r'$\rm{Re}(\it{y_{3, 2}})$',
    r'$\rm{Im}(\it{y_{1, 1}})$', r'$\rm{Im}(\it{y_{2, 1}})$', r'$\rm{Im}(\it{y_{3, 1}})$',
    r'$\rm{Im}(\it{y_{1, 2}})$', r'$\rm{Im}(\it{y_{2, 2}})$', r'$\rm{Im}(\it{y_{3, 2}})$',
]
LOG_FILE_PATH = os.path.join('log', f'use_{MEASURED_DAY}_data.log')
