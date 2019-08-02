import os

MEASURED_DAY = '20190726'
DATA_FILE_PATH = os.path.join('data', MEASURED_DAY)
DATASET_FILE_PATH = os.path.join(DATA_FILE_PATH, 'y_dataset.csv')
DATA_COL_NAMES = [
    'Position_x', 'Position_y', 
    r'$\rm{Re}(\it{y_{1, 1, 1}})$', r'$\rm{Re}(\it{y_{2, 1, 1}})$', r'$\rm{Re}(\it{y_{3, 1, 1}})$',
    r'$\rm{Re}(\it{y_{1, 2, 1}})$', r'$\rm{Re}(\it{y_{2, 2, 1}})$', r'$\rm{Re}(\it{y_{3, 2, 1}})$',
    r'$\rm{Im}(\it{y_{1, 1, 1}})$', r'$\rm{Im}(\it{y_{2, 1, 1}})$', r'$\rm{Im}(\it{y_{3, 1, 1}})$',
    r'$\rm{Im}(\it{y_{1, 2, 1}})$', r'$\rm{Im}(\it{y_{2, 2, 1}})$', r'$\rm{Im}(\it{y_{3, 2, 1}})$',
    r'$\rm{Re}(\it{y_{1, 1, 2}})$', r'$\rm{Re}(\it{y_{2, 1, 2}})$', r'$\rm{Re}(\it{y_{3, 1, 2}})$',
    r'$\rm{Re}(\it{y_{1, 2, 2}})$', r'$\rm{Re}(\it{y_{2, 2, 2}})$', r'$\rm{Re}(\it{y_{3, 2, 2}})$',
    r'$\rm{Im}(\it{y_{1, 1, 2}})$', r'$\rm{Im}(\it{y_{2, 1, 2}})$', r'$\rm{Im}(\it{y_{3, 1, 2}})$',
    r'$\rm{Im}(\it{y_{1, 2, 2}})$', r'$\rm{Im}(\it{y_{2, 2, 2}})$', r'$\rm{Im}(\it{y_{3, 2, 2}})$',
]
LOG_FILE_PATH = os.path.join('log', f'use_{MEASURED_DAY}_data.log')
ROOM_IMAGE_PATH = os.path.join('data', 'room.png')
FIGURE_SAVE_PATH = os.path.join(DATA_FILE_PATH, 'fig')
TRAINED_CLF_MODEL_PATH = os.path.join(DATA_FILE_PATH, 'model', 'classifier')
TRAINED_REG_MODEL_PATH = os.path.join(DATA_FILE_PATH, 'model', 'regressor')
