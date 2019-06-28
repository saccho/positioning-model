# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_measured_data
from logger import get_logger

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'stix'
scatter_size = 150
sns.set_context(context='paper', font_scale=2.5)
cmap = plt.get_cmap('tab20')

# %%
def misclass(y_test, y_pred):
    y_misclass = []
    indices_misclass = []
    for i in range(len(y_pred)):
        if y_test[i] != y_pred[i]:
            misclass = y_pred[i]
            y_misclass.append(misclass)
            indices_misclass.append(i)
    return indices_misclass, y_misclass

def plot_corr_features(data_df):
    colormap = plt.get_cmap('seismic')
    plt.figure(figsize=(10, 10))
    plt.title('Correlation of Features', y=1.05, size=20)
    fig = sns.heatmap(data_df.corr(), linewidths=0.1, vmin=-1.0, vmax=1.0, 
                square=True, cmap=colormap, linecolor='white', annot=True)
    fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
    fig.set_yticklabels(fig.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.show()

def histplot_features(data_df):
    cols = data_df.columns
    
    plt.figure(figsize=(18, 13))
    for i, col in enumerate(cols):
        ser = data_df[col]
        plt.subplot(2, 3, i+1)
        plt.hist(ser, bins=100)
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.title(col)
        # plt.title('{}(nan: {:.2f}%)'.format(col, 100 * ser.isnull().sum() / ser.shape[0]))
    plt.tight_layout()
    plt.show()
    
def boxplot_features(data_df):
    cols = data_df.columns
    
    plt.figure(figsize=(18, 13))
    for i, col in enumerate(cols):
        ser = data_df[col]
        plt.subplot(2, 3, i+1)
        sns.boxplot(ser)
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.title(col)
        # plt.title('{}(nan: {:.2f}%)'.format(col, 100 * ser.isnull().sum() / ser.shape[0]))
    plt.tight_layout()
    plt.show()

def features_frequency(data_df):
    cols = data_df.columns
    len_cols = len(cols)
    
    obj_cols = []
    obj_df = pd.DataFrame()
    num_df = pd.DataFrame()
    
    j = 0
    logger.debug('plot numerical features')
    while j < len_cols:
        if data_df[cols[j]].dtypes != 'O':
            num_df[cols[j]] = data_df[cols[j]]
        else:
            obj_cols.append(cols[j])
            obj_df[cols[j]] = data_df[cols[j]]
            
        if num_df.shape[1] == 6:
            histplot_features(num_df)
            boxplot_features(num_df)
            num_df = pd.DataFrame()
            
        j += 1
        
    if num_df.shape[1] != 0:
        histplot_features(num_df)
        boxplot_features(num_df)
    
    if len(obj_cols) != 0:
        logger.debug('objects features\n')
        for obj_col in obj_cols:
            vc_obj_key = obj_df[obj_col].value_counts(normalize=True, dropna=False).index
            vc_obj_val = 100 * obj_df[obj_col].value_counts(normalize=True, dropna=False).values
            logger.debug(obj_col)
            for k, v in zip(vc_obj_key, vc_obj_val):
                logger.debug(f'    {k}: {v:.2f}%')

# %%
if __name__ == "__main__":
    logger = get_logger(__name__)
    data_df = load_measured_data()
    plot_corr_features(data_df)
    features_frequency(data_df.drop(columns='Position'))
