# %%
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import seaborn as sns

from load_data import load_measured_data
from logger import get_logger

from config import ROOM_IMAGE_PATH, FIGURE_SAVE_PATH

logger = get_logger(__name__)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'stix'
sns.set_context(context='paper', font_scale=4)
room_image = Image.open(ROOM_IMAGE_PATH)

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
    logger.debug('plot feature correlation')
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
    
    obj_cols = []
    obj_df = pd.DataFrame()
    num_df = pd.DataFrame()
    
    logger.debug('plot numerical features')
    for col in cols:
        if data_df[col].dtypes != 'O':
            num_df[col] = data_df[col]
        else:
            obj_cols.append(col)
            obj_df[col] = data_df[col]
            
        if num_df.shape[1] == 6:
            histplot_features(num_df)
            boxplot_features(num_df)
            num_df = pd.DataFrame()

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

def plot_confusion_matrix(cmx, is_save=False):
    logger.debug('plot confusion matrix')
    sns.set(font_scale=3.0)
    colormap = plt.get_cmap('Blues')
    plt.figure(figsize=(10, 10))
    fig = sns.heatmap(cmx, linewidths=0.1, vmin=0, vmax=cmx.max(), 
                square=True, cmap=colormap, linecolor='white', annot=True)
    fig.set_xticklabels(fig.get_xticklabels(), rotation=0)
    fig.set_yticklabels(fig.get_yticklabels(), rotation=0)
    plt.tight_layout()
    if is_save:
        plt.savefig(os.path.join(FIGURE_SAVE_PATH, 'confusion_matrix.png'))
    plt.show()

def pred_animation(y_test, y_pred, is_save=False):
    fig = plt.figure(figsize=(15, 10))
    ims = []
    plt.imshow(room_image, extent=(0, 7.8, 0, 8.8))
    # Plot antenna position
    plt.scatter(7.25, 1.5, color='green', marker=7, s=500)
    plt.scatter(7.2, 7.1, color='green', marker=7, s=500)
    for m in range(3):
        plt.scatter(0.6, 2.763-m*0.063, color='cyan', marker=7, s=500)
    # Plot true position and predicted position (Animation)
    for i in range(len(y_test)):
        im_test = [plt.scatter(y_test[i, 0], y_test[i, 1], color='black', marker='X', s=1200)]
        im_pred = [plt.scatter(y_pred[i, 0], y_pred[i, 1], color='blue', s=1200, alpha=0.7)]
        ims.append(im_test+im_pred)
    plt.xlim(0, 7.8)
    plt.ylim(0, 8.8)
    ani = ArtistAnimation(fig, ims, interval=500)
    if is_save:
        ani.save(os.path.join(FIGURE_SAVE_PATH, 'pred_anim.mp4'), writer="ffmpeg")
    plt.show()

def plot_score(y_test, y_pred_score, is_save=False):
    plt.figure(figsize=(15, 10))
    plt.imshow(room_image, extent=(0, 7.8, 0, 8.8))
    # Plot antenna position
    plt.scatter(7.25, 1.5, color='green', marker=7, s=500)
    plt.scatter(7.2, 7.1, color='green', marker=7, s=500)
    for m in range(3):
        plt.scatter(0.6, 2.763-m*0.063, color='cyan', marker=7, s=500)
    plt.scatter(y_test[:, 0], y_test[:, 1], c=y_pred_score, cmap='hot_r', s=1200)
    cbar = plt.colorbar()
    cbar.set_label('Error (m)')
    plt.xlim(0, 7.8)
    plt.ylim(0, 8.8)
    plt.xlabel(r'$x$ (m)')
    plt.ylabel(r'$y$ (m)')
    if is_save:
        plt.savefig(os.path.join(FIGURE_SAVE_PATH, 'score.png'))
    plt.show()

def plot_score_cdf(y_pred_score, is_save=False):
    plt.figure(figsize=(10, 10))
    sorted_score = np.sort(y_pred_score)
    cdf = np.arange(len(sorted_score))/float(len(sorted_score)-1)
    plt.plot(sorted_score, cdf, lw=3, marker='o', ms=12)
    plt.grid(True)
    plt.xlim(0, y_pred_score.max())
    plt.ylim(0, 1)
    plt.xlabel('Error (m)')
    plt.ylabel('Cumulative probability')
    if is_save:
        plt.savefig(os.path.join(FIGURE_SAVE_PATH, 'score_cdf.png'))
    plt.show()

def plot_feature_importance(feature_importance, top=50, is_save=False):
    if len(feature_importance) == 0:
        return

    cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:top].index

    best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

    plt.figure(figsize=(15, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.tight_layout()
    if is_save:
        plt.savefig(os.path.join(FIGURE_SAVE_PATH, 'feature_importance.png'))
    plt.show()

# %%
if __name__ == "__main__":
    data_df = load_measured_data()
    plot_corr_features(data_df)
    features_frequency(data_df.drop(columns='Position'))
