import logging
from os import path
import pickle

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import sklearn
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import statsmodels.api as sm
from patsy import dmatrices
import spotipy
import spotipy.oauth2 as oauth2
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util
from datetime import datetime

#import spotifySetup

# global variables -- ALL CAPS

# Create a logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# -------------------------------------------- DATA LOADING ---------------------------------------------------------
def loadData(filename, featDict):
    df = pd.read_csv(filename, dtype=featDict)
    df = df.replace('?', np.NaN) # replace unknown parameters with NaN
    tempDf = df.dropna() # skip rows with missing data
    df = tempDf
    return df

# ------------------------------------------- DATA VISUALIZATION -------------------------------------------------------
def plotHist(df, var_list, plots, x, y, title):
    for i in range(x):
        for k in range(y):
            plots[i, k].hist(df[var_list[i + k]])
            plots[i, k].set_title(var_list[i + k])
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()

# ----------------------------------------- FEATURE REDUCTION: PCA -----------------------------------------------------
def runPCA(df, n, start_col, end_col):
    pca = PCA(n_components=n)
    y = pca.fit_transform(df.iloc[:, start_col:end_col])
    return y

# -------------------------------------------------- RUN ---------------------------------------------------------------

def main():
    data_csv = 'data.csv'
    # note: always good practice to specify dtype when importing large data sets; otherwise, python will need to guess
    feats = {'acousticness': 'float64', 'artists': 'O', 'danceability': 'float64', 'duration_ms': 'uint32',
             'energy': 'float64', 'explicit': '?', 'id': 'O', 'instrumentalness': 'float64', 'key': 'uint32',
             'liveness': 'float64', 'loudness': 'float64', 'mode': '?', 'name': 'O', 'popularity': 'uint32',
             'release_date': 'O', 'speechiness': 'float64', 'tempo': 'float64', 'valence': 'float64', 'year': 'O'}

    df_all = loadData(data_csv, feats)
    num_list = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness',
                 'mode', 'tempo', 'valence', 'name', 'artists', 'id']
    # copy data into new dataset with all the variables we actually care about
    df = df_all[num_list].copy()


    # visualize numeric data
    fig1, axs = plt.subplots(3, 3)
    plotHist(df, num_list, axs, 3, 3, "Raw Data Histograms")

    # normalize all data and plot
    for i in range(9):
        df[num_list[i]] = preprocessing.scale(df[num_list[i]])

    fig2, axs = plt.subplots(3, 3)
    plotHist(df, num_list, axs, 3, 3, "Normalized Data Histograms")

    # feature dimensionality reduction: PCA
    y = runPCA(df, 2 ,0 , -3)

    plt.scatter(y[:,0], y[:,1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

    # kmeans clustering
    silhouette = []
    kmeans_kwargs = {'init': 'random',
                     'n_init': 10,
                     'max_iter': 300,
                     'random_state': 42,}

    # figuring out how many clusters we want
    for k in range(2,13):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(y)
        silhouette.append(kmeans.inertia_)

    plt.style.use('fivethirtyeight')
    plt.plot(range(2,10), silhouette)
    plt.xticks(range(2,10))
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette')
    plt.show()

    # for this problem though, we just want as many distinct genres as we care about; I'll say 4 for now
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(y)
    y_kmeans = kmeans.predict(y)

    plt.scatter(y[:,0], y[:,1], c=y_kmeans, s=50, cmap = 'viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5)
    plt.show()

    # do we agree with these clusters? let's take a listen

main()