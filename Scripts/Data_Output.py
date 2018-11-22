import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import umap
import os

class Data_Output:

    def __init__(self, workpath='../../data/'):
        self.workpath = workpath + 'Output/'

    # def plot(X_0, X_1, y_pred, y_actual):
    def plot_scatters(self, X, y_pred, y_actual, filename, para):
        ## T-SNE
        # X_embedded = TSNE(n_components=2, learning_rate=3000).fit_transform(test_nona)

        ## UMAP
        X_embedding = umap.UMAP().fit_transform(X)

        ## Combine all data into one dataframe
        df = pd.DataFrame({'X0': X_embedding[:,0],'X1':X_embedding[:,1],'Y_PRED':y_pred,'Y_ACTUAL':y_actual})

        ## Split data into two parts based on acutal y for plot different shape of dots
        true_X = df[df['Y_ACTUAL'] == 1]
        false_X = df[df['Y_ACTUAL'] == 0]

        ## Plot graph
        colors = ['b', 'r']
        plt.figure(figsize=(15, 15))
        plt.scatter(true_X['X0'], true_X['X1'], c=[colors[x] for x in true_X['Y_PRED']], marker="+",s=50)
        plt.scatter(false_X['X0'], false_X['X1'], c=[colors[x] for x in false_X['Y_PRED']], marker='v', s=50)

        plt.title("KMeans %d" % para)
        plt.savefig(self.workpath + filename +'.png')

    # def save_as_csv(self):




