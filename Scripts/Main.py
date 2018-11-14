import sys
import matplotlib.pyplot as plt
import pandas as pd
from Data_Preprocessing import Data_Preprocessing
from Data_Output import Data_Output
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering


def main():

    # if len(sys.argv) != 4:
    #     print("Wrong command format, please follwoing the command format below:")
    #     print("python dbscan-template.py data_filepath eps minpts")
    #     exit(0)

    ## Get information of clinical measurements and then transform it into user_vectors
    dp = Data_Preprocessing(prefix='../../../data/')
    dp.read_measurements()
    user_vectors = dp.get_user_vectors(max_labels=6)
    uv_df = pd.DataFrame(user_vectors).T.dropna(axis=0, how='all')
    # TODO: fill null values with mean of the corresponding column
    uv_df = uv_df.fillna(uv_df.mean())

    ## Get information of clinical diagnoses and get y_acutal
    dp.read_diagnoses()
    y_actual_df = dp.get_labels()
    all_users_df = pd.DataFrame({'SUBJECT_ID': uv_df.index})
    y_actual_df = pd.merge(all_users_df, y_actual_df, how='left',on='SUBJECT_ID')
    y_actual_df = y_actual_df.fillna(0)

    ## TODO: Train KMeans model -> get y_pred
    y_pred = KMeans(n_clusters=2, random_state=0).fit_predict(uv_df,y_actual_df['LABEL'])
    ## Output KMeans graph
    dout = Data_Output('../../../data/')
    dout.plot_scatters(uv_df,y_pred, y_actual_df['LABEL'],'KMeans',0)

    ## TODO: Spectral
    y_pred_spec= SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=0).fit_predict(uv_df, y_actual_df['LABEL'])
    ## Output KMeans graph
    dout.plot_scatters(uv_df,y_pred_spec, y_actual_df['LABEL'],'Spectral',0)

    # X = read_data(sys.argv[1])

    # Compute DBSCAN
    # db = dbscan(X, float(sys.argv[2]), int(sys.argv[3]))

    # store output labels returned by your algorithm for automatic marking

    # with open('.' + os.sep + 'Output' + os.sep + 'labels.txt', "w") as f:
    #     for e in db[0]:
    #         f.write(str(e))
    #         f.write('\n')

    # store output core sample indexes returned by your algorithm for automatic marking
    # with open('.' + os.sep + 'Output' + os.sep + 'core_sample_indexes.txt', "w") as f:
    #     for e in db[1]:
    #         f.write(str(e))
    #         f.write('\n')
    #
    # _, dimension = X.shape

    # plot the graph is the data is dimensiont 2
    # if dimension == 2:
    #     core_samples_mask = np.zeros_like(np.array(db[0]), dtype=bool)
    #     core_samples_mask[db[1]] = True
    #     labels = np.array(db[0])
    #
    #     # Number of clusters in labels, ignoring noise if present.
    #     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #
    #     # Black removed and is used for noise instead.
    #     unique_labels = set(labels)
    #     colors = [plt.cm.Spectral(each)
    #               for each in np.linspace(0, 1, len(unique_labels))]
    #
    #     for k, col in zip(unique_labels, colors):
    #         if k == -1:
    #             # Black used for noise.
    #             col = [0, 0, 0, 1]
    #
    #         class_member_mask = (labels == k)
    #
    #         xy = X[class_member_mask & core_samples_mask]
    #         plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #                  markeredgecolor='k', markersize=14)
    #
    #         xy = X[class_member_mask & ~core_samples_mask]
    #         plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #                  markeredgecolor='k', markersize=6)
    #
    #     plt.title('Estimated number of clusters: %d' % n_clusters_)
    #     plt.savefig('.' + os.sep + 'Output' + os.sep + 'cluster-result.png')

main()