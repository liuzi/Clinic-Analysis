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



main()