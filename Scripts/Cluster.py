from Data_Cleaning.Abstract import Abstract
from Data_Cleaning.Labevents import Labevents
from Data_Cleaning.Prescriptions import Prescriptions
from Data_Cleaning.Diagosis import Diagnosis
from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd

# aa = Abstract()
# user_list = ll.read_data('temp/PATIENTS_5_PER')['SUBJECT_ID']
# lab_user_vectors = ll.get_uservectors(user_list,ll.read_measurements_data(user_list))
# pp = Prescriptions.Prescriptions()
# pres_user_vectors = pp.to_attributes(pp.read_prescriptions_data(user_list))
## TODO: only trial, need to be removed later
# lab_user_vectors = ll.read_data('temp/labtest_uservectors').sort_values(by = 'SUBJECT_ID')
# pres_user_vectors = ll.read_data('temp/prestest_uservectors').sort_values(by = 'SUBJECT_ID')

## Combine vectors from different dataset
# joined_user_vectors = ll.inner_join(lab_user_vectors,pres_user_vectors,'SUBJECT_ID')
'''
joined df
'''
# joined_df = aa.read_data('temp/joined_user_vectors').sort_values(by = 'SUBJECT_ID')
# joined_df.columns = ['SUBJECT_ID'] + list(map(str,range(joined_df.shape[1]-1)))
# aa.write2file(joined_df,'joined_user_vectors_resetindex')
''''''
# kmeans = KMeans(n_clusters=2).fit(joined_df)
# aa.write2file(kmeans.labels_,'predicted_diagnosis')

'''
    rand_index: measures the similarity of the two assignments, ignoring permutations and with chance normalization
    Mutual Information: a function that measures the agreement of the two assignments, ignoring permutations.
    homogeneity: each cluster contains only members of a single class.
    completeness: all members of a given class are assigned to the same cluster。
'''
def cluster_metrics(pred_col, true_col):
    rand_score = metrics.adjusted_rand_score(true_col,pred_col)
    mutual_info = metrics.mutual_info_score(true_col,pred_col)
    homo_score = metrics.homogeneity_score(true_col,pred_col)
    comp_score = metrics.completeness_score(true_col,pred_col)
    return [rand_score, mutual_info, homo_score, comp_score]

metrics_columns = ["Adjusted_Rand_Index","Mutual_Information","Homogeneity", "Completeness"]

diagnosis = Diagnosis()
all_diagnoses = diagnosis.get_labels()
labevents = Labevents()
lab_metrics = []
for i in range(0,10):
    selected_user_list = diagnosis.create_selected_users("PATIENTS_RAND%s" % i)
    ## Transform raw data in LABEVENTS into user vectors of clinical features
    lab_user_vectors = labevents.get_uservectors(selected_user_list,labevents.read_measurements_data(selected_user_list))
    ## Split user vectors into the user list and pure vectors
    lab_user_invec, lab_user_purevec = lab_user_vectors['SUBJECT_ID'], lab_user_vectors.drop('SUBJECT_ID',1)
    ## Run cluster model
    lab_kmeans = KMeans(n_clusters=2).fit(lab_user_purevec)
    lab_pred_df = pd.DataFrame({'SUBJECT_ID':lab_user_invec,'PRED':lab_kmeans.labels_})
    lab_valid_df = labevents.left_join(lab_pred_df,all_diagnoses,'SUBJECT_ID')
    lab_valid_df[['TRUE']] = lab_valid_df[['LABEL']].fillna(0)
    lab_metrics.append(cluster_metrics(lab_valid_df['TRUE'],lab_valid_df['PRED']))

print(lab_metrics)

