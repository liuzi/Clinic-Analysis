from Data_Cleaning.Abstract import Abstract
from Data_Cleaning.Labevents import Labevents
from Data_Cleaning.Prescriptions import Prescriptions
from sklearn.cluster import KMeans

aa = Abstract()
# user_list = ll.read_data('temp/PATIENTS_5_PER')['SUBJECT_ID']
# lab_user_vectors = ll.get_uservectors(user_list,ll.read_measurements_data(user_list))
# pp = Prescriptions.Prescriptions()
# pres_user_vectors = pp.to_attributes(pp.read_prescriptions_data(user_list))
## TODO: only trial, need to be removed later
# lab_user_vectors = ll.read_data('temp/labtest_uservectors').sort_values(by = 'SUBJECT_ID')
# pres_user_vectors = ll.read_data('temp/prestest_uservectors').sort_values(by = 'SUBJECT_ID')

## Combine vectors from different dataset
# joined_user_vectors = ll.inner_join(lab_user_vectors,pres_user_vectors,'SUBJECT_ID')
joined_df = aa.read_data('temp/joined_user_vectors').sort_values(by = 'SUBJECT_ID')
joined_df.columns = ['SUBJECT_ID'] + list(map(str,range(joined_df.shape[1]-1)))
aa.write2file(joined_df,'joined_user_vectors_resetindex')
# kmeans = KMeans(n_clusters=2).fit(joined_df)
# aa.write2file(kmeans.labels_,'predicted_diagnosis')

