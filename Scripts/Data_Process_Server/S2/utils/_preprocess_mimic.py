from _tools import *
from _path import *
# from S2.utils._tools import *
# from S2.utils._path import*
import pandas as pd
import numpy as np
from os.path import join


def to_datetime(pd_col,date_format="%Y-%m-%d %H:%M:%S"):
    return pd.to_datetime(pd_col,format=date_format)

def df2matrix(df):
    cols=df.columns
    print("Original Data:")
    print_patient_stats(df)
    ## HACK: sampling should be after creating matrix
    # df_sampled = left_join(self.hadm_sampled,df,list(self.hadm_sampled.columns))
    # print("Sampled Data:")
    # print_patient_stats(df_sampled)

    if("VALUE" not in cols):
        df["VALUE"]=1
    
    df_matrix = df.pivot_table(
        index=cols[1], columns=cols[2], values="VALUE",fill_value=0).reset_index()

    print("Number of %s: %d"%(cols[2],len(df_matrix.columns)-1))
    return df_matrix

def get_demographic_df():
    patient_df = read_data(join(read_prefix,"PATIENTS"))
    patient_df['DOB']=to_datetime(patient_df['DOB'])
    gender={"F":0,"M":1}
    patient_df['GENDER'] = patient_df['GENDER'].apply(lambda x: gender[x])
    
    admission_df = read_data(join(
        read_prefix,"ADMISSIONS"),dtype={"HADM_ID":str})[
        ['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME']]
    
    subject_hadm_df=admission_df[['SUBJECT_ID','HADM_ID']].drop_duplicates()
    
    admission_df[['ADMITTIME','DISCHTIME']]=admission_df[['ADMITTIME','DISCHTIME']].apply(to_datetime)
    ## keep the first admission record for each patient
    admission_df['LATEST_ADMITTIME'] = admission_df.groupby('SUBJECT_ID')['ADMITTIME'].transform('min')
    admission_df=admission_df[admission_df['ADMITTIME']==admission_df['LATEST_ADMITTIME']][['SUBJECT_ID','ADMITTIME','DISCHTIME']]
    
    patient_ad_df = left_join(patient_df,admission_df[['SUBJECT_ID','ADMITTIME','DISCHTIME']],'SUBJECT_ID')
    patient_ad_df['AGE']= patient_ad_df['ADMITTIME'].subtract(patient_ad_df['DOB']).dt.days//365

    patient_ad_df['LENGTH_STAY']=(patient_ad_df['DISCHTIME']-patient_ad_df['ADMITTIME']).dt.total_seconds()/86400

    write2file(patient_ad_df[['SUBJECT_ID','GENDER','AGE','EXPIRE_FLAG','LENGTH_STAY']],join(
        singledrug_featurepreprocess_prefix,'demographic_matrix'))
    write2file(subject_hadm_df,join(
        singledrug_featurepreprocess_prefix,'patient_epis_map'))

# def read_measurements_data(items_file='D_LABITEMS', measures_file='LABEVENTS'):
#     '''
#     Get measurements data with top N frequent labels.
#     :param items_file:
#     :param measures_file:
#     :return:
#     '''
#     left_columns = ['HADM_ID', 'ITEMID', 'VALUE', 'VALUENUM']
#     # right_columns = ['ITEMID', 'LABEL']
#     hadm_id, label, value = 'HADM_ID', 'LABEL', 'VALUE'
#     # link_field = 'ITEMID'
#     # top200_labels = 'temp/LAB_TOP200_LABELS'
#     # num_labels = 20

#     # most_frequent_items = self.read_data()
#     measurements_df = read_data(join(read_prefix,'LABEVENTS'),dtype={"HADM_ID":str})[left_columns]

#     measure_filter_df = measurements_df.dropna(subset=[value])
#     return measure_filter_df


def get_labmatrix(min_nulls=0.8):
    raw_measurements = read_data(
        join(read_prefix,'LABEVENTS'))[[
            'HADM_ID', 'ITEMID', 'VALUE', 'VALUENUM']].dropna(subset=['HADM_ID','VALUE'])

    user_list=raw_measurements['HADM_ID'].unique()
    

    value ='VALUE'
    item_id = 'ITEMID'
    ## For LABEL instead of ITEMID
    # label_id = 'LABEL'
    valuenum = 'VALUENUM'
    userid = 'HADM_ID'
    labels_nulls = raw_measurements.groupby(item_id)[valuenum].apply(
        lambda x: x.isnull().sum())

    # initiate all user vectors
    #         len_all_measures = len(self.measures)
    stc_ls = ['min', 'mean', 'max', 'std']
    stc_ls_len = len(stc_ls)
    user_vectors = {}
    for user in user_list:
        user_vectors[user] = np.empty([0])

    # Groupby label
    measure_label = raw_measurements.groupby(item_id)

    for label, label_df in measure_label:

        null_percentage = labels_nulls[label] * 1.0 / len(label_df)

        if (null_percentage < min_nulls):
            # pick one iteration of continuous label
            label_df = label_df.dropna(subset=[valuenum])

            missed_users = set(user_list) - set(label_df[userid].unique())
            label_df_agg = label_df.groupby(userid)[valuenum].agg(stc_ls)

            for user, row in label_df_agg.iterrows():
                if pd.isnull(row['std']): row['std'] = 0
                user_vectors[user] = np.append(user_vectors[user], [row[x] for x in stc_ls])
            for user in missed_users:
                user_vectors[user] = np.append(user_vectors[user], np.full(stc_ls_len, np.nan))

        else:
            # pick one iteration of discrete label
#                 label_df = label_df.dropna(subset=[value])
            missed_users = set(user_list) - set(label_df[userid].unique())
            label_df_agg = label_df.groupby([userid, value])[item_id].agg(['count'])
            label_df_agg_per = label_df_agg.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))

            label_df_agg_per.reset_index()
            label_matrix = pd.pivot_table(label_df_agg_per,
                                            values=['count'], index=[userid], columns=[value]).fillna(0)

            for user in label_matrix.index.get_level_values(userid):
                user_vectors[user] = np.append(user_vectors[user], label_matrix.ix[user].values)
            for user in missed_users:
                user_vectors[user] = np.append(user_vectors[user], np.full(label_matrix.shape[1], np.nan))

    ## Reomove patients with empty record
    user_vectors_notna = pd.DataFrame(user_vectors).T.dropna(axis=0,how='all')
    ## Imputation: with mean
    user_final_vectors = user_vectors_notna.fillna(user_vectors_notna.mean())
    
    user_final_vectors = user_final_vectors.rename_axis('HADM_ID').reset_index()
    user_final_vectors['HADM_ID'] = df['HADM_ID'].apply(int)
    # write2file(user_final_vectors, self.write_path('labtest_uservectors'))    
    write2file(user_final_vectors, join(singledrug_featurepreprocess_prefix,'lab_matrix'))


# get_demographic_df()
# get_labmatrix()
