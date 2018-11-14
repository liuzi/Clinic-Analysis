import pandas as pd
import numpy as np


class Data_Preprocessing():

    def __init__(self, prefix='../../data/'):
        self.prefix = prefix
        self.diagnoses = None
        self.measures = None

    # Descriptions of different kinds of diseases based on ICD_CODE
    def get_icd9codes(self, filename='D_ICD_DIAGNOSES.csv'):
        icd9_code_df = pd.read_csv(self.prefix + filename, sep=',', encoding='latin1')
        return icd9_code_df

    # Information about what diseases patients are diagnosed
    def read_diagnoses(self, filename='DIAGNOSES_ICD.csv',
                       link_filename='D_ICD_DIAGNOSES.csv', link_field='ICD9_CODE',
                       columns=['SUBJECT_ID', 'ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']):
        diagnose_df = pd.read_csv(self.prefix + filename, sep=',', encoding='latin1')
        diagnose_df = pd.merge(diagnose_df, self.get_icd9codes(link_filename), how='left', on=[link_field])
        self.diagnoses = diagnose_df[columns]

    # Descriptions of different kinds of clinical measurements
    def get_measure_items(self, filename='D_LABITEMS.csv'):
        measure_items_df = pd.read_csv(self.prefix + filename, sep=',', encoding='latin1')
        return measure_items_df

    # Information about what measurements patients have and values of those measurements -> attributes
    def read_measurements(self, filename='LABEVENTS.csv', columns=['SUBJECT_ID', 'VALUE', 'VALUENUM', 'LABEL']):
        measurements_df = pd.read_csv(self.prefix + filename, sep=',', encoding='latin1')
        measurements_df = pd.merge(measurements_df, self.get_measure_items(), how='left', on=['ITEMID'])[columns]
        self.measures = measurements_df

    # Pick data of specific diseases from diagnoses data
    # 4280 Congestive heart failure
    def get_labels(self, label_filed='ICD9_CODE', labels=['4280'], user_field='SUBJECT_ID'):
        if self.diagnoses is None:
            print('Please use read_diagnoses function to get diagnose data first')
            return False
        else:
            ## TODO: What if it is required to process multiple diseases?
            filtered_diseass = self.diagnoses[self.diagnoses[label_filed].isin(labels)].drop_duplicates()
            filtered_diseass['LABEL'] = 1
            return filtered_diseass[[user_field, 'LABEL']]

    # min_null: fields with none values more than 80% are categorical fields, otherwise continous fields.
    def get_user_vectors(self, fields=['VALUE', 'LABEL', 'VALUENUM', 'SUBJECT_ID'], min_nulls=0.8, max_labels=4):
        value = fields[0]
        label = fields[1]
        valuenum = fields[2]
        userid = fields[3]
        labels_nulls = self.measures.dropna(subset=[value]).groupby(label)[valuenum].apply(lambda x: x.isnull().sum())

        # initiate all user vectors
        len_all_measures = len(self.measures)
        stc_ls = ['min', 'mean', 'max', 'std']
        stc_ls_len = len(stc_ls)
        user_vectors = {}
        all_users = set(self.measures[userid].unique())
        for user in all_users:
            user_vectors[user] = np.empty([0])

        # Groupby label
        measure_label = self.measures.groupby(label)
        i = 1
        for label, label_df in measure_label:
            if i > max_labels: break
            # continuous
            if ((labels_nulls[label] / len_all_measures) < min_nulls):
                # pick one iteration of continuous label
                label_df = label_df.dropna(subset=[valuenum])
                #                 print(label_df[:10])
                missed_users = all_users - set(label_df[userid].unique())
                label_df_agg = label_df.groupby(userid)[valuenum].agg(stc_ls)

                for user, row in label_df_agg.iterrows():
                    if pd.isnull(row['std']): row['std'] = 0
                    user_vectors[user] = np.append(user_vectors[user], [row[x] for x in stc_ls])
                for user in missed_users:
                    user_vectors[user] = np.append(user_vectors[user], np.full(stc_ls_len, np.nan))

            else:
                # pick one iteration of discrete label
                label_df = label_df.dropna(subset=[value])
                missed_users = all_users - set(label_df[user].unique())
                label_df_agg = label_df.groupby([userid, value])[label].agg(['count'])
                label_df_agg_per = label_df_agg.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))

                label_df_agg_per.reset_index
                label_matrix = pd.pivot_table(label_df_agg_per,
                                              values=['count'], index=[userid], columns=[value]).fillna(0)

                for user in label_matrix.index.get_level_values(userid):
                    user_vectors[user] = np.append(user_vectors[user], label_matrix.ix[user].values)
                for user in missed_users:
                    user_vectors[user] = np.append(user_vectors[user], np.full(label_matrix.shape[1], np.nan))
            i += 1

        return user_vectors
