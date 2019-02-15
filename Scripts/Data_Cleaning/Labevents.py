from Data_Cleaning import Abstract
import pandas as pd
import numpy as np


class Labevents(Abstract.Abstract):
    def __init__(self):
        super().__init__()

    def read_measurements_data(self, user_list, items_file='D_LABITEMS', measures_file='LABEVENTS'):
        '''
        Get measurements data with top N frequent labels.
        :param items_file:
        :param measures_file:
        :return:
        '''
        left_columns = ['SUBJECT_ID', 'ITEMID', 'VALUE', 'VALUENUM']
        right_columns = ['ITEMID', 'LABEL']
        subject_id, label, value = 'SUBJECT_ID', 'LABEL', 'VALUE'
        link_field = 'ITEMID'
        top200_labels = 'temp/LAB_TOP200_LABELS'
        num_labels = 20

        # most_frequent_items = self.read_data()
        measurements_df = self.read_data(measures_file)[left_columns]

        '''
        Select records in labevents with frequent label instead of itemid
        '''
        # measure_items_df = self.read_data(items_file)[right_columns]
        # frequent_labels = self.read_data(top200_labels)[:num_labels][label]
        # joined_meas_df = self.left_join(measurements_df, measure_items_df, link_field).dropna(subset=[value])
        # measure_filter_df = joined_meas_df[(joined_meas_df[label].isin(frequent_labels)) \
        #                                         & (joined_meas_df[subject_id].isin(user_list))]

        measure_filter_df = measurements_df.dropna(subset=[value])
        return measure_filter_df


    def get_uservectors(self, user_list, raw_measurements, min_nulls=0.8):
        '''
        min_null: fields with none values more than 80% are categorical fields, otherwise are continuous fields.
        :param user_list:
        :param raw_measurements:
        :param min_nulls:
        :return:
        '''

        value ='VALUE'
        item_id = 'ITEMID'
        ## For LABEL instead of ITEMID
        # label_id = 'LABEL'
        valuenum = 'VALUENUM'
        userid = 'SUBJECT_ID'
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

        return user_final_vectors.rename_axis('SUBJECT_ID').reset_index()


ll = Labevents()
## Just run for one time: get seleted users and top items
# create_selected_users(ll)
selected_user_list = ll.read_data('PATIENTS')['SUBJECT_ID']
# ll.get_top_items(linkeddata,200)
user_vectors = ll.get_uservectors(selected_user_list,ll.read_measurements_data(selected_user_list))

ll.write2file(user_vectors,'USER_VECTORS/labtest_uservectors')
# ll.write2file(user_vectors.dropna(axis=0,how='all'),'labtest_uservectors_notna')
# ll.write2file(user_vectors.dropna(axis=1,how='all'),'labtest_uservectors_check_column')