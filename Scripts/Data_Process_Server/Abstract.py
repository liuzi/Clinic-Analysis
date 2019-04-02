import pandas as pd
import numpy as np


class Abstract(object):
    def __init__(self, read_prefix = '/Users/lynnjiang/liuGit/data/%s.csv',\
                 write_prefix = '/Users/lynnjiang/liuGit/data/temp/%s.csv'):
        self.read_prefix = read_prefix
        self.write_prefix = write_prefix

    def read_data(self, filename, header = 'infer'):
        return pd.read_csv(self.read_prefix % filename, sep=',', header = header, encoding='latin1')

    def write2file(self,temp_result,filename):
        temp_result.to_csv(self.write_prefix % filename,index=False)

    def left_join(self, left_df, right_df, joined_field):
        return pd.merge(left_df, right_df, how='left', on=[joined_field])

    def inner_join(self, left_df, right_df, joined_field):
        return pd.merge(left_df, right_df, how='inner', on=[joined_field])

    def get_top_items(self,dataset,topN,item_name,filename):
        '''
        Only run for one tiem
        :param dataset:
        :param topN:
        :param item_name:
        :param filename:
        :return:
        '''
        frequent_items = dataset.groupby(item_name).size().reset_index(name='counts').\
                             sort_values(ascending=0, by='counts')[:topN][[item_name]]
        self.write2file(frequent_items,filename)

    def create_selected_users(self, output_name, user_file='PATIENTS', sample_rate=0.05):
        userid = 'SUBJECT_ID'
        whole_paitients = self.read_data(user_file)[[userid]]
        selected_patients = whole_paitients.sample(frac=sample_rate)
        self.write2file(selected_patients, 'sample_patients/%s' % output_name)
        return selected_patients



# tt = Abstract()
# tt.get_top_items(tt.read_data('PRESCRIPTIONS'),200,'DRUG','PPRES_TOP200_DRUGS')


'''
Features from LABEVENTS
'''




