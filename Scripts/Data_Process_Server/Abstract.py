import pandas as pd
import numpy as np


class Abstract(object):
    def __init__(self, read_prefix = '/Users/lynnjiang/liuGit/data/',\
                 write_prefix = '/Users/lynnjiang/liuGit/data/temp/'):
        self.read_prefix = self.regularize_path(read_prefix)
        self.write_prefix = self.regularize_path(write_prefix)

    def regularize_path(self, path):
        if(path[-1]=='/'): return path+'%s%s'
        else: return path + '/%s%s'
    
    def read_data(self, filename, sep=',', header = 'infer', suffix = '.csv', pre=''):
        if(pre == ''):
            path = self.read_prefix
        else:
            path = self.regularize_path(pre)
        path = path % (filename, suffix)
        return pd.read_csv(path, sep=sep, header = header, encoding='latin1')

    def write2file(self,temp_result,filename,suffix = '.csv'):
        temp_result.to_csv(self.write_prefix % (filename, suffix),index=False)

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




