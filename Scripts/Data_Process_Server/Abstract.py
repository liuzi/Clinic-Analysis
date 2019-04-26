import pandas as pd
import numpy as np
from tools import *
import os

class Abstract(object):
    def __init__(self, read_prefix = '/Users/lynnjiang/liuGit/data/',\
                 write_prefix = '/Users/lynnjiang/liuGit/data/temp/'):
        self.read_prefix = read_prefix
        self.write_prefix = write_prefix
        
    def read_path(self, file_name):
        return os.path.join(self.read_prefix,file_name)
    
    def write_path(self, file_name):
        return os.path.join(self.write_prefix,file_name)
    
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
        write2file(frequent_items,self.write_path(filename))

    def create_selected_users(self, output_name, user_file='PATIENTS', sample_rate=0.05):
        userid = 'SUBJECT_ID'
        whole_paitients = self.read_data(user_file)[[userid]]
        selected_patients = whole_paitients.sample(frac=sample_rate)
        write2file(selected_patients, self.write_path('sample_patients/%s' % output_name))
        return selected_patients



# tt = Abstract()
# tt.get_top_items(tt.read_data('PRESCRIPTIONS'),200,'DRUG','PPRES_TOP200_DRUGS')


'''
Features from LABEVENTS
'''




