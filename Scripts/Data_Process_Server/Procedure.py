import pandas as pd
import numpy as np
from Abstract import Abstract
from tools import *

class Procedures(Abstract):
    def __init__(self, read_prefix, write_prefix):
        super(Procedures, self).__init__(read_prefix,write_prefix)

    def get_binary_features(self):
        procedures_icd_df = read_data(self.read_path('PROCEDURES_ICD')).dropna(subset=['ICD9_CODE'])

        ## SEQ_NUM: provides the order in which the procedures were performed.
        procedures_icd_df['VALUE'] = 1
        procedures_icd_df.sort_values(by='SUBJECT_ID')
        procedures_vec = pd.pivot_table(procedures_icd_df, index=['SUBJECT_ID'], columns=['ICD9_CODE'], \
                                        values=['VALUE']).fillna(0)
        procedures_vec = procedures_vec.reset_index()
        procedures_vec.columns = procedures_vec.columns.droplevel(0)
        procedures_vec=procedures_vec.rename(columns = {procedures_vecw.columns[0]:'SUBJECT_ID'})
        write2file(procedures_vec, self.write_path('procedures_uservectors'))
        return procedures_vec

# pp = Procedures()
# pp.get_binary_features()