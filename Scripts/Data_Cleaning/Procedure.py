from Data_Cleaning import Abstract
import pandas as pd
import numpy as np


class Procedures(Abstract.Abstract):
    def __init__(self):
        super().__init__()

    def get_binary_features(self):
        procedures_icd_df = self.read_data('PROCEDURES_ICD').dropna(subset=['ICD9_CODE'])

        ## SEQ_NUM: provides the order in which the procedures were performed.
        procedures_icd_df['VALUE'] = 1
        procedures_icd_df.sort_values(by='SUBJECT_ID')
        procedures_vec = pd.pivot_table(procedures_icd_df, index=['SUBJECT_ID'], columns=['ICD9_CODE'], \
                                        values=['VALUE']).fillna(0)
        procedures_vec = procedures_vec.reset_index()
        # procedures_vec.columns
        procedures_vec.columns = procedures_vec.columns.droplevel(0)
        self.write2file(procedures_vec, 'USER_VECTORS/procedures_uservectors')


pp = Procedures()
pp.get_binary_features()