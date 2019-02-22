from Data_Cleaning import Abstract
import pandas as pd
import numpy as np


class Diagnosis(Abstract.Abstract):
    def __init__(self):
        super().__init__()

    # Pick data of specific diseases from diagnoses data
    # 4280 Congestive heart failure
    def get_labels(self, label_filed='ICD9_CODE', user_field='SUBJECT_ID'):

        icd9_code_df = self.read_data('D_ICD_DIAGNOSES')
        diagnosis_df = self.read_data('DIAGNOSES_ICD')
        label_list = ['4280']
        link_field = 'ICD9_CODE'
        columns = ['SUBJECT_ID', 'ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']

        diagnosis_desc_df = self.left_join(diagnosis_df,icd9_code_df,link_field)


        ## TODO: What if it is required to process multiple diseases?
        ## Drop multiple same diagnoses for each patient
        filtered_diseass = diagnosis_desc_df [diagnosis_desc_df ['ICD9_CODE'].isin(label_list)]
        filtered_diseass['LABEL'] = 1
        return filtered_diseass[[user_field, 'LABEL']].drop_duplicates()

    def get_uservectors(self):

        ## Reomove rows where ICD9_CODE is null
        diagnosis_df = self.read_data('DIAGNOSES_ICD')
        diagnosis_df=diagnosis_df.dropna(subset=['ICD9_CODE'])

        ## Create binary value for diagnoses
        diagnosis_df['VALUE'] = 1
        user_diag_vec = pd.pivot_table(diagnosis_df, index=['SUBJECT_ID'], columns=['ICD9_CODE'],\
                                       values=['VALUE']).fillna(0)
        user_diag_vec = user_diag_vec.reset_index()
        self.write2file(user_diag_vec,'USER_VECTORS/diagnoses_uservectors')


dd = Diagnosis()
# dd.write2file(dd.get_labels(),'diagnosis_label_4280')
dd.get_uservectors()
# selected_user_list = dd.read_data('temp/PATIENTS_5_PER')
# dd.write2file(dd.left_join(selected_user_list,all_diagnoses,'SUBJECT_ID'),'selected_diagnoses')