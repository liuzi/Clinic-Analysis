from Data_Cleaning import Abstract
import re
import pandas as pd
import numpy as np


class Prescriptions(Abstract.Abstract):
    def __init__(self):
        super().__init__()

    def dosage_value(self, s):
        try:
            return float(s)
        except ValueError:
            return 0.0

    def read_prescriptions_data(self, prescriptions_file='PRESCRIPTIONS'):
        columns = ['SUBJECT_ID','DRUG','DOSE_VAL_RX','ENDDATE','STARTDATE']
        digit = re.compile('[^\d-]')

        prescriptions_df = self.read_data(prescriptions_file)
        pres_clean_df = prescriptions_df[columns].dropna(subset=['DOSE_VAL_RX'])

        ## Regularize value in ['DOSE_VAL_RX'] to numeric type, e.g. (300-600)->450
        pres_clean_df.DOSE_VAL_RX = pres_clean_df.DOSE_VAL_RX.apply(lambda x: np.average(list(map(self.dosage_value,digit.sub('',str(x)).split('-')))))

        return pres_clean_df

    def get_statistics_vec(self, pres_df):

        ## Calculate statistics for each prescription for each patient
        stc_ls = ['min', 'max', 'mean', 'count']
        pres_agg_df = pres_df.groupby(['SUBJECT_ID', 'DRUG'])['DOSE_VAL_RX'].agg(stc_ls)
        pres_agg_df.head()

        ## Reshape dataframe to user-drug-statistics matrix
        pres_agg_df = pres_agg_df.reset_index()
        pres_vec = pres_agg_df.pivot(index='SUBJECT_ID', columns='DRUG')
        pres_vec=pres_vec.reset_index()
        pres_vec.columns = pres_vec.columns.droplevel(0)
        ## Imputation of null value: 0
        pres_vec = pres_vec.fillna(0)

        self.write2file(pres_vec,'USER_VECTORS/prescriptions_uservectors')


        return pres_vec

    def get_average_dosage(self, pres_df):
        ## Get period for patient taking a medicine in each record (ENDDATE-STARTDATE+1) (/day)
        enddate, startdate, period, ave_value = 'ENDDATE', 'STARTDATE', 'PERIOD', 'AVG_DOSE_VAL_RX'
        pres_df[enddate] = pd.to_datetime(pres_df[enddate])
        pres_df[startdate] = pd.to_datetime(pres_df[startdate])
        pres_df[period] = (pres_df[enddate] - pres_df[startdate]).dt.days + 1

        ## Get average dosage for each patient taking each kind of drug
        pres_sum_df = pres_df.groupby(['SUBJECT_ID','DRUG'])['DOSE_VAL_RX','PERIOD'].sum().reset_index()
        pres_sum_df['AVG_DOSE_VAL_RX'] = round(pres_sum_df['DOSE_VAL_RX']/pres_sum_df['PERIOD'],6)

        ## Transform dataframe to matrix (USER*DRUG)
        user_prescription_vector = pd.pivot_table(pres_sum_df, index=['SUBJECT_ID'], columns=['DRUG'],\
                                                  values=['AVG_DOSE_VAL_RX'])
        ## Imputation: with 0
        user_final_presvec= user_prescription_vector.fillna(0)
        ## TODO: reset_index has some problems, the title is contained as a row
        user_final_presvec = user_final_presvec.reset_index()
        user_final_presvec.columns = user_final_presvec.columns.droplevel(-1)

        return user_final_presvec


pp = Prescriptions()
pp.get_statistics_vec(pp.read_prescriptions_data())
# user_list = pp.read_data('temp/PATIENTS_5_PER')['SUBJECT_ID']
# user_vectors = pp.to_attributes(pp.read_prescriptions_data(user_list))
# pp.write2file(user_vectors,'prestest_uservectors')



