from utils._tools import *
from utils._path import*
import os, sys
import pandas as pd
import numpy as np
from os.path import join
from pathlib import Path


class feature_creation():
    """ Generate features from different data sources using 
    
    Parameters
    ----------
    
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    Attributs
    ---------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.
    
    See also
    --------
    
    
    Notes
    -----
    
    
    Examples
    --------

    
    """
    

    

    def __init__(self):
        """
        Parameters
        ----------
        drug_id : string, default="197380" 
            RxNorm code (e.g. Atenolol, RxNorm: 197380, NDC:51079075920) for drug. 
            Determine the study object. All analysis are based on
            the selected drug/drugs.
        """



        # self.drugid = drugid
        # self.outputpath = singledrug_prefix%drugid
        # print(self.outputpath)
        ## folders for features
        self.sample_epis_file='HADM_ID_SAMPLE_PER_PATIENT'
        self.feature_folder=join(singledrug_prefix,"FEATURE")
        create_folder(self.feature_folder)
        self.hadm_sampled = None
    
    def sampling(self, df, size, sample_group="SUBJECT_ID", sample_unit="HADM_ID"):
        """Get randomly selected samples from each subgroup across all 

        Args:
            df (DataFrame): patient-episode record
            size (int): numbers of samples wanna be selected from each subgroups
            sample_group (str, optional): column name of subgroup. Defaults to "SUBJECT_ID".
            sample_unit (str, optional): column name of sampled subject. Defaults to "HADM_ID".

        Returns:
            [type]: 
        """

        fn = lambda obj: obj.loc[np.random.choice(obj.index, size),:]
        return df.groupby(sample_group, as_index=True).apply(fn)
    
    def get_sampled_epis(self, pres_ade_df, size=1):
        """Sample episode for each patient and generate the file of HADM IDs. 
        If the file already exists. skip this method
        
        Notes
        -----
        A patient may enter hospital several times so they have multiple episodes (HADM ID) 
        Thus we randomly select one episode for each patient
        """
        
        if Path(join(singledrug_prefix,"%s.csv"%self.sample_epis_file)).exists():
            hadm_sampled = read_data(join(singledrug_prefix,self.sample_epis_file))['HADM_ID']
        else:            
            ## RUN ONLY FOR THE FIRST TIME! ramdonly select one hospital stay for each patient
            size = 1        # sample size
            ## ramdonly get a sample hadm_id from each patient's record
            hadm_sampled = self.sampling(pres_ade_df[['SUBJECT_ID','HADM_ID']].drop_duplicates(),size)
    #         pres_patient_hadm = pres_ade_df[['SUBJECT_ID','HADM_ID']].drop_duplicates()
    #         hadm_sampled = pres_patient_hadm.groupby('SUBJECT_ID', as_index=True).apply(fn)['HADM_ID']
            write2file(pd.DataFrame(hadm_sampled),join(singledrug_prefix,self.sample_epis_file))
        self.hadm_sampled = hadm_sampled


      
    def create_pres_ade_feature(self,pres_ade_df,ade_df):
        """Second features

        Notes
        -----
        Data Source: 
            1) PRESCRIPTION table (pres_df) 
            2) SIDER table (ade_df)    

        Steps:
        ------
            1) remove drugs from PRESCRIPTION table that are not in SIDER table

        Args:
            pres_ade_df ([type]): [description]
        """        

    #     write2file(pres_ade_df,join(res_patient_subgroup_prefix,'PRESCRIPTION_SIDER'))

        pres_ade_sampled_df=pres_ade_df[pres_ade_df['HADM_ID'].isin(self.hadm_sampled)]
        # pres_ade_sampled_df.head()
        ## PATIENT DIAGNOSIS LOG
        diaglog_df = read_data(
            join(read_prefix,'DIAGNOSES_ICD'),usecols=['SUBJECT_ID','HADM_ID','ICD9_CODE']
            ).dropna(subset=['ICD9_CODE']).drop_duplicates()
        diaglog_sampled_df=diaglog_df[diaglog_df['HADM_ID'].isin(self.hadm_sampled)]


        pres_diag_sampled_df=inner_join(pres_ade_sampled_df,diaglog_sampled_df,['SUBJECT_ID','HADM_ID'])
        # write2file(pres_diag_sampled_df,join(res_patient_subgroup_prefix,'PRES_DIAG_SAMPLED'))
        presdiag_SIDER_df = inner_join(pres_diag_sampled_df,ade_df,['NDC','ICD9_CODE'])
        write2file(presdiag_SIDER_df,join(self.feature_folder,'PRES_DIAG_SIDER'))
        
    
    def create_data(self):
        """
        Notes:
        1) First Feature:

        2) Second Feature:
            Drugs: PRESCRIPTIONS.csvm only remain rows with drugs that can be found in SIDER
        3) Third Feature:
        """

        ## PATIENT PRESCRIPTION LOG
        pres_df=read_data(join(
            read_prefix,'PRESCRIPTIONS'),dtype={'NDC':str}).dropna(subset=['NDC'])

        ## DRUG-ADE IN SIDER4, !!SIDER HAVE DUPLICATED RECORDS
        ade_df = read_data(
            join(sideffect_prefix, 'ndc_icd9_side_effects'), 
            dtype={'NDC':str,'ICD9_CODE':str},usecols=['NDC','ICD9_CODE']).drop_duplicates()

        ## GET LIST OF DRUGS FROM SIDER4
        ade_drug=ade_df['NDC'].drop_duplicates()
        # NOTE:
        ## Remove records from Prescriptions where drugs cannot be found in SIDER
        pres_ade_df = pres_df[pres_df['NDC'].isin(ade_drug)].drop_duplicates()

        ## Sample only one episode for each patient
        self.get_sampled_epis(pres_ade_df)
        self.create_pres_ade_feature(pres_ade_df, ade_df)

        
        
        

def test():
    fc = feature_creation()
    fc.create_data()


test()
# print("test")

        
        
        