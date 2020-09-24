from utils._tools import *
from utils._path import*
# from S2.utils._tools import *
# from S2.utils._path import*
import os, sys
import pandas as pd
import numpy as np
from os.path import join
from pathlib import Path
import glob


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
        self.preprocess_folder=join(self.feature_folder,"PRE_PROCESS")
        self.hadm_sampled = None
    
    # HACK: FEATURE0
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
            hadm_sampled = read_data(join(singledrug_prefix,self.sample_epis_file), dtype=str)
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

        presdiag_SIDER_df = inner_join(pres_diag_sampled_df,ade_df,['NDC','ICD9_CODE'])
        write2file(presdiag_SIDER_df,join(self.feature_folder,'PRES_DIAG_SIDER'))


    # def print_epis_stats(self, df):  
    #     print("# of rows: %d"%len(df))
    #     print("# of patients: %d"%len(df['SUBJECT_ID'].unique()))
    #     print("# of patients: %d"%len(df['HADM_ID'].unique()))    


    # HACK: FEATURE 1
    def df2matrix(self,df):
        cols=df.columns
        print("Original Data:")
        print_patient_stats(df)
        ## HACK: sampling should be after creating matrix
        # df_sampled = left_join(self.hadm_sampled,df,list(self.hadm_sampled.columns))
        # print("Sampled Data:")
        # print_patient_stats(df_sampled)

        if("VALUE" not in cols):
            df["VALUE"]=1
        
        df_matrix = df.pivot_table(
            index=cols[1], columns=cols[2], values="VALUE",fill_value=0).reset_index()

        print("Number of %s: %d"%(cols[2],len(df_matrix.columns)-1))
        return df_matrix

    # TODO: 1ST: PUT AFTER condition: if(PREPROCESS dir not extits):run else:read
    def create_fivedata_repre128(self):
        
        create_folder(self.preprocess_folder)
        read_dtype={"SUBJECT_ID":str,"HADM_ID":str}
        ## pre-process four input data respectively
        #NOTE:1) diagnosis: ICD9_CODE, value=1
        # diag_df = read_data(join(
        #     read_prefix,"DIAGNOSES_ICD"),
        #     dtype=read_dtype).dropna(subset=["ICD9_CODE"])        
        # diag_matrix = self.df2matrix(
        #     diag_df[[*read_dtype]+["ICD9_CODE"]].drop_duplicates())
        # write2file(diag_matrix,join(self.preprocess_folder,"diag_matrix"))
    
   
        # NOTE:2) prescription: NDC, value=1
        # pres_df = read_data(
        #     join(read_prefix,'PRESCRIPTIONS'),
        #     dtype={**read_dtype,**{"NDC":str}})[[*read_dtype]+["NDC"]].dropna(subset=['NDC'])
        # pres_df = pres_df[pres_df['NDC']!="0"].drop_duplicates()
        # write2file(self.df2matrix(pres_df) ,join(self.preprocess_folder,"pres_matrix"))

        # HACK:
        # NOTE:3) labevents: ITEMID, randomly selected VALUE

        # NOTE:4) procedure: ICD9_CODE, value=1
        procedure_df=read_data(
            join(read_prefix,'PROCEDURES_ICD'),
            dtype=read_dtype).dropna(subset=["ICD9_CODE"])[[*read_dtype]+["ICD9_CODE"]].drop_duplicates() 
        write2file(self.df2matrix(procedure_df) ,join(self.preprocess_folder,"procedure_matrix"))


        # NOTE: 5) demographic: []
        
        # return diag_matrix

    
    # HACK:Feature 2



    def create_dissum_feature(self):
        # TODO: 2ND
        ## import original concat dataframe of clamp result
        section_titles=['HOSPITAL_COURSE', 'MEDICAL_HISTORY', 'MEDICATIONS_ON_ADMISSION']

        input_files=glob.glob(os.path.join(
            clamp_output_prefix,"CONCAT_Results", "%s*")%section_titles[0])
        print(input_files)  
        return 0

    
    # HACK: ALL FEATURES CREATION
    def create_data(self):
        """
        Notes:
        1) First Feature:

        2) Second Feature:
            Drugs: PRESCRIPTIONS.csvm only remain rows with drugs that can be found in SIDER
        3) Third Feature:
        """

        create_folder(self.feature_folder)

        # NOTE: feature0
        ## PATIENT PRESCRIPTION LOG
        # pres_df=read_data(join(
        #     read_prefix,'PRESCRIPTIONS'),dtype={'NDC':str}).dropna(subset=['NDC'])
        # ## DRUG-ADE IN SIDER4, !!SIDER HAVE DUPLICATED RECORDS
        # ade_df = read_data(
        #     join(sideffect_prefix, 'ndc_icd9_side_effects'), 
        #     dtype={'NDC':str,'ICD9_CODE':str},usecols=['NDC','ICD9_CODE']).drop_duplicates()

        # ## GET LIST OF DRUGS FROM SIDER4
        # ade_drug=ade_df['NDC'].drop_duplicates()
        # # NOTE:
        # ## Remove records from Prescriptions where drugs cannot be found in SIDER
        # pres_ade_df = pres_df[pres_df['NDC'].isin(ade_drug)].drop_duplicates()
        # ## Sample only one episode for each patient
        # self.get_sampled_epis(pres_ade_df)
        # self.create_pres_ade_feature(pres_ade_df, ade_df)

        # TODO:
        ## NOTE: feature1
        # self.create_fivedata_repre128()

        # NOTE: feature2
        self.create_dissum_feature()



        
        
        

def test():
    fc = feature_creation()
    fc.create_data()
    # fc.create_fivedata_repre128()



test()
# print("test")
# import tensorflow as tf
# # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)

# # Create some tensors
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)

# print(c)

        
        
        