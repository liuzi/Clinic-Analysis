from utils._tools import *
from utils._path import*
from utils._save_table2latex import *
# from utils._preprocess_mimic import *
# from S2.utils._tools import *
# from S2.utils._path import*

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from os.path import join
# from pathlib import Path
import os, sys
import glob

from sklearn.cluster import KMeans
from collections import Counter
import math
import urllib.request 
import xmltodict
import textwrap
import re
from scipy.stats import wasserstein_distance
import subprocess


def get_item_args(item):
    args=[
        "CODE_FIELD","INDEX_FILE","INDEX_NAME_FILE","CODE_INDEX",
        "FIG_SUPTITLE","FILENAME_ALL","FILENAME_UNIQUE","FILENAME_STATS",
        "ITEM_NAME"]
    disease_dict={
        args[0]:"ICD9_CODE",
        args[1]:"disease_index_icd9",
        args[2]:"disease_index_icd9_name",
        args[3]:"Disease Index",
        args[4]:"TOP 20 Frequent%s Diseases - Count of Episodes, Cluster: %s",
        args[5]:"COMBINED_%s%s_diseases_%s",
        args[6]:"DISTINCT_%s%s_diseases_%s",
        args[7]:"STATS_%s%s_diseases_%s",
        args[8]:"Disease"

    }
    drug_dict={
        args[0]:"RxNorm Code",
        args[1]:"drug_index_rxnorm",
        args[2]:"drug_index_rxnorm_name",
        args[3]:"Drug Index",
        args[4]:"TOP 20 Frequent%s Drugs - Count of Episodes, Cluster: %s",
        args[5]:"COMBINED_%s%s_drugs_%s",
        args[6]:"DISTINCT_%s%s_drugs_%s",
        args[7]:"STATS_%s%s_drugs_%s",
        args[8]:"Drug"

    }
    args_dict={
        "DISEASE":disease_dict,
        "DRUG":drug_dict
    }
    return args_dict[item]

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
    

    

    def __init__(self,rxnorm_id):
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
        self.rxnorm_id=rxnorm_id
        create_folder(join(singledrug_prefix,rxnorm_id))
        # self.sample_epis_file='HADM_ID_SAMPLE_PER_PATIENT'
        self.feature_folder=join(singledrug_prefix,"FEATURE")
        self.preprocess_folder=join(self.feature_folder,"PRE_PROCESS")
        self.epis_field = "HADM_ID"
        # self.hadm_sampled = None
    
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
    
    def get_sampled_epis(self, treated_epis, size=1):
        """Sample episode for each patient and generate the file of HADM IDs. 
        If the file already exists. skip this method
        
        Notes
        -----
        A patient may enter hospital several times so they have multiple episodes (HADM ID) 
        Thus we randomly select one episode for each patient
        """
        
        if os.path.exists(join(singledrug_prefix,"%s.csv"%"pres_subject_hadmid")):
            subject_hadm = read_data(join(singledrug_prefix,"pres_subject_hadmid"),dtype=str)
            # hadm_sampled = read_data(join(singledrug_prefix,self.sample_epis_file), dtype=str)
        else:  
            subject_hadm = read_data(join(read_prefix,'PRESCRIPTIONS'),
                dtype={"SUBJECT_ID":str,"HADM_ID":str})[['SUBJECT_ID',"HADM_ID"]].drop_duplicates()
            write2file(subject_hadm,join(singledrug_prefix,"pres_subject_hadmid"))
                    ## RUN ONLY FOR THE FIRST TIME! ramdonly select one hospital stay for each patient
        # print(len(treated_epis))
        treated_patient_epis=left_join(treated_epis,subject_hadm,"HADM_ID")
        size = 1        # sample size
        ## ramdonly get a sample hadm_id from each patient's record
        hadm_sampled = self.sampling(treated_patient_epis[['SUBJECT_ID','HADM_ID']].drop_duplicates(),size)['HADM_ID']
#         pres_patient_hadm = pres_ade_df[['SUBJECT_ID','HADM_ID']].drop_duplicates()
#         hadm_sampled = pres_patient_hadm.groupby('SUBJECT_ID', as_index=True).apply(fn)['HADM_ID']
        # self.hadm_sampled = hadm_sampled


        return hadm_sampled


    # # HACK: FEATURE 0  
    # def create_pres_ade_feature(
    #     self,
    #     pres_ade_df,
    #     ade_df
    #     ):
    #     """Second features

    #     Notes
    #     -----
    #     Data Source: 
    #         1) PRESCRIPTION table (pres_df) 
    #         2) SIDER table (ade_df)    

    #     Steps:
    #     ------
    #         1) remove drugs from PRESCRIPTION table that are not in SIDER table

    #     Args:
    #         pres_ade_df ([type]): [description]
    #     """        

    # #     write2file(pres_ade_df,join(res_patient_subgroup_prefix,'PRESCRIPTION_SIDER'))

    #     pres_ade_sampled_df=pres_ade_df[pres_ade_df['HADM_ID'].isin(self.hadm_sampled)]
    #     # pres_ade_sampled_df.head()
    #     ## PATIENT DIAGNOSIS LOG
    #     diaglog_df = read_data(
    #         join(read_prefix,'DIAGNOSES_ICD'),usecols=['SUBJECT_ID','HADM_ID','ICD9_CODE']
    #         ).dropna(subset=['ICD9_CODE']).drop_duplicates()
    #     diaglog_sampled_df=diaglog_df[diaglog_df['HADM_ID'].isin(self.hadm_sampled)]


    #     pres_diag_sampled_df=inner_join(pres_ade_sampled_df,diaglog_sampled_df,['SUBJECT_ID','HADM_ID'])

    #     presdiag_SIDER_df = inner_join(pres_diag_sampled_df,ade_df,['NDC','ICD9_CODE'])
    #     write2file(presdiag_SIDER_df,join(self.preprocess_folder,'PRES_DIAG_SIDER'))
    #     presdiag_SIDER_matrix = self.df2matrix(presdiag_SIDER_df[["SUBJECT_ID","HADM_ID","NDC"]])
    #     write2file(presdiag_SIDER_matrix,join(self.feature_folder,"pres_diag_sider_matrix"))   


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

    def mat_noZero(self,mat,axis_num=0):
        if(axis_num):
            ## 1 for drop row
            mat = mat[(mat!=0).any(axis=1)]
        else:
            ## drop column
            mat = mat.loc[:,(mat!=0).any(axis=0)]
        return mat

    # TODO: 1ST: PUT AFTER condition: if(PREPROCESS dir not extits):run else:read
    def create_fivedata_repre128(self):
        
        create_folder(self.preprocess_folder)
        read_dtype={"SUBJECT_ID":str,"HADM_ID":str}
        ## pre-process four input data respectively
        #NOTE:1) diagnosis: ICD9_CODE, value=1
        diag_df = read_data(join(
            read_prefix,"DIAGNOSES_ICD"),
            dtype=read_dtype).dropna(subset=["ICD9_CODE"])        
        diag_matrix = self.df2matrix(
            diag_df[[*read_dtype]+["ICD9_CODE"]].drop_duplicates())
        write2file(diag_matrix,join(self.preprocess_folder,"diag_matrix"))
    
   
        # NOTE:2) prescription: NDC, value=1
        pres_df = read_data(
            join(read_prefix,'PRESCRIPTIONS'),
            dtype={**read_dtype,**{"NDC":str}})[[*read_dtype]+["NDC"]].dropna(subset=['NDC'])
        pres_df = pres_df[pres_df['NDC']!="0"].drop_duplicates()
        write2file(self.df2matrix(pres_df) ,join(self.preprocess_folder,"pres_matrix"))

        # HACK:
        # NOTE:3) labevents: ITEMID, randomly selected VALUE
        # get_labmatrix()

        # NOTE:4) procedure: ICD9_CODE, value=1
        procedure_df=read_data(
            join(read_prefix,'PROCEDURES_ICD'),
            dtype=read_dtype).dropna(subset=["ICD9_CODE"])[[*read_dtype]+["ICD9_CODE"]].drop_duplicates() 
        write2file(self.df2matrix(procedure_df) ,join(self.preprocess_folder,"procedure_matrix"))


        # NOTE: 5) demographic: []
        # get_demographic_df()


    
    # HACK:Feature 2



    # def create_dissum_feature(self):
        # TODO: 2ND
        ## import original concat dataframe of clamp result
        # section_titles=['HOSPITAL_COURSE', 'MEDICAL_HISTORY', 'MEDICATIONS_ON_ADMISSION']

        # input_files=glob.glob(os.path.join(
        #     clamp_output_prefix,"CONCAT_Results", "%s*")%section_titles[0])
        # print(input_files)  
        # return 0

    
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
        self.create_pres_ade_feature()

        # TODO:
        ## NOTE: feature1
        # self.create_fivedata_repre128()

        # NOTE: feature2
        # self.create_dissum_feature()

    



    
    def runKMeans(self,data,n_clusters):
        km = KMeans(n_clusters=n_clusters).fit(data)
        print(Counter(km.labels_))
        return km.labels_

    def get_ade_rxnorm_df(self):
        if(os.path.exists(join(sideffect_prefix,'rxnorm_icd9_side_effects.csv'))):
            ade_df = read_data(join(sideffect_prefix,'rxnorm_icd9_side_effects'),dtype=str)
        else:
            ade_related_prefix="/data/liu/mimic3/CLAMP_NER/ADE_Related/"
            ndc_cui_map = read_data(join(ade_related_prefix,"ndc_cui_map"),dtype={'CODE':str,'CUI_CODE':str})
            ndc_cui_map = ndc_cui_map.set_index('CODE')['CUI_CODE'].to_dict()
            ade_df = read_data(
                join(sideffect_prefix, 'ndc_icd9_side_effects'), 
                dtype={'NDC':str,'ICD9_CODE':str},usecols=['NDC','ICD9_CODE']).drop_duplicates()
            ade_df['RxNorm']=ade_df['NDC'].apply(lambda x: ndc_cui_map.get(x, None))
            write2file(
                ade_df.dropna(subset=['RxNorm'])[["RxNorm","ICD9_CODE"]],
                join(sideffect_prefix,'rxnorm_icd9_side_effects'))   
        return ade_df

    def compare_sider_newdisease(self,treated_epis,new_icd9_matrix):
        # if(os.path.exists(join(sideffect_prefix,'rxnorm_icd9_side_effects.csv'))):
        #     ade_df = read_data(join(sideffect_prefix,'rxnorm_icd9_side_effects'),dtype=str)
        # else:
        #     ade_related_prefix="/data/liu/mimic3/CLAMP_NER/ADE_Related/"
        #     ndc_cui_map = read_data(join(ade_related_prefix,"ndc_cui_map"),dtype={'CODE':str,'CUI_CODE':str})
        #     ndc_cui_map = ndc_cui_map.set_index('CODE')['CUI_CODE'].to_dict()
        #     ade_df = read_data(
        #         join(sideffect_prefix, 'ndc_icd9_side_effects'), 
        #         dtype={'NDC':str,'ICD9_CODE':str},usecols=['NDC','ICD9_CODE']).drop_duplicates()
        #     ade_df['RxNorm']=ade_df['NDC'].apply(lambda x: ndc_cui_map.get(x, None))
        #     write2file(
        #         ade_df.dropna(subset=['RxNorm'])[["RxNorm","ICD9_CODE"]],
        #         join(sideffect_prefix,'rxnorm_icd9_side_effects'))
        ade_df=self.get_ade_rxnorm_df()
        
        single_drug_ades=ade_df[ade_df['RxNorm']==self.rxnorm_id]['ICD9_CODE']
        
        # new_icd9_matrix =read_data(join(concat_clamp_prefix,"allepis_newICD9_CODE"),dtype={"HADM_ID":str})
        treated_new_icd9=self.mat_noZero(
            inner_join(treated_epis,new_icd9_matrix,"HADM_ID"), axis_num=0
        )
        newdurgs_sider=self.mat_noZero(
            treated_new_icd9.reindex(columns=["HADM_ID"]+list(single_drug_ades)).dropna(axis=1, how='all'),
            axis_num=1)
        print("newdurgs_sider: %s"%str(newdurgs_sider.shape))
        newdrugs_notin_sider=self.mat_noZero(
            treated_new_icd9.reindex(
                columns=list(set(treated_new_icd9.columns)-set(single_drug_ades))
            ),axis_num=1)
        print("newdurgs_notin_sider: %s"%str(newdrugs_notin_sider.shape))
        item_list_prefix=join(singledrug_prefix,self.rxnorm_id)
        write2file(newdurgs_sider,join(item_list_prefix,"newdurgs_sider"))
        write2file(newdrugs_notin_sider,join(item_list_prefix,"newdurgs_not_sider"))

    

    ## same for all models
    def model_plot(self, original_data, model, n_clusters,prescribed_patients,figure_path):

        prescribed_data=inner_join(prescribed_patients[[self.epis_field]].drop_duplicates(),original_data,"HADM_ID")
        onedrug_patients=prescribed_data["HADM_ID"].unique()
        data=prescribed_data.iloc[:,1:]

        plotrows=int(n_clusters/2)

        if(model=="kmeans"):
            labels=self.runKMeans(data,n_clusters)

        label_df = pd.DataFrame({self.epis_field:onedrug_patients,'LABEL':labels})
        
        ade_label_df = left_join(label_df, prescribed_patients, self.epis_field)
        grouped_count = ade_label_df.groupby(['LABEL','ICD9_CODE'])[
            self.epis_field].count().reset_index(name='count').sort_values(['count'], ascending=False) \

            # .reset_index(name='count').sort_values(['count'], ascending=False)
        grouped_count_df = pd.DataFrame(
            grouped_count).pivot(
                index='LABEL', columns='ICD9_CODE', values='count').fillna(0)

        layout=(plotrows,math.ceil(n_clusters/plotrows))
        print(layout)
        grouped_count_df.T.plot(kind='line', subplots=True, sharey=True, layout=layout,figsize=(15,8))
        plt.savefig(join(figure_path,'%s_%d.png'%(model,n_clusters)))
        
    def getandsave_item_name(self):
        item_list_prefix=singledrug_prefix
        all_drug_df=read_data(join(item_list_prefix,"drug_index_rxnorm"),dtype=str)
        all_drug_df['Drug Name']=all_drug_df['RxNorm Code'].apply(get_drugname_byrxcui_api)
        write2file(all_drug_df,join(item_list_prefix,"drug_index_rxnorm_name"))

        all_disease_df=read_data(join(item_list_prefix,"disease_index_icd9"),dtype=str)

        disease_icd9_name_df=read_data(
            join(read_prefix,"D_ICD_DIAGNOSES"),dtype={"ICD9_CODE":str})[["ICD9_CODE","SHORT_TITLE"]].drop_duplicates()
        all_disease_df=left_join(all_disease_df,disease_icd9_name_df,"ICD9_CODE")
        write2file(all_disease_df,join(item_list_prefix,"disease_index_icd9_name"))
        return all_drug_df,all_disease_df,disease_icd9_name_df


    def add_item_name_to_CATEGORY(self,cat_title="STATS"):
        item_list_prefix=singledrug_prefix
        feature_list=["pres_diag_sider_matrix","dissum_autoencoder","five_autoencoder"]
        folder_list=[join(
            singledrug_prefix,self.rxnorm_id,folder) for folder in feature_list]
        
        # if(os.path.exists(join(item_list_prefix,"drug_index_rxnorm_name.csv"))):
        #     all_drug_df=read_data(join(item_list_prefix,"drug_index_rxnorm_name"),dtype=str)
        #     disease_icd9_name_df=read_data(join(item_list_prefix,"disease_index_icd9_name"),dtype=str)
        # else:
        all_drug_df, disease_icd9_name_df, disease_icd9_name_df=self.getandsave_item_name()
        disease_icd9_name_map=disease_icd9_name_df.set_index('ICD9_CODE')['SHORT_TITLE'].to_dict()
        drug_rxnorm_name_map=all_drug_df.set_index('RxNorm Code')['Drug Name'].to_dict()
        if(os.path.exists(join(singledrug_prefix,"rxnorm_drugname.csv"))):
            rxnorm_drugname=read_data(join(singledrug_prefix,"rxnorm_drugname.csv"),dtype=str)
        else:
            rxnorm_drugname=all_drug_df[["RxNorm Code","Drug Name"]]

        for folder in folder_list:
            for file in os.listdir(folder):
                if file.startswith(cat_title):
                    if "drug" in file:
                        file_df=read_data(join(folder,file),dtype=str)
                        if("Drug Name" not in file_df.columns):
                            # TODO: add column of name using apply instead of using join
                            # file_df_withname=left_join(file_df,all_drug_df,["RxNorm Code"])
                            file_df_withname=file_df
                            file_df_withname["Drug Name"]=file_df["RxNorm Code"].apply(
                                lambda x: drug_rxnorm_name_map.get(x,None))
                            # not_found_rxnorm=file_df_withname[file_df_withname["Drug Name"].isnull()]["RxNorm Code"].unique()
                            # not_found_rxnorm=list(set(not_found_rxnorm)-set(rxnorm_drugname["RxNorm Code"]))
                            # not_found_rxnorm_name=list(map(get_drugname_byrxcui_api,not_found_rxnorm))
                            # new_rxnorm_name=pd.DataFrame({"RxNorm Code":not_found_rxnorm,"Drug Name":not_found_rxnorm_name})
                            # rxnorm_drugname=pd.concat([rxnorm_drugname,new_rxnorm_name], axis=0, sort=False)
                            # rxnorm_drugname_map=rxnorm_drugname.set_index('RxNorm Code')['Drug Name'].to_dict()
                            # file_df_withname["Drug Name"]=file_df_withname["RxNorm Code"].apply(
                            #     lambda x: rxnorm_drugname_map.get(x,None)
                            # )
                            write2file(file_df_withname,join(folder,file))
                    else:
                        file_df=read_data(join(folder,file),dtype=str)
                        if("SHORT_TITLE" not in file_df.columns):
                            file_df_withname =left_join(file_df,disease_icd9_name_df,["ICD9_CODE"])
                            file_df_withname["SHORT_TITLE"]=file_df_withname["ICD9_CODE"].apply(
                                lambda x: disease_icd9_name_map.get(x,None))
                            write2file(file_df_withname,join(folder,file))
        



    def run_model_plot(self):
        ##NOTE: 1)DRUGS IN PRES 
        if(not os.path.exists(join(self.preprocess_folder,"pres_rxnorm_matrix.csv"))):
            pres_rxnorm_matrix = self.df2matrix(
                read_data(join(
                    self.preprocess_folder,"pres_rxnorm_df"),dtype=str)[
                        ['SUBJECT_ID','HADM_ID','RxNorm']].drop_duplicates().dropna(subset=["RxNorm"])
            ).fillna(0)
            write2file(pres_rxnorm_matrix,join(self.preprocess_folder,"pres_rxnorm_matrix"))
        else:
            pres_rxnorm_matrix =read_data(join(self.preprocess_folder,"pres_rxnorm_matrix"),dtype={"HADM_ID":str})
        ##ll randomly select one episode of treated patients
        if(not os.path.exists(join(singledrug_prefix,self.rxnorm_id,"treated_epis.csv"))):
            treated_epis = self.mat_noZero(
                pres_rxnorm_matrix[["HADM_ID",self.rxnorm_id]].set_index("HADM_ID"),1).reset_index()['HADM_ID']
            treated_epis=self.get_sampled_epis(treated_epis)
            write2file(pd.DataFrame({"HADM_ID":treated_epis}),join(singledrug_prefix,self.rxnorm_id,"treated_epis"))
        else:
            treated_epis=read_data(join(singledrug_prefix,self.rxnorm_id,"treated_epis"),dtype=str)

        ###NOTE: 2)NEW DRUGS
        new_rxnorm_matrix =read_data(join(concat_clamp_prefix,"allepis_newRxNorm"),dtype={"HADM_ID":str})
       
        ###NOTE: 3)Disease in diagnosie
        diag_icd9_matrix=read_data(join(self.preprocess_folder,"diag_matrix"),dtype={"HADM_ID":str})

        ##NOTE: 4)New Diseases
        new_icd9_matrix =read_data(join(concat_clamp_prefix,"allepis_newICD9_CODE"),dtype={"HADM_ID":str})
        # self.compare_sider_newdisease(treated_epis,new_icd9_matrix)

        # #HACK: Import feature data
        # pres_diag_sider_matrix=read_data(
        #     join(self.feature_folder,"pres_diag_sider_matrix"),dtype=str).fillna(0)
        pres_diag_sider_matrix=inner_join(treated_epis,diag_icd9_matrix,"HADM_ID")
        ades_df=self.get_ade_rxnorm_df()
        ades_list=ades_df[ades_df['RxNorm']==self.rxnorm_id]['ICD9_CODE']
        pres_diag_sider_matrix=pres_diag_sider_matrix.reindex(columns=['HADM_ID']+list(ades_list)).dropna(axis=1, how='all')


        dissum_autoencoder=pd.concat(
            [read_data(join(self.feature_folder,"dissum_Autoencoder_EPIS"),dtype=str),
            read_data(join(self.feature_folder,"dissum_Autoencoder_128"))],axis=1,sort=False)
        five_autoencoder=pd.concat(
            [read_data(join(self.feature_folder,"five_Autoencoder_EPIS"),dtype=str),
            read_data(join(self.feature_folder,"five_Autoencoder_128"))],axis=1,sort=False)
        ##HACK: Import feature data
        feature_list=["pres_diag_sider_matrix","dissum_autoencoder","five_autoencoder"]
        folder_list=[join(
            singledrug_prefix,self.rxnorm_id,folder) for folder in feature_list]
        [create_folder(folder) for folder in folder_list]
        data_list=[pres_diag_sider_matrix, dissum_autoencoder,five_autoencoder]

        ## get features
        # for i in [0]:
        for i in range(len(folder_list)):
        # range(0,len(folder_list)):
            treated_feature=inner_join(treated_epis,data_list[i],"HADM_ID")
            updated_treated_epis=treated_feature["HADM_ID"]
            # NOTE: 1. For TOP 20 FREQUENT DRUGS 2. For TOP 20 FREQUENT NEW DRUGS 3. For TOP 20 FREQUENT DISEASES  4. For TOP 20 FREQUENT NEW DISEASES
            frequent_item_treated = [self.mat_noZero(
                inner_join(updated_treated_epis,frequent_item_matrix,"HADM_ID"), axis_num=0
            ) for frequent_item_matrix in [pres_rxnorm_matrix,new_rxnorm_matrix,diag_icd9_matrix,new_icd9_matrix]]

            # for n_clusters in range(2,6,2):
            for n_clusters in [2]:

                if(os.path.exists(join(folder_list[i],'CLUSTER_label_C%d.csv'%n_clusters))):
                    label_df=read_data(join(folder_list[i],'CLUSTER_label_C%d'%n_clusters),dtype=str)
                    counter_labels=Counter(label_df['LABEL'])
                else:
                    labels=self.runKMeans(treated_feature.iloc[:,1:],n_clusters)
                    counter_labels=Counter(labels)
                    label_df = pd.DataFrame({self.epis_field:updated_treated_epis,'LABEL':labels})
                    write2file(label_df,join(folder_list[i],'CLUSTER_label_C%d'%n_clusters))

                distance_list=[]
                for (item,treated_frequent_df, newitem_flag) in zip(
                    ["DRUG"]*2+["DISEASE"]*2,frequent_item_treated,[False,True,False,True]):
                    treated_frequent_label=inner_join(
                        label_df,treated_frequent_df,"HADM_ID")
                    grouped = treated_frequent_label.groupby("LABEL")
                    distance_list.append(plot_top20items(
                        item,grouped,n_clusters,counter_labels,figure_path=folder_list[i],
                        feature_name=feature_list[i],new_flag=newitem_flag))
                # if()
                cluster_distance_df = pd.DataFrame(
                    distance_list,
                    columns=['N_DISTANCE','M_DISTANCE','Q_DISTANCE'])
                cluster_distance_df.insert(0, 'ITEMS', ['DRUG','NEW_DRUG','DISEASE','NEW_DISEASE'])
                write2file(cluster_distance_df,join(folder_list[i],'CLUSTER_DISTANCE_C%d'%n_clusters))
        
            ## deprecated
            # range(0,len(folder_list)):
                # self.model_plot(data_list[i],"kmeans",n_clusters,prescribed_patients,folder_list[i])

    def debug(self):
        # NOTE: 
        pres_rxnorm_df=read_data(join(self.preprocess_folder,'pres_rxnorm_df'),dtype=str).dropna(subset=['RxNorm'])
        pres_rxnorm_df_group= pres_rxnorm_df.groupby(
            'RxNorm')['HADM_ID'].count().sort_values(ascending=False).to_frame().reset_index()
        pres_rxnorm_df_group.rename(columns={'HADM_ID':'NUM_OF_EPISODES'},inplace=True)

        ade_df=self.get_ade_rxnorm_df()[['RxNorm']].drop_duplicates()
        pres_rxnorm_df_group_sider=inner_join(pres_rxnorm_df_group,ade_df,"RxNorm").head(100)
        pres_rxnorm_df_group_sider.insert(
            1,"Drug Name",list(map(get_drugname_byrxcui_api,pres_rxnorm_df_group_sider['RxNorm'])))
        
        write2file(pres_rxnorm_df_group_sider,join(singledrug_prefix,"TOP100rxnorm_in_presANDsider"))
    

def get_drugname_byrxcui_api(rxcui):

    with urllib.request.urlopen("https://rxnav.nlm.nih.gov/REST/rxcui/%s"%rxcui) as url:

        data = url.read()
        data = xmltodict.parse(data)
        return data['rxnormdata']['idGroup'].get('name',rxcui)


def norm_wasserstein_distance(list_vals):
    list_valsarray=list(map(np.array,list_vals))
    list_valsarray=[[i/array.sum() for i in array] for array in list_valsarray]
    w_distance=wasserstein_distance(*list_valsarray)
    return w_distance


def plot_top20items(item,grouped,n_clusters,counter_labels,figure_path,feature_name,new_flag=False):
    args=get_item_args(item)

    ##import drug list 
    if os.path.exists(join(singledrug_prefix,"%s.csv"%args["INDEX_FILE"])):
        all_item_df=read_data(join(singledrug_prefix,args["INDEX_FILE"]),dtype=str)
        all_item_dict=dict(zip(all_item_df[args['CODE_FIELD']],all_item_df[args["CODE_INDEX"]]))
    else:
        all_item_dict={}

    nrows = int(math.ceil(n_clusters/2.))
    plt.close('all')
    fig, axs = plt.subplots(nrows,2,sharey=True)

    fig.set_size_inches(30, 15*(nrows*0.75))
    fig.subplots_adjust(left=0.2,top=1.6, wspace=0.2,hspace=0.5)

    fig.suptitle(
        args["FIG_SUPTITLE"] % (" New"*new_flag, counter_labels),
        fontweight='bold',fontsize=16)

    ## top 20 frequent serires
    all_serires_df_list=[]
    ## all serires
    combined_serires_df_list=[]

    for (name, groupdf), ax in zip(grouped, axs.flatten()):      
        serires_whole=groupdf.iloc[:,2:].sum(axis=0).sort_values(ascending=False)
        serires=serires_whole.head(20)
        combined_serires_df_list=combined_serires_df_list+[serires_whole[serires_whole>0]]

        ## remove icd9 codes which are already recoded in "all_item_dict" from current series
        group_item_dict=list(set(serires.index).difference(set([*all_item_dict])))
        group_item_dict=dict(zip(
            group_item_dict,
            ["%s_%d"%(args["ITEM_NAME"],id) for id in list(range(len(all_item_dict),len(group_item_dict)+len(all_item_dict)))]))
        all_item_dict={**all_item_dict,**group_item_dict}


        ax.bar(
            [all_item_dict[item_code] for item_code in serires.index],
            # textwrap.fill(get_drugname_byrxcui_api(rxcui)[:35],25)+"..." for rxcui in serires.index], 
            list(serires)
        )

        serires_df= serires.to_frame().reset_index()
        serires_df.columns=[args["CODE_FIELD"],"Count of Episodes"]
        serires_df["Cluster_id"]=name
        serires_df[args["CODE_INDEX"]]=serires_df[args["CODE_FIELD"]].apply(lambda x:all_item_dict[x])
        serires_df["New %s"%(args["ITEM_NAME"])]=new_flag
        all_serires_df_list=all_serires_df_list+[serires_df]
        
        ax.set_title("Cluster Label: %s"%name)
        plt.setp(
            ax.get_xticklabels(), rotation=30, 
            fontsize=10,
            horizontalalignment='right')
        
    plt.setp([a.get_yticklabels() for a in np.reshape(axs,(-1,2))[:, 1]], visible=False)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(join(figure_path,'TOP20%s_%ss_C%d.png'%("_new"*new_flag,args["ITEM_NAME"],n_clusters)))
    
    all_item_df=pd.DataFrame(all_item_dict.items(), columns=[args["CODE_FIELD"],args["CODE_INDEX"]])

    write2file(all_item_df,join(join(singledrug_prefix,args["INDEX_FILE"])))

    all_serires_df=pd.concat(all_serires_df_list,sort=False,ignore_index=True)
    counter_labels_string=re.sub('[^0-9a-zA-Z]+','_',str(counter_labels))
    write2file(all_serires_df,join(figure_path,args["FILENAME_STATS"]%(
        feature_name,"_new"*new_flag,counter_labels_string)))

    # NOTE: find distinct items for each cluster
    common_items=list(set.intersection(
        *[set(serires_whole.index) for serires_whole in combined_serires_df_list]))
    common_items_df=pd.DataFrame({args['CODE_FIELD']:common_items,"Common":True})
    combined_cluster_df_list=[]
    actual_num_clusers=len(combined_serires_df_list)
    for i in range(actual_num_clusers):
        combined_cluster_df_list=combined_cluster_df_list+[
            combined_serires_df_list[i].to_frame(name="Cluster_%d"%i)]
 
    combined_cluster_df=pd.concat(
        combined_cluster_df_list,sort=True,axis=1).fillna(0).reset_index().rename(
            {'index': args['CODE_FIELD']},axis=1)
    combined_cluster_df=left_join(combined_cluster_df,common_items_df, args['CODE_FIELD']).fillna({"Common":False})
    write2file(
        combined_cluster_df,
        join(figure_path,args["FILENAME_ALL"]%(
        feature_name,"_new"*new_flag,counter_labels_string)))
    distinct_cluster_df=combined_cluster_df[combined_cluster_df['Common']==False]
    write2file(
        distinct_cluster_df,
        join(figure_path,args["FILENAME_UNIQUE"]%(
        feature_name,"_new"*new_flag,counter_labels_string)))

    # NOTE: wasserstein DISTANCE: 1) ALL DISTRIBUTION n 2) TOP 20 m  3) TOP 20 unique q
    if(actual_num_clusers>1):
        distances=[]
        unique_serires_df_list=[(
            distinct_cluster_df.loc[:,"Cluster_%d"%cluster_id]).sort_values(ascending=False).head(20) 
            for cluster_id in range(actual_num_clusers)]
        for distribution in [
            combined_serires_df_list,
            [series.head(20) for series in combined_serires_df_list],
            unique_serires_df_list]:
            distances.append(norm_wasserstein_distance(set_union_index(distribution)))
        return distances    
    else:
        return [0,0,0]



def set_union_index(pd_series_list):
    union_index=pd_series_list[0].index
    for series in pd_series_list[1:]:
        union_index=union_index.union(series.index)
    return [series.reindex(union_index,fill_value=0) for series in pd_series_list]

def concat_wasserstein_distance(distance_filename="CLUSTER_DISTANCE_C2"):
    os.chdir(singledrug_prefix)
    dirs = [f for f in glob.glob("*[0-9]*") if (os.path.isdir(f))]
    drug_names = [get_drugname_byrxcui_api(dir_num) for dir_num in  dirs]
    
    # for drug_order in range(5):
    feature_list=["pres_diag_sider_matrix","dissum_autoencoder","five_autoencoder"]
    result_path=join(singledrug_prefix,"CONCAT_RESULTS")
    create_folder_overwrite(join(singledrug_prefix,"CONCAT_RESULTS"))
    for feature_name in feature_list:
        all_distance_df=[]
        for drug_order in range(len(dirs)):
            drug_path=join(singledrug_prefix,dirs[drug_order])
            distance_df=read_data(join(drug_path,feature_name,distance_filename))
            distance_df.insert(0,"Drug Name",drug_names[drug_order])
            distance_df.insert(0,'RxNorm',dirs[drug_order])
            all_distance_df.append(distance_df)

        all_distance_df=pd.concat(all_distance_df, axis=0, sort=False)
        write2file(all_distance_df,join(
            result_path,"CONCAT_%s_%s"%(distance_filename,feature_name)))
        # NOTE: subplot line chart
        plt.close('all')
        fig, axs = plt.subplots(2,2,sharey=True)
        # for item, grouped_df in all_distance_df.groupby("ITEMS"):
        for (item, groupdf), ax in zip(all_distance_df.groupby("ITEMS"), axs.flatten()): 
            groupdf_resetindex=groupdf.set_index("RxNorm") 
            for col in ["N_DISTANCE","M_DISTANCE","Q_DISTANCE"]:
                ax.plot(
                    groupdf_resetindex[col],
                    label=col)
                # ax.legend()
            legend = ax.legend(loc='upper right', shadow=True, fontsize='x-small')
            # legend.get_frame().set_facecolor('C0')
            plt.setp(
                ax.get_xticklabels(), rotation=30, 
                fontsize=6,
                horizontalalignment='right')
            ax.set_title(item)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plt.savefig(join(result_path,"%s_PLOT_%s.png"%(distance_filename, feature_name)))
        # print(item)
        # print(grouped_df)
        # NOTE: subplot line chart


def get_icd9_name_dict():
    icd9_name_file="/data/liu/mimic3/EXTERNEL_DATASET/dexur_ICD9_DISEASENAME/CONCAT_RESULTS/ICD9_NAME_ALL_COMBINE.csv"
    icd9_name_df=read_data(icd9_name_file,dtype=str)
    return icd9_name_df.set_index("ICD9_CODE")["SHORT_TITLE"].to_dict()

# TODO:
def get_ade_rxnorm_df():

    if(os.path.exists(join(sideffect_prefix,'rxnorm_icd9_side_effects.csv'))):
        ade_df = read_data(join(sideffect_prefix,'rxnorm_icd9_side_effects'),dtype=str)
    else:
        ade_related_prefix="/data/liu/mimic3/CLAMP_NER/ADE_Related/"
        ndc_cui_map = read_data(join(ade_related_prefix,"ndc_cui_map"),dtype={'CODE':str,'CUI_CODE':str})
        ndc_cui_map = ndc_cui_map.set_index('CODE')['CUI_CODE'].to_dict()
        ade_df = read_data(
            join(sideffect_prefix, 'ndc_icd9_side_effects'), 
            dtype={'NDC':str,'ICD9_CODE':str},usecols=['NDC','ICD9_CODE']).drop_duplicates()
        ade_df['RxNorm']=ade_df['NDC'].apply(lambda x: ndc_cui_map.get(x, None))
        write2file(
            ade_df.dropna(subset=['RxNorm'])[["RxNorm","ICD9_CODE"]],
            join(sideffect_prefix,'rxnorm_icd9_side_effects'))   
    return ade_df

def concat_top_newdiseases(top_num=20):
    root_path=singledrug_prefix
    all_items_filename="COMBINED*_new_diseases_*.csv"   
    feature_list=['dissum_autoencoder', 'five_autoencoder', 'pres_diag_sider_matrix']
    code_field="ICD9_CODE"
    item_name="DISEASE_NAME"
    cluster_field="Cluster_%s"
    result_path=join(singledrug_prefix,"CONCAT_RESULTS")
    n_clusters=2
    
    ade_rxnorm_df=get_ade_rxnorm_df()

    icd9_name_dict=get_icd9_name_dict()
    drug_folders=list(filter(
        lambda file:re.match("[0-9]{3,}",file),os.listdir(root_path)))
    
    for feature in feature_list:
        # print(os.path.abspath(drug_folders[0]))
        top_df_list=[]
        for drug_rxnorm in drug_folders:
            one_drug_ade_list = ade_rxnorm_df[ade_rxnorm_df["RxNorm"]==drug_rxnorm][code_field]
            # print(one_drug_ade_list)
            sub_path=join(root_path,drug_rxnorm)
            # feature_list=[
            #     feature for feature in os.listdir(sub_path) 
            #     if os.path.isdir(join(sub_path, feature))]
            
            sub_feature_path=join(sub_path,feature)
            combined_file=list(filter(
                lambda file:all([word in file for word in ["COMBINED","new_diseases"]]),
                os.listdir(sub_feature_path)
            ))[0]
            combined_df=read_data(join(sub_feature_path,combined_file))
            distinct_df=combined_df[combined_df["Common"]==False]
            
            # print(combined_df[code_field])
            print(feature)
            print(drug_rxnorm)
            print(set(ade_rxnorm_df).intersection(set(combined_df[code_field])))

            unique_top_table_list=[]
            all_top10_table_list=[]
            for i in range(n_clusters):


                unique_all_top_dfs=list(map(lambda df: df[[code_field,cluster_field%i]] \
                    .set_index(code_field)\
                        .sort_values(by=[cluster_field%i],ascending=False)\
                            .head(top_num).rename(
                                columns={cluster_field%i:"NUM_EPISODES_"+cluster_field%i}
                                ).reset_index(), [distinct_df,combined_df]))
                for df in unique_all_top_dfs:
                    df[item_name]=df[code_field].apply(
                        lambda x: icd9_name_dict.get(x,None))
                    df["IN_SIDER"]=df[code_field].apply(
                        lambda x: x in one_drug_ade_list)
                unique_top_table_list.append(unique_all_top_dfs[0])
                all_top10_table_list.append(unique_all_top_dfs[1])
            top_df_list.append(unique_top_table_list+all_top10_table_list)
        rxnorm_dflists=zip(
            drug_folders,
            list(map(get_drugname_byrxcui_api,drug_folders)),
            top_df_list
            )        
        save_df_as_latextable(
            rxnorm_dflists,join(result_path,"{}_Top{}_NewDisease.tex".format(feature,top_num)))
                # plot_df_as_table(unique_top10_table_list,result_path)


if __name__ == '__main__':
    # rxnorm_id = "197380"
    # print(get_item_args("DISEASE")['INDEX_FILE'])



    # # # NOTE: Get preliminary results
    # for rxnorm_id in ["1658259","866924","966571","885257","836358","855334",
    #     "855290","1808219","1724383","1719291","1807516","213169","1659151"][10:]:
    # # for rxnorm_id in [
    # #     "197380","1807632","1807639","1807634","1658259",
    # #     "866924","1807633","1860466","847630","866508","1361615"][:1]:
    #     fc = feature_creation(rxnorm_id)
    #     fc.run_model_plot()
    #     # fc.add_item_name_to_CATEGORY(cat_title="STATS")
    #     fc.add_item_name_to_CATEGORY(cat_title="DISTINCT")
    #     fc.add_item_name_to_CATEGORY(cat_title="COMBINED")
    # concat_wasserstein_distance()
    # # NOTE: Get preliminary results

    # # TODO: NOTE: supplement disease names

    # concat_top_newdiseases()
    print("a")
    # # NOTE: supplement disease names




# def main():
#     # rxnorm_id = "197380"
#     # fc = feature_creation(rxnorm_id)
#     # fc.run_model_plot()
    
#     if len(sys.argv) != 2:
#         print("Wrong command format, please follwoing the command format below:")
#         print("python single_drug_analysis.py [RxNorm]")
#         exit(0)

#     if len(sys.argv) == 2:    
#         rxnorm_id = sys.argv[1]         
#         fc = feature_creation(rxnorm_id)
#         fc.run_model_plot()

        






        
        
        