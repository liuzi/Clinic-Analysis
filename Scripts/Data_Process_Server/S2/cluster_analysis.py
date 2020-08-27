from utils.tools import *
import os, sys
import pandas as pd
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
    
    >>> from sklearn.cluster import KMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    >>> kmeans.labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> kmeans.predict([[0, 0], [12, 3]])
    array([1, 0], dtype=int32)
    >>> kmeans.cluster_centers_s
    array([[10.,  2.],
           [ 1.,  2.]])
    
    """
    
    """ Static paths """
    read_prefix = "/data/MIMIC3/"
    res_patient_subgroup_prefix = "/data/liu/adverse_events/patient_subgroup"
    ## DRUG-ADE IN SIDER4, !!SIDER HAVE DUPLICATED RECORDS
    sideffect_prefix = '/data/liu/adverse_events'
    output_prefix="/data/liu/mimic3/CLAMP_NER/single_drug_analysis/%s/cluster"
    
    @_deprecate_positional_args
    def __init__(self,drugid="197380"):
        """
        Parameters
        ----------
        drug_id : string, default="197380" 
            RxNorm code (e.g. Atenolol, RxNorm: 197380, NDC:51079075920) for drug. 
            Determine the study object. All analysis are based on
            the selected drug/drugs.
        """
        self.drugid = drugid
        self.outputpath = output_prefix%drugid
        create_folder(self.outputpath)
        self.hadm_sampled = None
    
    def sampling(self, df, size, sample_group="SUBJECT_ID", sample_unit="HADM_ID"):
        fn = lambda obj: obj.loc[np.random.choice(obj.index, size),:]
        return df.groupby(sample_group, as_index=True).apply(fn)[sample_unit]
    
    def sampled_epis(self, pres_ade_df, size=1):
        """Sample episode for each patient and generate the file of HADM IDs. 
        If the file already exists. skip this method
        
        Notes
        -----
        A patient may enter hospital several times so they have multiple episodes (HADM ID) 
        Thus we randomly keep one episode for each patients
        """
        
        if Path(join(self.outputpath,'HADM_ID_SAMPLE_PER_PATIENT')).exists():
            hadm_sampled = read_data(join(res_patient_subgroup_prefix,'HADM_ID_SAMPLEPER_PATIENT'))['HADM_ID']
        else:            
            ## RUN ONLY FOR THE FIRST TIME! ramdonly select one hospital stay for each patient
            size = 1        # sample size
            ## ramdonly get a sample hadm_id from each patient's record
            hadm_sampled = self.sampling(pres_ade_df[['SUBJECT_ID','HADM_ID']].drop_duplicates(),
                                        size)
    #         pres_patient_hadm = pres_ade_df[['SUBJECT_ID','HADM_ID']].drop_duplicates()
    #         hadm_sampled = pres_patient_hadm.groupby('SUBJECT_ID', as_index=True).apply(fn)['HADM_ID']
            write2file(pd.DataFrame(hadm_sampled),join(self.outputpath,'HADM_ID_SAMPLE_PER_PATIENT'))
        self.hadm_sampled = hadm_sampled
        
    
    def ade_matrix():
        """
        
        Parameters
        ----------
        
        
        Notes
        -----
        Data Source: 
            1) PRESCRIPTION table (pres_df) 
            2) SIDER table (ade_df)
        
        
        Steps:
        ------
            1) remove drugs from PRESCRIPTION table that are not in SIDER table
        """

        ## PATIENT PRESCRIPTION LOG
        pres_df=read_data(join(read_prefix,'PRESCRIPTIONS'),dtype={'NDC':str}).dropna(subset=['NDC'])

        ## DRUG-ADE IN SIDER4, !!SIDER HAVE DUPLICATED RECORDS
        sideffect_prefix = '/data/liu/adverse_events'
        ade_df = read_data(join(sideffect_prefix, 'ndc_icd9_side_effects'), 
                           dtype={'NDC':str,'ICD9_CODE':str},usecols=['NDC','ICD9_CODE']).drop_duplicates()

        ## GET LIST OF DRUGS FROM SIDER4
        ade_drug=ade_df['NDC'].drop_duplicates()

        ## Remove records from Prescriptions where drugs cannot be found in SIDER
        pres_ade_df = pres_df[pres_df['NDC'].isin(ade_drug)].drop_duplicates()


        write2file(pres_ade_df,join(res_patient_subgroup_prefix,'PRESCRIPTION_SIDER'))



        pres_ade_sampled_df=pres_ade_df[pres_ade_df['HADM_ID'].isin(hadm_sampled)]
        # pres_ade_sampled_df.head()

        diaglog_sampled_df=diaglog_df[diaglog_df['HADM_ID'].isin(hadm_sampled)]


        pres_diag_sampled_df=inner_join(pres_ade_sampled_df,diaglog_sampled_df,['SUBJECT_ID','HADM_ID'])

        write2file(pres_diag_sampled_df,join(res_patient_subgroup_prefix,'pres_diag_sampled'))

        presdiag_SIDER_df = inner_join(pres_diag_sampled_df,ade_df,['NDC','ICD9_CODE'])

        write2file(presdiag_SIDER_df,join(res_patient_subgroup_prefix,'pres_diag_SIDER'))
        
        
        
        
        
        
        