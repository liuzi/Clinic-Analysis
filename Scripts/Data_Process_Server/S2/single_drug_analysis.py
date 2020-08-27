from utils.tools import *
import os, sys
import pandas as pd
from os.path import join


read_prefix = "/data/MIMIC3/"
# input_prefix="/data/liu/mimic3/CLAMP_NER/input"
# output_prefix="/data/liu/mimic3/CLAMP_NER/ner-attribute/output"
# save_csv_prefix="/data/liu/mimic3/CLAMP_NER/ner-attribute/output_csv"
# mapping_prefix = "/data/liu/mimic3/MAPPING/"
# processed_map_prefix="/data/liu/mimic3/MAPPING/PROCESSED/"
ade_related_prefix="/data/liu/mimic3/CLAMP_NER/ADE_Related/"
all_epis_clamp_prefix="/data/liu/mimic3/CLAMP_NER/ADE_Related/all"
output_prefix="/data/liu/mimic3/CLAMP_NER/single_drug_analysis"





# def df2mat(df,field):
#     df['value']=1
#     matrix = df.pivot_table(index='HADM_ID',columns=field,values='value',fill_value=0)
#     matrix = matrix.reset_index()
#     return matrix

def get_binarymatrix(raw_df,field, all_item):
    df = raw_df[["HADM_ID", field]]
    all_item_df=all_item
    all_item_df["HADM_ID"]='remove'
    df = df.append(all_item_df, sort=True)
    df['value']=1

    matrix = df.pivot_table(index='HADM_ID',columns=field,values='value',fill_value=0)

    matrix = matrix.reset_index()
    matrix= matrix[matrix['HADM_ID']!='remove']
    return matrix



def mat_noEmpty(mat,axis_num=0):
    if(axis_num):
        ## drop row
        mat = mat[(mat!=0).any(axis=1)]
    else:
        ## drop column
        mat = mat.loc[:,(mat!=0).any(axis=0)]
    return mat

def mat_df(mat,item):
    df1 = pd.DataFrame(mat.set_index('HADM_ID').stack()).reset_index()
    df2 = df1[df1[0]!=0].drop(0,1)
    df2.columns=["HADM_ID",item]
    return df2

    
def find_newdrugs(pres_df, output_dir):
    rxnorm_name_map = read_data(join(ade_related_prefix,"RxNorm_DRUGNAME"),dtype=str).set_index("RxNorm")['DRUG'].to_dict()
    
    mat_newdrug = get_new_item(read_data(join(all_epis_clamp_prefix,"hospital_course_drugMatrix"),dtype={"HADM_ID":str}),
                               read_data(join(all_epis_clamp_prefix,"medications_on_admission_drugMatrix"),dtype={"HADM_ID":str}),
                               rxnorm_name_map,full_df=pres_df[['HADM_ID','RxNorm']],output_dir=output_dir,item="RxNorm")
    return mat_newdrug

# def find_newdiseases(diag_df, output_dir):
#     code_name_map = read_data(join(read_prefix,"D_ICD_DIAGNOSES"),dtype=str).set_index("ICD9_CODE")['SHORT_TITLE'].to_dict()

#     mat_newdisease = get_new_item(read_data(join(all_epis_clamp_prefix,"hospital_course_diseaseMatrix"),dtype={"HADM_ID":str}),
#                                   read_data(join(all_epis_clamp_prefix,"medical_history_diseaseMatrix"),dtype={"HADM_ID":str}),
#                                   map_dict=code_name_map,full_df=diag_df,output_dir=output_dir)
#     return mat_newdisease

def update_op_des(file, NO, matrix):    
    output_description = {}
    output_description["Section"]=file
    description = NO + "=%d, Start_Index=%s, End_Index=%s"%(matrix.shape[1]-1, matrix.columns[1],matrix.columns[-1])
    output_description["Description"] = description
    return output_description
    
    
def get_hps_alldrugs(rxnorm, output_dir):
    ## get full prescription table, get list of treated patients(hospital stays)
    pres_df = read_data(join(ade_related_prefix,"pres_df"),dtype=str).dropna(subset=['RxNorm'])
    all_items = pres_df[["RxNorm"]].drop_duplicates()
    treated_hpstays = pres_df[pres_df['RxNorm']==rxnorm][['HADM_ID']].drop_duplicates()
    output_description = [{"Section":"Number of treated hospital stays", "Description":str(len(treated_hpstays))}]

    ## get total drugs for treated hps, P
    all_drugs_df = left_join(treated_hpstays,pres_df,['HADM_ID'])[['HADM_ID','RxNorm']].drop_duplicates()

    ## drug matrix
    all_drug_matrix = get_binarymatrix(all_drugs_df,"RxNorm",all_items)
    output_description=output_description+[update_op_des("Drugs in MIMIC prescriptions table", "P", all_drug_matrix)]
    
    ## new drugs
#     new_drug_matrix = find_newdrugs(all_drugs_df, output_dir)
#     output_description=output_description+ [update_op_des("“New” drugs over treated episodes in hospital stays","NP",new_drug_matrix)]

    ## new drugs
    new_drug_fulldf = read_data(join(all_epis_clamp_prefix,"allepis_newRxNorm"),dtype={"HADM_ID":str})
    new_drug_matrix = left_join(treated_hpstays,new_drug_fulldf,"HADM_ID").fillna(0)
    output_description=output_description+ [update_op_des("“New” drugs over treated episodes in hospital stays","NP",new_drug_matrix)]
    
    write2file(all_drug_matrix,join(output_dir,"all_drugs_matrix"))
    write2file(new_drug_matrix,join(output_dir,"new_drugs_matrix"))
    
    return treated_hpstays, all_drug_matrix, new_drug_matrix, output_description

    
def get_all_disease(rxnorm, treated_hpstays,output_dir):
    diaglog_df = read_data(join(read_prefix,'DIAGNOSES_ICD'),dtype={'SUBJECT_ID':str, "HADM_ID":str,'ICD9_CODE':str},\
                           usecols=['HADM_ID','ICD9_CODE']).dropna(subset=['ICD9_CODE']).drop_duplicates()
    all_items = diaglog_df[['ICD9_CODE']].drop_duplicates()
    all_disease_df = left_join(treated_hpstays,diaglog_df,['HADM_ID'])
     
    ## disease matrix
    all_disease_matrix = get_binarymatrix(all_disease_df,"ICD9_CODE",all_items)
    output_description=update_op_des("Diseases in MIMIC diagnosis table", "D", all_disease_matrix)
    
    ## new diseases
    new_disease_fulldf = read_data(join(all_epis_clamp_prefix,"allepis_newICD9_CODE"),dtype={"HADM_ID":str})
    new_disease_matrix = left_join(treated_hpstays,new_disease_fulldf,"HADM_ID").fillna(0)
#     = find_newdiseases(all_disease_df, output_dir)
    output_description=[output_description] + [update_op_des("“New” diseases over treated episodes in hospital stays","ND",new_disease_matrix)]
    
    ## diseases in  sider
#     sider_fulldf = read_data(join(ade_related_prefix,"sider4_v1"),dtype=str)[["RxNorm","ICD9_CODE"]].drop_duplicates()
#     sider_diseases = sider_fulldf[sider_fulldf['RxNorm']==rxnorm]['ICD9_CODE'].drop_duplicates()
#     sider_disease_matrix = new_disease_matrix.reindex(sider_diseases,drop=True)
    
    write2file(all_disease_matrix,join(output_dir,"all_diseases_matrix"))
    write2file(new_disease_matrix,join(output_dir,"new_diseases_matrix"))
#     write2file(new_disease_matrix,join(output_dir,"sider_disease_matrix"))

        
    return all_disease_matrix, new_disease_matrix, output_description   
    

## codes path:  /home/liu/project/Clinic-Analysis/Scripts/Data_Process_Server/S2    
## 1. output_description: list of statistics of results  
## 2. Refer to S2 Extract ADE tables with discharge summaries for finding new diseases/drugs for each episode

def main():
    
    if len(sys.argv) != 2:
        print("Wrong command format, please follwoing the command format below:")
        print("python single_drug_analysis.py [RxNorm]")
        exit(0)

    if len(sys.argv) == 2:
        ## folder for selected drug
        rxnorm = sys.argv[1]
        output_dir = join(output_prefix,rxnorm)
        create_folder(output_dir)
        
        ## get total drugs for treated hps, P
        treated_hpstays, all_drug_matrix, new_drug_matrix, output_description_list = get_hps_alldrugs(rxnorm, output_dir)

        
        ## get total diseases for treated hps, D
        all_disease_matrix, new_disease_matrix, output_description = get_all_disease(rxnorm, treated_hpstays,output_dir)
        output_description_list = output_description_list + output_description
        
        
        ## save outputdescription
#         output_description_df = pd.DataFrame(output_description_list)
#         output_description_df = df.ix[:, cols]
        write2file(pd.DataFrame(output_description_list,
                                columns=["Section","Description"]),join(output_dir,"output_description"))
    
        ## concat all matrices
        all_matrices = [matrix.set_index("HADM_ID") for matrix in [all_drug_matrix, 
                                                                   all_disease_matrix, 
                                                                   new_drug_matrix, 
                                                                   new_disease_matrix]]
        all_matrix = pd.concat(all_matrices, axis=1,sort=True).rename_axis('HADM_ID').reset_index()
        write2file(all_matrix,join(output_dir,"combined_matrix"))

    
main()




