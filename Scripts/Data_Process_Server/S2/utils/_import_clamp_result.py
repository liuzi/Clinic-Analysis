import os
import re
from _path import *
from _tools import *
import pandas as pd
import glob


## Import CLAMP NER result
def splitfilename(filename):
    ids= filename.rsplit(".",1)[0].split("_")
    return ids

def readtxt(filename):
    text=""
    with open(filename,'r') as myfile:
        text=myfile.read()
    return text
                         
def parseNErow(row):
    indexes=row[1:3]
    attrs=[attr.split('=')[1] for attr in row[3:] if ("=" in attr)]
    return indexes+attrs
    
def parseNEtext(text):
    df = pd.DataFrame([x.split('\t') for x in text.split('\n')])
    return df

def addidcolumns(file, df):
    idcols=['SUBJECT_ID','HADM_ID','GROUP_RANK']
    ids = splitfilename(file)
    for i in range(3):
        df.insert(i, idcols[i], ids[i])
    return df

def get_df_list(folder_path):
    os.chdir(folder_path)
    files = [f for f in glob.glob("*.txt") if (os.path.isfile(f))]
    # print(files[:5])

    li=[]
    for file in files:
        text=readtxt(file)
        df = parseNEtext(text.split("\nSentence",1)[0])
        if(df is not None):
            li.append((file, df))
    concatcsv = [addidcolumns(file, df) for file, df in li]
    if(len(concatcsv)>0):
        concateddf = pd.concat(concatcsv, axis=0, sort=True)
        return concateddf
    else:
        return None

def concatcsvs(write=False):
    rule_dirs = [
        os.path.join(clamp_output_prefix, dir) for dir in os.listdir(clamp_output_prefix)
        if os.path.splitext(dir)[0].startswith('Rule')]
    # print(rule_dirs)
    output_prefix = os.path.join(clamp_output_prefix, "CONCAT_Results")
    create_folder(output_prefix)
        
    sub_dirs = os.listdir(rule_dirs[0])
#    print(rule_benchmark_episodes.head())


    rule_benchmark_episodes=read_data(
        os.path.join(
            clamp_rule_benchmark,"ReRule_Discharge summary_Only_NORMAL_ALL"),dtype=str)

    for sub_dir in sub_dirs:
        # print(sub_dir)
        # print([
        #     os.path.join(rule_dirs[rule_index],
        #     sub_dir) for rule_index in range(0,len(rule_dirs))])

        df_rulelist = [
            get_df_list(os.path.join(rule_dirs[rule_index],sub_dir))
            for rule_index in range(0,len(rule_dirs))]
        
        # print(df_rulelist[0])
        df_rule_concat = pd.concat(
            [df_rule for df_rule in df_rulelist if df_rule is not None], axis=0)
        df_rule_concat = left_join(
            rule_benchmark_episodes,df_rule_concat,list(rule_benchmark_episodes.columns))

        if(write):
            write2file(df_rule_concat,os.path.join(
                output_prefix,"%s_all_results_from_clamp_nerattribute")%sub_dir)
        
        # print(section_result_map.head())




# HACK: FEATURE 2
def clean_value(value):

    if isinstance(value, str):
        if value.find("="):
            return re.sub('[\[\]]', '', value.split("=")[-1])

def getdrug(df):
    drugdf = df[(df['3']=="semantic=drug") & (df['4'].str.contains("RxNorm"))].drop(columns=['0']).iloc[:,0:7]
    cols=list(drugdf.columns[:3])+["START INDEX","END INDEX","SEMANTIC","CUI"]
    drugdf.columns=cols
    drugdf = drugdf.reset_index()
            
    drugdf["CUI"], drugdf["RxNorm"], drugdf["Generic"] = drugdf.CUI.str.split(",").str

    for col in drugdf.columns[6:]:
        drugdf[col].astype(str)
        drugdf[col]=drugdf[col].apply(clean_value)

    return drugdf.drop(columns=["index"])

def getdisease(df):
    diseasedf = df[(df['3']=="semantic=problem") & (df['5'].str.contains("SNOMEDCT_US"))].drop(columns=['0']).iloc[:,0:11]
    cols=list(diseasedf.columns[:3])+["START INDEX","END INDEX","SEMANTIC","ASSERTION","CUI","sentProb","conceptProb","NE"]
    diseasedf.columns=cols
    
    diseasedf = diseasedf.reset_index()
    
    diseasedf["CUI"], diseasedf["SNOMEDCT_US"] = diseasedf.CUI.str.split(" ",1).str
    diseasedf=diseasedf.dropna(subset=["SNOMEDCT_US"])
    diseasedf["SNOMEDCT_US"]=diseasedf["SNOMEDCT_US"].apply(lambda x: x[x.find("[")+1:-1])
    return diseasedf.drop(columns=["index"])

def extract_items():
    read_dtype={"HADM_ID":str, "GROUP_RANK":str}
    section_titles=['HOSPITAL_COURSE', 'MEDICAL_HISTORY', 'MEDICATIONS_ON_ADMISSION']

    # icd9_SNOMED_df =pd.read_csv(join(mapping_prefix,'ICD9CM_SNOMED_MAP_1TO1_201912.txt')
    #                     ,sep='\t', dtype={'ICD9_CODE':str, 'SNOMED_CID':str}).dropna(subset=['SNOMED_CID'])
    # icd9_SNOMED_df = icd9_SNOMED_df[["ICD_CODE","SNOMED_CID"]]
    # icd9_SNOMED_df['ICD9_CODE']=icd9_SNOMED_df['ICD_CODE'].apply(lambda x: x.replace(".",""))
    # write2file(icd9_SNOMED_df,join(processed_map_prefix,"snomed_icd9_df"))
    icd9_SNOMED_df=read_data(os.path.join(processed_map_prefix,"snomed_icd9_df"),dtype={"SNOMED_CID":str,"ICD9_CODE":str})
    icd9_SNOMED_map = icd9_SNOMED_df.set_index('SNOMED_CID')['ICD9_CODE'].to_dict()

    for section_title in section_titles:
        df=read_data(os.path.join(clamp_output_prefix,"CONCAT_Results", "%s_all_results_from_clamp_nerattribute")%section_title, dtype=read_dtype)
        # print(df.columns)
        drugdf=getdrug(df)
        write2file(drugdf,os.path.join(clamp_output_prefix,"CONCAT_Results","%s_ALL_Drugs"%section_title))
        del drugdf

        diseasedf=getdisease(df)
        diseasedf["ICD9_CODE"]=diseasedf["SNOMEDCT_US"].apply(lambda x: (",").join(
            filter(None,[icd9_SNOMED_map.get(key, None) for key in x.strip().split(',')])))
        write2file(diseasedf,os.path.join(clamp_output_prefix,"CONCAT_Results","%s_ALL_Diseases"%section_title))

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

def get_binarymatrix(raw_df,extract_fields, all_item):
    df = raw_df[extract_fields]
    all_item_df=all_item
    all_item_df[extract_fields[0]]='remove'
    df = df.append(all_item_df, sort=True)
    df['value']=1

    matrix = df.pivot_table(index='HADM_ID',columns=extract_fields[1],values='value',fill_value=0)

    matrix = matrix.reset_index()
    matrix= matrix[matrix['HADM_ID']!='remove']
    return matrix


def get_allitems_across_sections():
    all_diseases = []
    for folder in ['HOSPITAL_COURSE', 'MEDICAL_HISTORY', 'MEDICATIONS_ON_ADMISSION']:
        new_diseases = read_data(os.path.join(
            clamp_output_prefix,"CONCAT_Results","%s_ALL_Diseases"%folder),
            dtype={'SUBJECT_ID':str,'HADM_ID':str}).dropna(subset=['ICD9_CODE'])["ICD9_CODE"].unique().tolist()
        print("%s: # of DISEASE"%folder)
        print(len(new_diseases))
        all_diseases=all_diseases+new_diseases

    all_diseases=list(set(all_diseases))
    all_diseases = pd.DataFrame({"ICD9_CODE":all_diseases})
    write2file(all_diseases,os.path.join(clamp_output_prefix,"CONCAT_Results","all_diseases_3sections"))

    all_drug = []
    for folder in ['HOSPITAL_COURSE', 'MEDICAL_HISTORY', 'MEDICATIONS_ON_ADMISSION']:
        cols=["SUBJECT_ID","HADM_ID","RxNorm"]
        new_drug_df = read_data(os.path.join(clamp_output_prefix,"CONCAT_Results","%s_ALL_Drugs"%folder),
                                dtype={'SUBJECT_ID':str,'HADM_ID':str})[cols]

        print("%s: # of Drugs"%folder)
        print(len(new_drug_df['RxNorm'].unique()))
        all_drug = all_drug+list(new_drug_df['RxNorm'].unique())    
    all_drug = list(set(all_drug))
    all_drug_df = pd.DataFrame({"RxNorm":all_drug})
    write2file(all_drug_df,os.path.join(clamp_output_prefix,"CONCAT_Results","all_drugs_3sections"))


def get_matrix_from_dissum():
    all_diseases = read_data(os.path.join(clamp_output_prefix,"CONCAT_Results","all_diseases_3sections"),dtype=str)
    all_drugs = read_data(os.path.join(clamp_output_prefix,"CONCAT_Results","all_drugs_3sections"),dtype=str)


    for folder in ['HOSPITAL_COURSE', 'MEDICAL_HISTORY', 'MEDICATIONS_ON_ADMISSION']:
        # NOTE: diseases
        dissum_disease_df = read_data(os.path.join(
            clamp_output_prefix,"CONCAT_Results","%s_ALL_Diseases"%folder),
            dtype={'SUBJECT_ID':str,'HADM_ID':str}).dropna(subset=['ICD9_CODE'])[["HADM_ID","SNOMEDCT_US","ICD9_CODE"]]

        print("{:}: {:,d} episodes, {:,d} SNOMEDCT_US, {:,d} ICD9_CODE, {:,d} Rows".format(folder,
            len(dissum_disease_df.HADM_ID.unique()), len(dissum_disease_df.SNOMEDCT_US.unique()),len(dissum_disease_df.ICD9_CODE.unique()), len(dissum_disease_df)))
            
        disease_matrix = get_binarymatrix(dissum_disease_df[['HADM_ID','ICD9_CODE']],extract_fields =['HADM_ID','ICD9_CODE'], all_item=all_diseases)   
        print(disease_matrix.shape)
        write2file(disease_matrix,os.path.join(clamp_output_prefix,"CONCAT_Results","%s_diseaseMatrix"%folder))

        # NOTE: drugs
        cols=["SUBJECT_ID","HADM_ID","RxNorm"]
        dissum_drug_df = read_data(os.path.join(
            clamp_output_prefix,"CONCAT_Results","%s_ALL_Drugs"%folder),
            dtype={'SUBJECT_ID':str,'HADM_ID':str})[cols]

        print("{:}: {:,d} episodes, {:,d} RxNorm codes, {:,d} Rows".format(folder,
            len(dissum_drug_df.HADM_ID.unique()), len(dissum_drug_df.RxNorm.unique()), len(dissum_drug_df)))
        
        drug_matrix = get_binarymatrix(dissum_drug_df[['HADM_ID','RxNorm']],extract_fields =['HADM_ID','RxNorm'],all_item=all_drugs)    
        print(drug_matrix.shape)
        write2file(drug_matrix,os.path.join(clamp_output_prefix,"CONCAT_Results","%s_drugMatrix"%folder))
    #     save_items_byfolder(patients_drugs,file="drugs_%s"%folder,newfolder=False)


## Revised version: New items per episode
def get_new_item(
    mat1, mat2,
    # map_dict,
    # full_df, 
    epi_field="HADM_ID",item="ICD9_CODE"):
#     both_episodes = set(mat1[epi_field].unique()).union(set(mat2[epi_field].unique()))
    common_episodes = set(mat1[epi_field].unique()).intersection(set(mat2[epi_field].unique()))
    
    df3 = mat_df(mat1,item).append(mat_df(mat2,item)).drop_duplicates()
    # df4 = full_df.merge(df3, on=["HADM_ID",item], how='left', indicator=True)
    # df_missEPi = df4[df4['_merge']=='left_only'].drop("_merge",1)
#     write2file(df_missEPi,join(drug_single_dir,"missing_episode%s"%item))

    print("Matrix including HADM_ID as the first column")
    if(item=="RxNorm"):
        print("medications on admission, hospital course")
    else:
        print("medical history, hospital course")

    print((mat_noEmpty(mat2).shape))
    print((mat_noEmpty(mat1).shape))

    common_episodes_df = pd.DataFrame({epi_field:list(common_episodes)})
    mat1_comepi = mat_noEmpty(left_join(common_episodes_df,mat1,epi_field))
    mat2_comepi = mat_noEmpty(left_join(common_episodes_df,mat2,epi_field))
    if(item=="RxNorm"):
        print("medications on admission, hospital course, after keeping common episodes")
    else:
        print("medical history, hospital course, after keeping common episodes")
    print((mat2_comepi.shape))
    print((mat1_comepi.shape))

    ## mat1: hospital course
    ## List of new codes are based on collection of codes in hospital course
    mat2_comepi_standard=mat2_comepi.reindex(list(mat1_comepi.columns),axis=1).fillna(0)
    
    # find new item
    mat1_newitems = (mat1_comepi.set_index("HADM_ID")>mat2_comepi_standard.set_index("HADM_ID"))*1
    
    ## remove empty rows (hospital stays) and columns (diseases)
    mat1_newitems = mat_noEmpty(mat1_newitems,axis_num=1)
    mat1_newitems = mat_noEmpty(mat1_newitems,axis_num=0)
    print("Matrix without HADM_ID as the first column")
    print("New %s Matrix"%item)
    print(mat1_newitems.shape)
    write2file(mat1_newitems.reset_index(),os.path.join(clamp_output_prefix,"CONCAT_Results","allepis_new%s"%item))
    
    ## reset index (hadm_id) as the first column
    # mat1_newitems_distri = pd.DataFrame(mat1_newitems.sum()).reset_index()
    # mat1_newitems_distri.columns=[item, "NUM_OF_EPISODES"]
    # mat1_newitems_distri[name]=mat1_newitems_distri[item].apply(lambda x: map_dict.get(x,None))
    
    # print("Number of espisodes containing new items: %d" % len(mat1_newitems))
    # write2file(mat1_newitems_distri,join(drug_single_dir,"distribution_of_new%s"%item))

def get_new_diseases_drugs():
    ## revised: get new diseases
    # code_name_map = read_data(join(read_prefix,"D_ICD_DIAGNOSES"),dtype=str).set_index("ICD9_CODE")['SHORT_TITLE'].to_dict()
    # diaglog_df = read_data(join(read_prefix,'DIAGNOSES_ICD'),dtype={'SUBJECT_ID':str, "HADM_ID":str,'ICD9_CODE':str},\
    #                     usecols=['SUBJECT_ID','HADM_ID','ICD9_CODE']).dropna(subset=['ICD9_CODE']).drop_duplicates()

    get_new_item(read_data(os.path.join(
        clamp_output_prefix,"CONCAT_Results","HOSPITAL_COURSE_diseaseMatrix"),
        dtype={"HADM_ID":str}),
        read_data(os.path.join(clamp_output_prefix,"CONCAT_Results","MEDICAL_HISTORY_diseaseMatrix"),
        dtype={"HADM_ID":str}))

    ## TODO: FIND drug rxnorm map creating codes
    # rxnorm_name_map = read_data(join(ade_related_prefix,"RxNorm_DRUGNAME"),dtype=str).set_index("RxNorm")['DRUG'].to_dict()
    ## get new drugs
    get_new_item(read_data(os.path.join(
        clamp_output_prefix,"CONCAT_Results","HOSPITAL_COURSE_drugMatrix"),dtype={"HADM_ID":str}),
        read_data(os.path.join(clamp_output_prefix,"CONCAT_Results","MEDICATIONS_ON_ADMISSION_drugMatrix"),dtype={"HADM_ID":str}),
        epi_field="HADM_ID",item="RxNorm")



# HACK:
# concatcsvs(True)
# extract_items()
# get_allitems_across_sections()
# get_matrix_from_dissum()
get_new_diseases_drugs()
# HACK:        
          

# print("shabi")
# get_df_list("/data/liu/mimic3/CLAMP_NER/ner-attribute/output/Rule2/MEDICATIONS_ON_ADMISSION")
    
# print(rule_benchmark)        
# print(glob.glob(os.path.join(clamp_rule_benchmark,"*_NORMAL.csv")))
    