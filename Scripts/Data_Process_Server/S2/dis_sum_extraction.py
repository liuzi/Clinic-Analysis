import numpy as np
import pandas as pd
import re
from os.path import join
from utils.tools import *

read_prefix = "/data/MIMIC3/"
dis_sum_prefix = "/data/liu/mimic3/NOTESEVENT_EXTRACTION/ALL"
write_prefix="/data/liu/mimic3/CLAMP_NER/input"

def clean_string(string):
    return string.strip('\n').lower()

## Revised Section Extraction Script
def re_split(text, regex = r"\n\n[A-Z][A-Za-z/ ]+:"):
    """Split Discharge Summary to mulitple sections according to respective titles using regex rules

    Return:
    ------
    {titles:sections}: dict
        titiles : list. 
            1) Remove enter character
            2) Transfer all titles to lower case, making it easier to check and match
        line_number : int.
            line number of each title
        sections: list. 
            len(sections)=len(titles)+1. The first section doesn't have title as it is not beginning 
            with "\n"       
    """
    titles = re.findall(regex, text)
    sections = re.split(regex, text)[1:]
    split_lines=text.splitlines()
    title_linenums = [num+1 for num, line in zip(range(len(split_lines)),split_lines) 
                        if any(title.strip('\n') in line for title in titles)]
    
    ## 1) transform extracted titles to lower case  2)zip line numbers and sections
    if len(titles):
        return dict(zip(map(clean_string,titles), zip(title_linenums, sections)))
    else:
        return None
    
    
def extract_sections(text,rule_index = 0, rerules = [
    r"\n[A-Z][A-Z/ ]+:",
    r"\n\n[A-Z][A-Z/ ]+:",
    r"\n\n[A-Z][A-Za-z/ ]+:",
    r"\n[A-Z][A-Za-z/ ]+:"
], selected_titles = ["hospital course","medical history","medications"]):
    """

    Arguments:
    ---------
    selected_titles: list
        1)hospital course: brief hc
        2)medical history:
        3)medications on admission: admission medications
    
    Step:
    -----
    1) split original text to multiple sections with titles: dict{title -> section}
    2) select key sections by matching their titles to selected titles
    """
    section_dict = re_split(text, rerules[rule_index])  
    # return section_dict

    if(section_dict):
        ## selected title -> matched title
        titles_dict =dict([(selected_title, title) for title in section_dict.keys() for selected_title in selected_titles if selected_title in title])
        ## get actually extracted titles that are matched to selected titles
        section_titles = [titles_dict.get(title, None) for title in selected_titles]
        ## unzip line numbers and sections       
        line_numbers, sections = list(zip(*[section_dict.get(title, (None,None)) for title in section_titles]))
        # return section_titles
        return tuple([(',').join(filter(None, section_titles)),
            (',').join(filter(None, map(str,line_numbers)))]+list(sections))

    else:
        return tuple([None]*5)

def save_keysessions_byfile(
    notes_discharge_df,
    rule_index,
    file="%s_%s_%d",
    folder_titles = ['HOSPITAL COURSE', 'MEDICAL HISTORY', 'MEDICATIONS ON ADMISSION']):
    
    write_rule_prefix = join(write_prefix, "Rule%d"%rule_index)
    create_folder(write_rule_prefix)
    tilfolder_path = [join(write_rule_prefix, title) for title in folder_titles]
    [create_folder(path) for path in tilfolder_path]
    title_path_dict = dict(zip(folder_titles, tilfolder_path))
    
    for subject_id, group in notes_discharge_df.groupby(['SUBJECT_ID']):
        hadm_ids = group['HADM_ID'].unique()
        ## create folder ~/subject_id/hadm_id
        cols = ['GROUP_RANK','HOSPITAL COURSE', 'MEDICAL HISTORY', 'MEDICATIONS ON ADMISSION']
        
        for hadm_id, subgroup in group.groupby(['HADM_ID']):    
            
            for _, row in subgroup[cols].iterrows():
                file_name = file%(str(subject_id), hadm_id, row['GROUP_RANK'])
                [write2txt(row[title], join(
                    title_path_dict[title],file_name)) for title in folder_titles if row[title]]
                # write2txt(row["HOSPITAL COURSE"],join(course_pre,file_name))
                # write2txt(row["MEDICAL HISTORY"],join(pastdrug_pre,file_name))
                # write2txt(row["MEDICATIONS ON ADMISSION"],join(admi_medi_pre,file_name))        
        
rule_index = 0
notes_discharge_df=read_data(join(dis_sum_prefix,"Discharge summary_All"),dtype={"HADM_ID":str})
add_cols = ["TITLE", "LINE NUMBER", "HOSPITAL COURSE", "MEDICAL HISTORY", "MEDICATIONS ON ADMISSION"]
notes_discharge_df[add_cols] = pd.DataFrame(notes_discharge_df['TEXT'].apply(lambda x: 
                                    extract_sections(x,rule_index)).tolist())

write2file(notes_discharge_df,join(write_prefix,"ReRule%d_Discharge summary_All"%rule_index))
save_keysessions_byfile(notes_discharge_df,rule_index)


"""
    test code: one text
"""
# notes_discharge_df=read_data(join(dis_sum_prefix,"Discharge summary_All"),dtype={"HADM_ID":str})
# # print(len(notes_discharge_df))
# notes_discharge_df.head()
# notes_df=notes_discharge_df
# test_df = notes_df[notes_df['HADM_ID']=="196600"]
# test_df_text = test_df.loc[36180,"TEXT"]
# titles = extract_sections(test_df_text,3)
# print(len(titles))
