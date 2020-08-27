import numpy as np
import pandas as pd
import re
from os.path import join
from utils.tools import *

read_prefix = "/data/MIMIC3/"
dis_sum_prefix = "/data/liu/mimic3/NOTESEVENT_EXTRACTION/ALL"

## Revised Section Extraction Script
def re_split(text, regex = r"\n\n[A-Z][A-Za-z/ ]+:"):
    """Split Discharge Summary to mulitple sections according to respective titles using regex rules

    Return:
    ------
    {titles:sections}: dict
        titiles : list. 
            Transfer all titles to lower case, making it easier to check and match
        line_number : int.
            line number of each title
        sections: list. 
            len(sections)=len(titles)+1. The first section doesn't have title as it is not beginning 
            with "\n"       
    """
    titles = re.findall(regex, text)
    sections = re.split(regex, text)[1:]
    split_lines=text.splitlines()
    title_linenums = [num+1 for num, line in zip(range(len(split_lines)),split_lines) if any(title.strip('\n') in line for title in titles)]
    
    ## 1) transform extracted titles to lower case  2)zip line numbers and sections
    return dict(zip(map(str.lower,titles), zip(title_linenums, sections)))
    
    
def extract_sections(text,rule_index = 0, rerules = [
    r"\n[A-Z][A-Z/ ]+:",
    r"\n\n[A-Z][A-Z/ ]+:",
    r"\n\n[A-Z][A-Za-z/ ]+:",
    r"\n[A-Z][A-Za-z/ ]+:"
], selected_titles = ["hospital course","medical history","medications on admission"]):
    """
    
    Step:
    -----
    1) split original text to multiple sections with titles: dict{title -> section}
    2) select key sections by matching their titles to selected titles
    """
    section_dict = re_split(text, rerules[rule_index])  
    # return section_dict

    ## selected title -> matched title
    titles_dict =dict([(selected_title, title) for title in section_dict.keys() 
                    for selected_title in selected_titles if selected_title in title])
    ## get actually extracted titles that are matched to selected titles
    section_titles = [titles_dict.get(title, None) for title in selected_titles]

    ## unzip line numbers and sections
    line_numbers, sections = list(zip(*[section_dict.get(title, None) for title in section_titles]))
    
    # return section_titles
    return tuple([(',').join(filter(None, section_titles)),(',').join(filter(None, map(str,line_numbers)))])
    # +list(sections))
    
                    
notes_discharge_df=read_data(join(dis_sum_prefix,"Discharge summary_All"),dtype={"HADM_ID":str})
# print(len(notes_discharge_df))
notes_discharge_df.head()
notes_df=notes_discharge_df
test_df = notes_df[notes_df['HADM_ID']=="196600"]
test_df_text = test_df.loc[36180,"TEXT"]
titles = extract_sections(test_df_text,3)
print(titles)
