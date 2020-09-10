from _path import *
from _tools import *
from os.path import *

import io

def list_delta(str1, str2):
    l1=[i for i in str1.split(",") if i!="None"]
    l2=[i for i in str2.split(",") if i!="None"]
    if(l1):
        delta = [y-x for x,y in zip(map(int,l1),map(int,l2))]
        return delta
    else:
        return None

def get_one_row(df, row_id):
    
    for col, value in zip(df.columns[4:9],df.loc[row_id,df.columns[4:9]]):
        print('{} :\n {}\n'.format(col,value))

    print("\n{}TEXT{}\n".format("-"*45,"-"*45))

    with io.StringIO(df.loc[row_id,"TEXT"]) as f:
        for i, line in enumerate(f, start=1):
            print('{} : {}'.format(i, line.strip()))
            
    print("\n{}MEDICATIONS ON ADMISSION{}\n".format("-"*45,"-"*45))
    print(df.loc[row_id,"MEDICATIONS ON ADMISSION"])


def create_benchmark(rule_id,benchmark_folder):

    rerule_dissum_extract_file="ReRule%d_Discharge summary_Only"
    benchmark_cols=["HADM_ID","GROUP_RANK","delta","check delta in [1,2]"]

    dissum_df = read_data(
        join(clamp_input_prefix,rerule_dissum_extract_file%rule_id),
        dtype={"HADM_ID":str})

    print("Rule %d:\n\n   Original DF:"%rule_id)
    print_patient_stats(dissum_df)
    # dissum_df.head()
    ## create benchamrk
    dissum_df["delta"]=dissum_df[["SECTION_END_LINE","NEXT_TITLE_LINE"]].apply(
        lambda x: list_delta(*x), axis=1)
    dissum_df["check delta in [1,2]"]=dissum_df["delta"].apply(
        lambda x: all(delta in [1,2] for delta in x))
    
    abnormal_df, normal_df = [
        x for _, x in dissum_df.groupby(dissum_df["check delta in [1,2]"])]
    
    print("\n   Normal DF:" )
    # print(normal_df['check delta in [1,2]'].unique())
    print_patient_stats(normal_df)

    print("\n   Abnormal DF:" )
    # print(abnormal_df['check delta in [1,2]'].unique())
    print_patient_stats(abnormal_df)
    write2file(normal_df[benchmark_cols],join(
        benchmark_folder,rerule_dissum_extract_file%rule_id+"_NORMAL"))
    write2file(abnormal_df[benchmark_cols],join(
        benchmark_folder,rerule_dissum_extract_file%rule_id+"_ABNORMAL"))


def run_for_4rules():
    benchmark_folder=join(clamp_input_prefix,"RULE_BENCHMARK")
    create_folder_overwrite(benchmark_folder) 
    for i in range(0,4):
        create_benchmark(i,benchmark_folder)

run_for_4rules()



