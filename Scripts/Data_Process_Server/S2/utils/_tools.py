import pandas as pd
import os
# import errno
import json
import shutil
from pathlib import Path

def csv_suffix(file_path,suffix = '.csv'):
    if(file_path.endswith(suffix)):
        final_file_path=file_path
    else:
        final_file_path = file_path+suffix
    return final_file_path

def read_data(file_path, dtype=None, usecols=None, sep=',', header = 'infer', suffix = '.csv', pre=''):
    return pd.read_csv(csv_suffix(file_path), dtype=dtype, usecols=usecols,sep=sep, header = header, encoding='latin1')

def write2txt(string, file_path):
    textfile = open("%s.txt"%file_path, 'w')
    textfile.write(string)
    textfile.close()
    
def write2json(jdata,file_path):   
    with open("%s.json"%file_path, 'w') as fp:
        json.dump(jdata, fp)
    
# def create_folder(dir):

#     if not os.path.exists(dir):
#         try:
#             os.makedirs(dir)
#         except OSError as exc: # Guard against race condition
#             if exc.errno != errno.EEXIST:
#                 raise

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError:
        print ("Directory %s already exists" % path)
    else:
        print ("Successfully create the directory %s" % path)

def create_folder_overwrite(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def write2file(df, file_path):
    df.to_csv(csv_suffix(file_path), index=False)

def write2file_nooverwrite(df, file_path):
    if Path(csv_suffix(file_path)).exists():
        print("%s already exists."%csv_suffix(file_path))
    else:
        df.to_csv(file_path+suffix, index=False)
        print("%s is successfully saved."%csv_suffix(file_path))


def left_join(left_df, right_df, joined_field):
    return pd.merge(left_df, right_df, how='left', on=joined_field)

def inner_join(left_df, right_df, joined_field):
    return pd.merge(left_df, right_df, how='inner', on=joined_field)

def print_patient_stats(df):
    
    print("# of rows: %d"%len(df))
    
    for col in df.columns:
        print("# of %s: %d"%(col,len(df[col].unique())))
        # print("# of %s: %d"%len(df['HADM_ID'].unique()))
