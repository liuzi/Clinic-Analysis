import pandas as pd
import os
# import errno
import json
import shutil
from os import makedirs

def read_data(file_path, dtype=None, usecols=None, sep=',', header = 'infer', suffix = '.csv', pre=''):
    return pd.read_csv(file_path+suffix, dtype=dtype, usecols=usecols,sep=sep, header = header, encoding='latin1')

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

def write2file(df, file_path, suffix = '.csv'):
    df.to_csv(file_path+suffix, index=False)

def left_join(left_df, right_df, joined_field):
    return pd.merge(left_df, right_df, how='left', on=joined_field)

def inner_join(left_df, right_df, joined_field):
    return pd.merge(left_df, right_df, how='inner', on=joined_field)
