import pandas as pd

def read_data(file_path, dtype=None, usecols=None, sep=',', header = 'infer', suffix = '.csv', pre=''):
    return pd.read_csv(file_path+suffix, dtype=dtype, usecols=usecols,sep=sep, header = header, encoding='latin1')

def write2file(df, file_path, suffix = '.csv'):
    df.to_csv(file_path+suffix, index=False)

def left_join(left_df, right_df, joined_field):
    return pd.merge(left_df, right_df, how='left', on=[joined_field])

def inner_join(left_df, right_df, joined_field):
    return pd.merge(left_df, right_df, how='inner', on=[joined_field])