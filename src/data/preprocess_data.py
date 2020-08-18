# -*- coding: utf-8 -*-
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
PATH_RAW = 'data/raw'
PATH_INTERIM = 'data/interim'
PATH_PROCESSED = 'data/processed'


def main():
    train_df = pd.read_csv(PROJECT_DIR.joinpath(PATH_RAW, 'train_sessions.csv'), index_col='session_id')
    test_df = pd.read_csv(PROJECT_DIR.joinpath(PATH_RAW, 'test_sessions.csv'), index_col='session_id')
    
    
    train_df = train_df.sort_values(by='time1')
    
    train_test_df = pd.concat([train_df, test_df], sort=False)
    train_test_df_sites = train_test_df[['site%d' % i for i in range(1, 11)]].fillna(0).astype('int')
    train_test_df_timestamps = train_test_df[['time%d' % i for i in range(1, 11)]].fillna(np.datetime64('NaT')).astype(np.datetime64)
    y = train_df['target'].astype('int').values
    
    
    with open(PROJECT_DIR.joinpath(PATH_INTERIM, 'sites.pkl'), 'wb') as sites_pkl:
        pickle.dump(train_test_df_sites, sites_pkl, protocol=2)
    
    with open(PROJECT_DIR.joinpath(PATH_INTERIM, 'timestamps.pkl'), 'wb') as timestamps_pkl:
        pickle.dump(train_test_df_timestamps, timestamps_pkl, protocol=2)
    
    with open(PROJECT_DIR.joinpath(PATH_INTERIM, 'train_size.pkl'), 'wb') as train_size_pkl:
        pickle.dump(len(train_df), train_size_pkl, protocol=2)
    
    with open(PROJECT_DIR.joinpath(PATH_PROCESSED, 'y.pkl'), 'wb') as y_pkl:
        pickle.dump(y, y_pkl, protocol=2)
    

if __name__ == '__main__':
    main()