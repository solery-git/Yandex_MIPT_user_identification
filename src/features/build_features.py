# -*- coding: utf-8 -*-
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MaxAbsScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

PROJECT_DIR = Path(__file__).resolve().parents[2]
PATH_RAW = 'data/raw'
PATH_INTERIM = 'data/interim'
PATH_PROCESSED = 'data/processed'


def make_features(X_sites, X_times):
    def count_unique_sites(row):
        row_unique = np.unique(row)
        return len(row_unique[row_unique != 0]) #0 is not a valid site ID
    
    def count_popular_sites(sites_df, popular_sites_indicators):
        result = []
        for sites_row, popular_sites_mask in zip(sites_df.values, popular_sites_indicators):
            popular_sites = sites_row[popular_sites_mask]
            result.append( len(popular_sites) )
        return result
    
    def get_session_timespan(row):
        row_ne = row[~np.isnat(row)]
        return int((row_ne[-1]-row_ne[0]) / np.timedelta64(1, 's'))
    
    def between(left, x, right):
        return left <= x and x <= right
    
    def get_time_of_day(hour):
        if between(0, hour, 6):
            return 0
        elif between(7, hour, 11):
            return 1
        elif between(12, hour, 18):
            return 2
        elif between(19, hour, 23):
            return 3
    
    
    X_features = pd.DataFrame(index=X_sites.index)
    
    sites_counter = Counter()
    for row in X_sites.values:
        sites_counter.update(row)
    popular_sites = [site for site, count in sites_counter.most_common(30)]
    popular_sites_indicators = np.isin(X_sites.values, popular_sites)
    
    X_features['start_hour'] = X_times['time1'].dt.hour
    X_features['time_of_day'] = X_features['start_hour'].apply(get_time_of_day)
    X_features['day_of_week'] = X_times['time1'].dt.dayofweek
    X_features['weekend'] = (np.isin(X_features['day_of_week'].values, [5, 6])).astype(int)
    #X_features['#unique_sites'] = X_sites.apply(count_unique_sites, raw=True, axis=1)
    X_features['#popular_sites'] = count_popular_sites(X_sites, popular_sites_indicators)
    #X_features['%popular_sites'] = X_features['#popular_sites'] / X_features['#unique_sites']
    X_features['year_month'] = X_times['time1'].dt.strftime('%y%m').astype(int)
    X_features['month'] = X_times['time1'].dt.month
    X_features['session_timespan'] = X_times.apply(lambda row: get_session_timespan(row), axis=1, raw=True)

    return X_features


def main():
    with open(PROJECT_DIR.joinpath(PATH_INTERIM, 'sites_train_test.pkl'), 'rb') as fin:
        sites_train_test = pickle.load(fin)
    
    with open(PROJECT_DIR.joinpath(PATH_INTERIM, 'timestamps_train_test.pkl'), 'rb') as fin:
        timestamps_train_test = pickle.load(fin)
    
    
    features_train_test = make_features(sites_train_test, timestamps_train_test)
    
    
    with open(PROJECT_DIR.joinpath(PATH_INTERIM, 'features_train_test.pkl'), 'wb') as fout:
        pickle.dump(features_train_test, fout, protocol=2)

if __name__ == '__main__':
    main()