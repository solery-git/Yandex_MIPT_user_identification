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


def csr_hstack(arglist):
    return csr_matrix(sparse_hstack(arglist))

def sparsify_data(X, vectorizer_params, site_dic, train_part=None, method='count'):
    id2site = {v:k for (k, v) in site_dic.items()}
    id2site[0] = 'unknown'
    
    X_text = [' '.join(map(id2site.get, row)) if len(row) > 0 else '' for row in X]
    #X_text = [' '.join(map(str, row)) for row in X]

    if method == 'count':
        vectorizer = CountVectorizer(**vectorizer_params)
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer(**vectorizer_params)
    else:
        raise ValueError(method)
    
    if train_part is None:
        vectorizer.fit(X_text)
    else:
        vectorizer.fit(X_text[:train_part])
    
    return vectorizer.transform(X_text)

def prepare_features(X_features, feature_types):
    def extract_features(feat_type):
        empty_array = np.array([]).reshape(X_features.shape[0], 0)
        if len(feature_types[feat_type]) > 0:
            return X_features[feature_types[feat_type]].values
        else:
            return empty_array
    
    def transform_features(transformer, feat_array):
        if feat_array.shape[1] > 0:
            return transformer.fit_transform(feat_array)
        else:
            return feat_array
    
    prepared_features = extract_features('prepared')
    encoded_features = transform_features(OneHotEncoder(sparse=True, dtype=np.int16), extract_features('categorical'))
    log_features = np.log(extract_features('to_log') + 1)
    #also standard scale log features
    scaled_standard_features = transform_features(StandardScaler(), np.hstack([extract_features('to_scale_standard'), 
                                                                               log_features]))
    scaled_maxabs_features = transform_features(MaxAbsScaler(), extract_features('to_scale_maxabs'))
    
    return csr_hstack([prepared_features, 
                       encoded_features, 
                       scaled_standard_features, 
                       scaled_maxabs_features])

def make_features(X_sites, X_times, prepare=True):
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
    X_features['session_timespan'] = X_times.apply(lambda row: get_session_timespan(row), axis=1, raw=True)
    
    feature_types = {'prepared': [],
                     'categorical': ['time_of_day', 
                                    ], 
                     'to_log': [], 
                     'to_scale_standard': ['session_timespan'], 
                     'to_scale_maxabs': ['day_of_week', 'year_month']
                    }
    
    if prepare:
        return prepare_features(X_features, feature_types)
    else:
        return X_features


def main():
    with open(PROJECT_DIR.joinpath(PATH_INTERIM, 'sites.pkl'), 'rb') as sites_pkl:
        train_test_df_sites = pickle.load(sites_pkl)
    
    with open(PROJECT_DIR.joinpath(PATH_INTERIM, 'timestamps.pkl'), 'rb') as timestamps_pkl:
        train_test_df_timestamps = pickle.load(timestamps_pkl)
    
    with open(PROJECT_DIR.joinpath(PATH_INTERIM, 'train_size.pkl'), 'rb') as train_size_pkl:
        train_size = pickle.load(train_size_pkl)
    
    with open(PROJECT_DIR.joinpath(PATH_RAW, 'site_dic.pkl'), 'rb') as site_dic_pkl:
        site_dic = pickle.load(site_dic_pkl)
    
    
    vectorizer_params = {'ngram_range': (1, 5), 'max_features': 50000, 'tokenizer': lambda s: s.split(), 'stop_words': ['unknown']}
    train_test_sparse_with_features = csr_hstack([#sparsify_data(train_test_df_sites.values, vectorizer_params, site_dic, train_size), 
                                                  sparsify_data(train_test_df_sites.values, vectorizer_params, site_dic, train_size, method='tfidf'), 
                                                  #encode_sites_with_time_diffs(train_test_df_sites, train_test_df_timestamps), 
                                                  #make_site_names(train_test_df_sites, site_dic, method='tfidf'), 
                                                  make_features(train_test_df_sites, train_test_df_timestamps) 
                                                 ])
    
    X_train_sparse = train_test_sparse_with_features[:train_size]
    X_test_sparse = train_test_sparse_with_features[train_size:]
    
    
    with open(PROJECT_DIR.joinpath(PATH_PROCESSED, 'X_train.pkl'), 'wb') as X_train_pkl:
        pickle.dump(X_train_sparse, X_train_pkl, protocol=2)
    
    with open(PROJECT_DIR.joinpath(PATH_PROCESSED, 'X_test.pkl'), 'wb') as X_test_pkl:
        pickle.dump(X_test_sparse, X_test_pkl, protocol=2)

if __name__ == '__main__':
    main()