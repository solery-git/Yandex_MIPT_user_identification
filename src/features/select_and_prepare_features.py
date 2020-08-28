# -*- coding: utf-8 -*-
import pickle
import yaml
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MaxAbsScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

PROJECT_DIR = Path(__file__).resolve().parents[2]
PATH_RAW = 'data/raw'
PATH_INTERIM = 'data/interim'
PATH_PROCESSED = 'data/processed'
PARAMS = yaml.safe_load(open(PROJECT_DIR.joinpath('params.yaml')))['featurize']


class Doc2VecVectorizer():
    def __init__(self, *args, **kwargs):
        self.model = Doc2Vec(*args, **kwargs)
    
    def fit(self, X_text):
        X_tagged = [TaggedDocument(simple_preprocess(line), [i]) for i, line in enumerate(X_text)]
        self.model.build_vocab(X_tagged)
        self.model.train(X_tagged, total_examples=self.model.corpus_count, epochs=self.model.epochs)
    
    def transform(self, X_text):
        return np.array([self.model.infer_vector(simple_preprocess(line)) for line in X_text])
    
    def fit_transform(self, X_text):
        self.fit(X_text)
        return self.transform(X_text)


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
    elif method == 'doc2vec':
        vectorizer = Doc2VecVectorizer(vector_size=50, min_count=2, epochs=30, workers=8)
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


def main():
    with open(PROJECT_DIR.joinpath(PATH_INTERIM, 'sites_train_test.pkl'), 'rb') as fin:
        sites_train_test = pickle.load(fin)
    
    with open(PROJECT_DIR.joinpath(PATH_INTERIM, 'features_train_test.pkl'), 'rb') as fin:
        features_train_test = pickle.load(fin)
    
    with open(PROJECT_DIR.joinpath(PATH_INTERIM, 'train_size.pkl'), 'rb') as fin:
        train_size = pickle.load(fin)
    
    with open(PROJECT_DIR.joinpath(PATH_RAW, 'site_dic.pkl'), 'rb') as fin:
        site_dic = pickle.load(fin)
    
    
    feature_types = PARAMS['feature_types']
    vec_method = PARAMS['sites_vectorizer']['method']
    vec_params = {**PARAMS['sites_vectorizer']['params'], 'tokenizer': lambda s: s.split(), 'stop_words': ['unknown']}
    X_train_test_sparse = csr_hstack([sparsify_data(sites_train_test.values, vec_params, site_dic, train_size, method=vec_method), 
                                      #encode_sites_with_time_diffs(sites_train_test, timestamps_train_test), 
                                      #make_site_names(sites_train_test, site_dic, method='tfidf'), 
                                      prepare_features(features_train_test, feature_types)
                                     ])
    
    X_train_sparse = X_train_test_sparse[:train_size]
    X_test_sparse = X_train_test_sparse[train_size:]
    
    
    with open(PROJECT_DIR.joinpath(PATH_PROCESSED, 'X_train.pkl'), 'wb') as fout:
        pickle.dump(X_train_sparse, fout, protocol=2)
    
    with open(PROJECT_DIR.joinpath(PATH_PROCESSED, 'X_test.pkl'), 'wb') as fout:
        pickle.dump(X_test_sparse, fout, protocol=2)

if __name__ == '__main__':
    main()