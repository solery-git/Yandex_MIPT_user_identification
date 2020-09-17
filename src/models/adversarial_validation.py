# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import pickle
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as sparse_hstack, vstack as sparse_vstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import eli5


PROJECT_DIR = Path(__file__).resolve().parents[2]
PATH_PROCESSED = 'data/processed'
PATH_MODELS = 'models'
PARAMS_ALL = yaml.safe_load(open(PROJECT_DIR.joinpath('params.yaml')))
SEED = PARAMS_ALL['meta']['seed']


def csr_hstack(arglist):
    return csr_matrix(sparse_hstack(arglist))

def csr_vstack(arglist):
    return csr_matrix(sparse_vstack(arglist))

def show_feature_weights(estimator, data_feature_names, fe_feature_names):
    feature_names = data_feature_names + fe_feature_names
    # top 30 data features
    data_feature_names_set = set(data_feature_names)
    data_explanation = eli5.explain_weights(estimator, feature_names=feature_names, top=30, feature_filter=lambda name: name in data_feature_names_set)
    print(eli5.format_as_text(data_explanation, highlight_spaces=True))
    # features from feature engineering
    fe_feature_names_set = set(fe_feature_names)
    fe_explanation = eli5.explain_weights(estimator, feature_names=feature_names, feature_filter=lambda name: name in fe_feature_names_set)
    print(eli5.format_as_text(fe_explanation, show=['targets']))


def main():
    with open(PROJECT_DIR.joinpath(PATH_PROCESSED, 'X_train.pkl'), 'rb') as fin:
        X_train_sparse = pickle.load(fin)
    
    with open(PROJECT_DIR.joinpath(PATH_PROCESSED, 'X_test.pkl'), 'rb') as fin:
        X_test_sparse = pickle.load(fin)
    
    with open(PROJECT_DIR.joinpath(PATH_PROCESSED, 'y.pkl'), 'rb') as fin:
        target = pickle.load(fin)
    
    with open(PROJECT_DIR.joinpath(PATH_PROCESSED, 'data_feature_names.pkl'), 'rb') as fin:
        data_feature_names = pickle.load(fin)
    
    with open(PROJECT_DIR.joinpath(PATH_PROCESSED, 'fe_feature_names.pkl'), 'rb') as fin:
        fe_feature_names = pickle.load(fin)
    
    
    train_len = X_train_sparse.shape[0]
    test_len = X_test_sparse.shape[0]
    y = np.array([0] * train_len + [1] * test_len)
    X = csr_vstack([X_train_sparse, X_test_sparse])
    
    logit = LogisticRegression(C=1, random_state=SEED, solver='liblinear')
    logit.fit(X, y)
    predictions_proba = logit.predict_proba(X)[:, 1]
    logit_score = roc_auc_score(y, predictions_proba)
    print('Score:', logit_score)
    print('Number of train examples:', X_train_sparse.shape[0])
    
    adv_valid_mask = (predictions_proba > 0.5)[:train_len]
    validation_examples = X_train_sparse[adv_valid_mask]
    print('Number of train examples that look like test:', validation_examples.shape[0])
    
    validation_targets = target[predictions_proba[:train_len] > 0.5]
    class_0, class_1 = list(np.bincount(validation_targets))
    print(f'Class 0: {class_0}, class 1: {class_1}')
    
    show_feature_weights(logit, data_feature_names, fe_feature_names)
    
    
    with open(PROJECT_DIR.joinpath(PATH_PROCESSED, 'adv_valid_mask.pkl'), 'wb') as fout:
        pickle.dump(adv_valid_mask, fout, protocol=2)
    

if __name__ == '__main__':
    main()