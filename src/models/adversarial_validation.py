# -*- coding: utf-8 -*-
import pickle
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as sparse_hstack, vstack as sparse_vstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parents[2]
PATH_PROCESSED = 'data/processed'
PATH_MODELS = 'models'
PATH_SUBMISSIONS = 'kaggle_submissions'
PARAMS_ALL = yaml.safe_load(open(PROJECT_DIR.joinpath('params.yaml')))
SEED = PARAMS_ALL['meta']['seed']
PARAMS = PARAMS_ALL['evaluate']


def csr_hstack(arglist):
    return csr_matrix(sparse_hstack(arglist))

def csr_vstack(arglist):
    return csr_matrix(sparse_vstack(arglist))


def main():
    with open(PROJECT_DIR.joinpath(PATH_PROCESSED, 'X_train.pkl'), 'rb') as fin:
        X_train_sparse = pickle.load(fin)
    
    with open(PROJECT_DIR.joinpath(PATH_PROCESSED, 'X_test.pkl'), 'rb') as fin:
        X_test_sparse = pickle.load(fin)
    
    with open(PROJECT_DIR.joinpath(PATH_PROCESSED, 'y.pkl'), 'rb') as fin:
        target = pickle.load(fin)
    
    
    train_len = X_train_sparse.shape[0]
    test_len = X_test_sparse.shape[0]
    y = np.array([0] * train_len + [1] * test_len)
    X = csr_vstack([X_train_sparse, X_test_sparse])
    
    logit = LogisticRegression(C=1, random_state=SEED, solver='liblinear')
    logit.fit(X, y)
    predictions_proba = logit.predict_proba(X)[:, 1]
    logit_score = roc_auc_score(y, predictions_proba)
    print('Score:', logit_score)
    print('Number of train examples:', X[y == 0].shape[0])
    validation_examples = X[(y == 0) & (predictions_proba > 0.5)]
    print('Number of train examples that look like test:', validation_examples.shape[0])
    validation_targets = target[predictions_proba[y == 0] > 0.5]
    class_0, class_1 = list(np.bincount(validation_targets))
    print(f'Class 0: {class_0}, class 1: {class_1}')
    print('Feature importances:', logit.coef_[0][-7:].tolist())
    print('Max feature importance:', np.max(np.abs(logit.coef_[0])))
    

if __name__ == '__main__':
    main()