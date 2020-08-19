# -*- coding: utf-8 -*-
import pickle
import yaml
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit, learning_curve, cross_val_score
from sklearn.metrics import roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parents[2]
PATH_PROCESSED = 'data/processed'
SEED = yaml.safe_load(open(PROJECT_DIR.joinpath('model_params.yaml')))['meta']['seed']


def main():
    with open(PROJECT_DIR.joinpath(PATH_PROCESSED, 'X_train.pkl'), 'rb') as fin:
        X_train_sparse = pickle.load(fin)
    
    with open(PROJECT_DIR.joinpath(PATH_PROCESSED, 'X_test.pkl'), 'rb') as fin:
        X_test_sparse = pickle.load(fin)
    
    with open(PROJECT_DIR.joinpath(PATH_PROCESSED, 'y.pkl'), 'rb') as fin:
        y = pickle.load(fin)
    
    
    train_share = int(.7 * X_train_sparse.shape[0])
    X_train, y_train = X_train_sparse[:train_share, :], y[:train_share]
    X_valid, y_valid  = X_train_sparse[train_share:, :], y[train_share:]
    
    tss = TimeSeriesSplit(n_splits=10)
    
    logit = LogisticRegression(C=1, random_state=SEED, solver='liblinear')
    logit_train_scores = cross_val_score(logit, X_train_sparse, y, cv=tss, scoring='roc_auc', n_jobs=1)
    logit_train_stats = (logit_train_scores, np.mean(logit_train_scores), np.std(logit_train_scores))
    print('Scores on train: {}\nMean: {}, std: {}'.format(*logit_train_stats))
    logit.fit(X_train, y_train)
    logit_valid_score = roc_auc_score(y_valid, logit.predict_proba(X_valid)[:, 1])
    print('Score on valid: {}'.format(logit_valid_score))

if __name__ == '__main__':
    main()