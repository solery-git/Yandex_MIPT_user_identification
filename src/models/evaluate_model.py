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
PATH_MODELS = 'models'
SEED = yaml.safe_load(open(PROJECT_DIR.joinpath('params.yaml')))['meta']['seed']


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
    
    metrics = {}
    
    logit = LogisticRegression(C=1, random_state=SEED, solver='liblinear')
    logit_train_scores = cross_val_score(logit, X_train_sparse, y, cv=tss, scoring='roc_auc', n_jobs=1)
    metrics['train_scores'] = {}
    for i, value in enumerate(logit_train_scores.tolist(), start=1):
        metrics['train_scores'][f'fold{i}'] = float(value)
    metrics['train_mean'] = float(np.mean(logit_train_scores))
    metrics['train_std'] = float(np.std(logit_train_scores))
    logit.fit(X_train, y_train)
    logit_valid_score = roc_auc_score(y_valid, logit.predict_proba(X_valid)[:, 1])
    metrics['valid'] = float(logit_valid_score)
    
    with open(PROJECT_DIR.joinpath(PATH_MODELS, 'metrics.yaml'), 'w') as fout:
        yaml.dump(metrics, fout)
    

if __name__ == '__main__':
    main()