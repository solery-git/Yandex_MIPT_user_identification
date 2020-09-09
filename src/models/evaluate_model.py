# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import pickle
import yaml
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score
import eli5


PROJECT_DIR = Path(__file__).resolve().parents[2]
PATH_PROCESSED = 'data/processed'
PATH_MODELS = 'models'
PATH_SUBMISSIONS = 'kaggle_submissions'
PARAMS_ALL = yaml.safe_load(open(PROJECT_DIR.joinpath('params.yaml')))
SEED = PARAMS_ALL['meta']['seed']
PARAMS = PARAMS_ALL['evaluate']


def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    # turn predictions into data frame and save as csv file
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)

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
        y = pickle.load(fin)
    
    with open(PROJECT_DIR.joinpath(PATH_PROCESSED, 'data_feature_names.pkl'), 'rb') as fin:
        data_feature_names = pickle.load(fin)
    
    with open(PROJECT_DIR.joinpath(PATH_PROCESSED, 'fe_feature_names.pkl'), 'rb') as fin:
        fe_feature_names = pickle.load(fin)
    
    
    train_share = int(.7 * X_train_sparse.shape[0])
    X_train, y_train = X_train_sparse[:train_share, :], y[:train_share]
    X_holdout, y_holdout  = X_train_sparse[train_share:, :], y[train_share:]
    
    tss = TimeSeriesSplit(n_splits=10)
    
    metrics = {}
    
    logit = LogisticRegression(C=1, random_state=SEED, solver='liblinear')
    logit_train_scores = cross_val_score(logit, X_train, y_train, cv=tss, scoring='roc_auc', n_jobs=1)
    metrics['train_scores'] = {}
    for i, value in enumerate(logit_train_scores.tolist(), start=1):
        metrics['train_scores'][f'fold{i}'] = float(value)
    metrics['train_mean'] = float(np.mean(logit_train_scores))
    metrics['train_std'] = float(np.std(logit_train_scores))
    logit.fit(X_train, y_train)
    logit_holdout_score = roc_auc_score(y_holdout, logit.predict_proba(X_holdout)[:, 1])
    metrics['holdout'] = float(logit_holdout_score)
    show_feature_weights(logit, data_feature_names, fe_feature_names)
    
    
    if PARAMS['submission']['make']:
        logit.fit(X_train_sparse, y)
        logit_test_proba = logit.predict_proba(X_test_sparse)[:, 1]
        out_file = PARAMS['submission']['name'] + '.csv'
        write_to_submission_file(logit_test_proba, PROJECT_DIR.joinpath(PATH_SUBMISSIONS, out_file))
    
    
    with open(PROJECT_DIR.joinpath(PATH_MODELS, 'metrics.yaml'), 'w') as fout:
        yaml.dump(metrics, fout, sort_keys=False)
    

if __name__ == '__main__':
    main()