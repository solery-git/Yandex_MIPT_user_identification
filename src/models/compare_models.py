import sys
import subprocess
import yaml
from pathlib import Path
import pandas as pd
from scipy.stats import ttest_1samp

PROJECT_DIR = Path(__file__).resolve().parents[2]
PATH_MODELS = 'models'
PARAMS = yaml.safe_load(open(PROJECT_DIR.joinpath('params.yaml')))

metrics_diff_path = PROJECT_DIR.joinpath(PATH_MODELS, 'metrics_diff.tsv')

cmd_tokens = ['dvc', 'metrics', 'diff']
cmd_input = sys.argv[1:]
mode_cmd = (len(cmd_input) > 0)
if mode_cmd: #ignore params file in cmd mode
    cmd_tokens += cmd_input
else:
    cmd_tokens.append(PARAMS['compare']['with'])
with open(metrics_diff_path, 'w') as fout:
    fout.write(subprocess.run(cmd_tokens, capture_output=True, text=True).stdout)

try:
    diff_df = pd.read_csv(metrics_diff_path, usecols=['Metric', 'Change'], delim_whitespace=True)
    diff_df.set_index('Metric', inplace=True)
    diff_df = diff_df.loc[[f'train_scores.fold{i}' for i in range(2, 10+1)]] #remove fold1 bc it's not representative
    scores_diff = diff_df['Change'].astype(float).values
    tstat, pvalue = map(float, list(ttest_1samp(scores_diff, 0)))
    result = {'t-stat': tstat, 'p-value': pvalue}
    if mode_cmd:
        print(result)
    else:
        with open(PROJECT_DIR.joinpath(PATH_MODELS, 'metrics_ttest.yaml'), 'w') as fout:
            yaml.dump(result, fout)
except pd.errors.EmptyDataError:
    print('Models are identical')
    if not mode_cmd:
        with open(PROJECT_DIR.joinpath(PATH_MODELS, 'metrics_ttest.yaml'), 'w') as fout:
            pass