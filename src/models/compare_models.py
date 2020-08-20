import sys
import subprocess
from pathlib import Path
import pandas as pd
from scipy.stats import ttest_1samp

PROJECT_DIR = Path(__file__).resolve().parents[2]
PATH_MODELS = 'models'


metrics_diff_path = PROJECT_DIR.joinpath(PATH_MODELS, 'metrics_diff.tsv')

cmd_tokens = ['dvc', 'metrics', 'diff']
cmd_tokens += sys.argv[1:]
with open(metrics_diff_path, 'w') as fout:
    fout.write(subprocess.run(cmd_tokens, capture_output=True, text=True).stdout)

diff_df = pd.read_csv(metrics_diff_path, usecols=['Metric', 'Change'], delim_whitespace=True)
diff_df.set_index('Metric', inplace=True)
diff_df = diff_df.loc[[f'train_scores.fold{i}' for i in range(1, 10+1)]]
scores_diff = diff_df['Change'].values
print(ttest_1samp(scores_diff, 0))