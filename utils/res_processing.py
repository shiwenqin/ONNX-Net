import pandas as pd
import os
import argparse
import scipy.stats as stats


parser = argparse.ArgumentParser()
parser.add_argument('--res_path', type=str)
args = parser.parse_args()

df = pd.read_csv(os.path.join(args.res_path, 'preds.csv'))
output_json = os.path.join(args.res_path, 'results.json')

res = {}
datasets = df['dataset'].unique()
for dataset in datasets:
    subset = df[df['dataset'] == dataset]
    spearman = stats.spearmanr(subset['pred'], subset['true'])[0]
    kendall = stats.kendalltau(subset['pred'], subset['true'])[0]
    res[dataset] = {
        'spearman': spearman,
        'kendall': kendall
    }

with open(output_json, 'w') as f:
    import json
    json.dump(res, f, indent=4)