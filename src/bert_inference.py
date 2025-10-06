import os
import json
import argparse

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from datasets import Dataset
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--val_size', type=int)
parser.add_argument('--prefix', type=str, default='')

args = parser.parse_args()

output_path = os.path.join(args.model_path, f'{args.prefix}pred')
if not os.path.exists(output_path):
    os.makedirs(output_path)
output_pred = os.path.join(output_path, 'predictions.csv')
output_json = os.path.join(output_path, 'results.json')

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_path,
    num_labels=1,
    problem_type="regression"
).to(device)
model.eval()

def tokenize_function(examples):
    return tokenizer(examples["onnx_encoding"], padding='longest', truncation=True)

#val_df = pd.read_csv(os.path.join(args.data_path, 'val.csv'))
val_df = pd.read_csv(args.data_path)
if args.val_size is not None and args.val_size < len(val_df):
    val_df = val_df.sample(n=args.val_size, random_state=args.seed)
val_df = val_df.reset_index(drop=True)
val_ds = Dataset.from_pandas(val_df).rename_column("accuracy", "labels")
val_ds = val_ds.map(tokenize_function, batched=True)

val_ds.set_format(
    type='torch', 
    columns=["input_ids", "attention_mask"]  # or omit "label" if not available
)
dataloader = torch.utils.data.DataLoader(val_ds, batch_size=8)

results = []
with torch.no_grad():
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # For regression, outputs.logits is shape (batch_size, 1)
        logits = outputs.logits.squeeze(-1)  # shape: (batch_size)
        predictions = logits.cpu().numpy().tolist()
        results.extend(predictions)

res = {}

mae = mean_absolute_error(val_df['accuracy'], results)
spearman = stats.spearmanr(val_df['accuracy'], results)[0]
kendall = stats.kendalltau(val_df['accuracy'], results)[0]

res['global'] = {
    'mae': mae,
    'spearman': spearman,
    'kendall': kendall
}

val_df['predictions'] = results
val_df = val_df[['accuracy', 'predictions', 'dataset']]

datasets = val_df['dataset'].unique()
for dataset in datasets:
    subset = val_df[val_df['dataset'] == dataset]
    mae = mean_absolute_error(subset['accuracy'], subset['predictions'])
    spearman = stats.spearmanr(subset['accuracy'], subset['predictions'])[0]
    kendall = stats.kendalltau(subset['accuracy'], subset['predictions'])[0]
    res[dataset] = {
        'mae': mae,
        'spearman': spearman,
        'kendall': kendall
    }

val_df.to_csv(output_pred, index=False)
with open(output_json, 'w') as f:
    json.dump(res, f, indent=4)

val_df = val_df.sort_values(by='accuracy')
fig = plt.figure(figsize=(16, 16))
sns.scatterplot(x='accuracy', y='predictions', data=val_df, hue='dataset', alpha=0.5)
plt.xlabel('True Accuracy')
plt.ylabel('Predicted Accuracy')
plt.title('True vs Predicted Accuracy')
plt.legend(title='Dataset')
plt.savefig(os.path.join(output_path, 'scatter_plot.png'))