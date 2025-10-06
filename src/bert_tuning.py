import os
import json
import shutil
import argparse
import datetime

import torch
import wandb
import pandas as pd
import scipy.stats as stats
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import Dataset
from sklearn.metrics import mean_absolute_error

from losses import (
    HuberTrainer, 
    MSECapTrainer, 
    PWRTrainer, 
    PairwiseLogTrainer,
    SoftmaxListwiseTrainer,
    Poly1ListwiseTrainer,
    #SpearmanTrainer,
    PWRMiningTrainer,
    #PlackettTrainer,
    #ApproxRankTrainer,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

# Model params
parser.add_argument('--model_name', type=str, default='answerdotai/ModernBERT-base')

# Data params
parser.add_argument('--data_path', type=str)
parser.add_argument('--eval_path', type=str, default=None)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--train_task', type=str, default='none', choices=['einspace', 'hnasbench201', 'nasbench101', 'nas201nats', 'nasbench301', 'none'])
parser.add_argument('--eval_task', type=str, default='none', choices=['einspace', 'hnasbench201', 'nasbench101', 'nas201nats', 'nasbench301', 'none'])
parser.add_argument('--train_size', type=int, default=None)

# Train params
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--warmup_ratio', type=float, default=0.06)
parser.add_argument('--wandb_project', type=str, default='eintool_surrogate')
parser.add_argument('--loss_fn', type=str, default='mse', choices=['mse', 'huber', 'mse_cap', 'pwr', 'spearman', 'plackett', 'approxrank', 'pwr_mining', 'pairlog', 'softmaxlist', 'poly1list'])
parser.add_argument('--eval_strategy', type=str, default='epoch', choices=['steps', 'epoch'])
parser.add_argument('--flash_attention', type=bool, default=False)
parser.add_argument('--gradient_checkpointing', type=bool, default=False)


# Output params
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--save_pred', type=bool, default=True)

args = parser.parse_args()

data_name = args.data_path.split('/')[-1]
run_name = f'eval_task{args.eval_task}_train_task{args.train_task}_{args.loss_fn}_{data_name}_seed{args.seed}_lr{args.lr}_epochs{args.epochs}_frac42'

accelerator = Accelerator()
if accelerator.is_main_process:
    wandb.init(
        project = args.wandb_project,
        config = vars(args),
        name = run_name,
    )

# Reproducibility
set_seed(args.seed, deterministic=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if not os.path.exists(args.data_path):
    raise FileNotFoundError(f"Data path {args.data_path} does not exist")

if args.data_path.endswith('.csv'):
    train_df = pd.read_csv(args.data_path)
else:
    train_df = pd.read_csv(os.path.join(args.data_path, 'train.csv'))

if args.train_size is not None and args.train_size < len(train_df):
    train_df = train_df.sample(n=args.train_size, random_state=args.seed)
# stratify sample 10% training data
#train_df = train_df.sample(frac=0.1, random_state=42)
if args.eval_path is not None:
    val_df = pd.read_csv(args.eval_path)
else:
    val_df = pd.read_csv(os.path.join(args.data_path, 'val.csv'))

if args.train_task != 'none' and args.eval_task != 'none':
    raise ValueError("Train task and eval task cannot be set at the same time")

if args.eval_task != 'none':
    train_df = train_df[~(train_df['dataset'] == args.eval_task)]
if args.train_task != 'none':
    train_df = train_df[train_df['dataset'] == args.train_task]

train_df = train_df[['onnx_encoding', 'accuracy', 'dataset']].reset_index(drop=True)
val_df = val_df[['onnx_encoding', 'accuracy', 'dataset']].reset_index(drop=True)

train_ds = Dataset.from_pandas(train_df).rename_column("accuracy", "labels")
val_ds = Dataset.from_pandas(val_df).rename_column("accuracy", "labels")
# only keep the onnx_encoding and labels columns
train_ds = train_ds.remove_columns([col for col in train_ds.column_names if col not in ['onnx_encoding', 'labels']])
val_ds = val_ds.remove_columns([col for col in val_ds.column_names if col not in ['onnx_encoding', 'labels']])
print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.padding_side = 'left' if args.model_name in ['Qwen/Qwen3-0.6B'] else 'right'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(examples):
    return tokenizer(examples["onnx_encoding"], padding='longest', truncation=True)
train_ds = train_ds.map(tokenize_function, batched=True)
val_ds = val_ds.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=1,
    problem_type="regression",
    attn_implementation="flash_attention_2" if args.flash_attention else "sdpa",
).to(device)
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.eos_token_id

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    predictions = predictions.flatten()
    mae = mean_absolute_error(labels, predictions)
    spearman_corr = stats.spearmanr(labels, predictions)[0]
    kendall_corr = stats.kendalltau(labels, predictions)[0]
    return {"mae": mae, "spearman_corr": spearman_corr, "kendall_corr": kendall_corr}

# Train Setup
date = datetime.datetime.now().strftime("%Y-%m-%d")
output_dir = os.path.join(args.output_path, date, run_name)
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'info.json'), 'w') as f:
    json.dump(vars(args), f, indent=4)

training_args = TrainingArguments(
    seed=args.seed,
    output_dir=output_dir,
    eval_strategy=args.eval_strategy,
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    warmup_ratio=args.warmup_ratio,
    logging_steps=200,
    lr_scheduler_type="polynomial",
    lr_scheduler_kwargs={"lr_end": args.lr * 0.1},
    gradient_accumulation_steps=1,
    push_to_hub=False,
    report_to="wandb",
    save_strategy=args.eval_strategy,
    run_name=run_name,
    load_best_model_at_end=True,
    save_steps=800 if args.eval_strategy == 'steps' else None,
    metric_for_best_model="kendall_corr",
    greater_is_better=True,
    bf16_full_eval=True,
    gradient_checkpointing=args.gradient_checkpointing,
)

universal_trainer_params = {
    "model": model,
    "args": training_args,
    "train_dataset": train_ds,
    "eval_dataset": val_ds,
    "processing_class": tokenizer,
    "compute_metrics": compute_metrics,
}
if args.loss_fn == 'mse':
    trainer = Trainer(**universal_trainer_params)
elif args.loss_fn == 'huber':
    trainer = HuberTrainer(**universal_trainer_params, delta=5)
elif args.loss_fn == 'mse_cap':
    trainer = MSECapTrainer(**universal_trainer_params, cap=10)
elif args.loss_fn == 'pwr':
    trainer = PWRTrainer(**universal_trainer_params, compare_threshold=0.0, max_compare_ratio=4, margin=0.1)
# elif args.loss_fn == 'spearman':
#     trainer = SpearmanTrainer(**universal_trainer_params, tau=1.0)
elif args.loss_fn == 'pwr_mining':
    trainer = PWRMiningTrainer(**universal_trainer_params, mining_mode='topk')
elif args.loss_fn == 'pairlog':
    trainer = PairwiseLogTrainer(**universal_trainer_params)
elif args.loss_fn == 'softmaxlist':
    trainer = SoftmaxListwiseTrainer(**universal_trainer_params)
elif args.loss_fn == 'poly1list':
    trainer = Poly1ListwiseTrainer(**universal_trainer_params)
# elif args.loss_fn == 'plackett':
#     trainer = PlackettTrainer(**universal_trainer_params)
# elif args.loss_fn == 'approxrank':
#     trainer = ApproxRankTrainer(**universal_trainer_params, alpha=10.0)

trainer.train()

# Pred & Save
if args.save_pred:
    preds = trainer.predict(val_ds)
    preds = pd.DataFrame(preds.predictions.flatten(), columns=['pred'])
    preds['true'] = val_df["accuracy"]
    preds['dataset'] = val_df['dataset']
    preds.to_csv(os.path.join(output_dir, 'preds.csv'), index=False)