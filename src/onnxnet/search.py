import argparse
import datetime
import json
from pathlib import Path
import random
from typing import Any, cast

from datasets import Dataset
from losses import (
    HuberTrainer,
    MSECapTrainer,
    PairwiseLogTrainer,
    Poly1ListwiseTrainer,
    # SpearmanTrainer,
    PWRMiningTrainer,
    # PlackettTrainer,
    # ApproxRankTrainer,
    PWRTrainer,
    SoftmaxListwiseTrainer,
)
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error
import torch
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.trainer_utils import set_seed
from transformers.training_args import TrainingArguments

import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

# Model params
parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")

# Data params
parser.add_argument("--data_path", type=Path)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--train_task",
    type=str,
    default="none",
    choices=["einspace", "hnasbench201", "nasbench101", "nas201nats", "nasbench301", "none"],
)
parser.add_argument("--train_size", type=int, default=None)

# Train params
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--warmup_ratio", type=float, default=0.06)
parser.add_argument("--wandb_project", type=str, default="eintool_surrogate")
parser.add_argument(
    "--loss_fn",
    type=str,
    default="mse",
    choices=[
        "mse",
        "huber",
        "mse_cap",
        "pwr",
        "spearman",
        "plackett",
        "approxrank",
        "pwr_mining",
        "pairlog",
        "softmaxlist",
        "poly1list",
    ],
)
parser.add_argument("--flash_attention", type=bool, default=False)
parser.add_argument("--gradient_checkpointing", type=bool, default=False)


# Output params
parser.add_argument("--output_path", type=Path, default=None)
parser.add_argument("--save", type=bool, default=True)
parser.add_argument("--save_pred", type=bool, default=True)

parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--start_indices", type=lambda x: set(map(int, x.split(","))), default=set())
parser.add_argument("--start_iteration", type=int, default=0)

args = parser.parse_args()

data_name = args.data_path.name
run_name = f"train_task{args.train_task}_{args.loss_fn}_{data_name}\
             _seed{args.seed}_lr{args.lr}_frac42"

# accelerator = Accelerator()
# if accelerator.is_main_process:
wandb.init(
    project=args.wandb_project,
    config=vars(args),
    name=run_name,
)

# Reproducibility
set_seed(args.seed, deterministic=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if not args.data_path.exists():
    msg = f"Data path {args.data_path} does not exist"
    raise FileNotFoundError(msg)

if args.data_path.name.endswith(".csv"):
    dataset = pd.read_csv(args.data_path)
else:
    dataset = pd.read_csv(args.data_path / "train.csv")

if args.train_size is not None and args.train_size < len(dataset):
    dataset = dataset.sample(n=args.train_size, random_state=args.seed)
# stratify sample 10% training data
# dataset = dataset.sample(frac=0.1, random_state=42)

dataset = dataset[["onnx_encoding", "accuracy", "dataset"]].reset_index(drop=True)

dataset = Dataset.from_pandas(dataset).rename_column("accuracy", "labels")
# only keep the onnx_encoding and labels columns
dataset = dataset.remove_columns([col for col in dataset.column_names if col not in {"onnx_encoding", "labels"}])
print(f"Train size: {len(dataset)}")

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.padding_side = "left" if args.model_name == "Qwen/Qwen3-0.6B" else "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
    return tokenizer(examples["onnx_encoding"], padding="longest", truncation=True)


dataset = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name if not args.checkpoint else args.checkpoint,
    num_labels=1,
    problem_type="regression",
    attn_implementation="flash_attention_2" if args.flash_attention else "sdpa",
).to(device)
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.eos_token_id


def compute_metrics(eval_preds: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]) -> dict[str, float]:
    predictions, labels = eval_preds
    predictions = predictions.flatten()
    mae = mean_absolute_error(labels, predictions)
    spearman_corr = cast("float", stats.spearmanr(labels, predictions)[0])
    kendall_corr = cast("float", stats.kendalltau(labels, predictions)[0])
    return {"mae": mae, "spearman_corr": spearman_corr, "kendall_corr": kendall_corr}


# Train Setup
date = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")
output_dir = args.output_path / date / run_name
output_dir.mkdir(exist_ok=True, parents=True)
with output_dir.joinpath("info.json").open("w", encoding="utf-8") as f:
    json.dump(vars(args), f, indent=4, default=str)

training_args = TrainingArguments(
    seed=args.seed,
    output_dir=output_dir,
    eval_strategy="no",
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=1,
    weight_decay=args.weight_decay,
    warmup_ratio=args.warmup_ratio,
    logging_steps=200,
    lr_scheduler_type="cosine",  # "polynomial",
    #  lr_scheduler_kwargs={"lr_end": args.lr * 0.1},
    gradient_accumulation_steps=1,
    push_to_hub=False,
    report_to="wandb",
    save_strategy="no",
    run_name=run_name,
    save_steps=None,  # pyright: ignore [reportArgumentType]
    metric_for_best_model="kendall_corr",
    greater_is_better=True,
    bf16_full_eval=True,
    gradient_checkpointing=args.gradient_checkpointing,
)

universal_trainer_params = {
    "model": model,
    "args": training_args,
    "train_dataset": dataset.select([0], keep_in_memory=True),
    "processing_class": tokenizer,
    "compute_metrics": compute_metrics,
}
if args.loss_fn == "mse":
    trainer = Trainer(**universal_trainer_params)
elif args.loss_fn == "huber":
    trainer = HuberTrainer(**universal_trainer_params, delta=5)
elif args.loss_fn == "mse_cap":
    trainer = MSECapTrainer(**universal_trainer_params, cap=10)
elif args.loss_fn == "pwr":
    trainer = PWRTrainer(**universal_trainer_params, compare_threshold=0.0, max_compare_ratio=4, margin=0.1)
# elif args.loss_fn == 'spearman':
#     trainer = SpearmanTrainer(**universal_trainer_params, tau=1.0)
elif args.loss_fn == "pwr_mining":
    trainer = PWRMiningTrainer(**universal_trainer_params, mining_mode="topk")
elif args.loss_fn == "pairlog":
    trainer = PairwiseLogTrainer(**universal_trainer_params)
elif args.loss_fn == "softmaxlist":
    trainer = SoftmaxListwiseTrainer(**universal_trainer_params)
elif args.loss_fn == "poly1list":
    trainer = Poly1ListwiseTrainer(**universal_trainer_params)
# elif args.loss_fn == 'plackett':
#     trainer = PlackettTrainer(**universal_trainer_params)
# elif args.loss_fn == 'approxrank':
#     trainer = ApproxRankTrainer(**universal_trainer_params, alpha=10.0)
else:
    msg = f"Loss function {args.loss_fn} not recognized"
    raise ValueError(msg)

trainer.train()
print("Initial training done")

for _ in range(1):
    curr_indices = args.start_indices
    for i, (new_best, new_rand) in enumerate(
        [
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
            (4, 4),
            (2, 2),
            (2, 2),
            (8, 8),
            (8, 8),
            (16, 16),
            (16, 16),
        ][args.start_iteration :],
        args.start_iteration,
    ):
        print("Predicting for iteration", i + 1)
        preds = cast(
            "npt.NDArray[np.float64]",
            trainer.predict(cast("torch.utils.data.Dataset[dict[str, Any]]", dataset)).predictions,
        ).flatten()
        print("Prediction done")
        indices = [x[0] for x in sorted(enumerate(preds), key=lambda x: x[1], reverse=True)]
        curr_indices |= set([i for i in indices if i not in curr_indices][:new_best])
        frac = set(indices[: max(512, len(indices) // 2**i)]) - curr_indices
        curr_indices |= set(random.sample(list(frac), min(len(frac), new_rand)))
        print("Starting training for iteration", i + 1)
        # trainer.train_dataset = dataset.select(list(curr_indices), keep_in_memory=True)
        universal_trainer_params = {
            "model": model,
            "args": training_args,
            "train_dataset": dataset.select(list(curr_indices), keep_in_memory=True),
            "processing_class": tokenizer,
            "compute_metrics": compute_metrics,
        }
        if args.loss_fn == "mse":
            trainer = Trainer(**universal_trainer_params)
        elif args.loss_fn == "huber":
            trainer = HuberTrainer(**universal_trainer_params, delta=5)
        elif args.loss_fn == "mse_cap":
            trainer = MSECapTrainer(**universal_trainer_params, cap=10)
        elif args.loss_fn == "pwr":
            trainer = PWRTrainer(**universal_trainer_params, compare_threshold=0.0, max_compare_ratio=4, margin=0.1)
        elif args.loss_fn == "pwr_mining":
            trainer = PWRMiningTrainer(**universal_trainer_params, mining_mode="topk")
        elif args.loss_fn == "pairlog":
            trainer = PairwiseLogTrainer(**universal_trainer_params)
        elif args.loss_fn == "softmaxlist":
            trainer = SoftmaxListwiseTrainer(**universal_trainer_params)
        elif args.loss_fn == "poly1list":
            trainer = Poly1ListwiseTrainer(**universal_trainer_params)
        else:
            msg = f"Loss function {args.loss_fn} not recognized"
            raise ValueError(msg)
        trainer.train()
        model.save_pretrained(output_dir.joinpath("model"))
        print("Training done")
        accs = trainer.train_dataset["labels"]
        print(
            f"Num Samples: {len(curr_indices)} | Best Accuracy: {max(accs):.4f} | Median Accuracy: {np.median(accs):.4f} | Mean Accuracy: {np.mean(accs):.4f}",
        )
        print(curr_indices)

        # /home/pkw63980/ONNX-Net/.venv/bin/python /home/pkw63980/ONNX-Net/src/onnxnet/search.py --model_name answerdotai/ModernBERT-large --data_path data/chain_slim_v1/onnxbench.csv --output_path out --batch_size 16 --seed 42 --lr 5e-5 --loss_fn pwr --weight_decay 0.1 --gradient_checkpointing True
