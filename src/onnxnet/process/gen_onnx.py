from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer

from onnxnet.process.utils import ONNXConverter, list_onnx_files


SS = [
    "einspace_augmentation",
    # "einspace",
    "hnasbench201",
    "nasbench101",
    "nasbench201",
    "nasbench301",
    "natsbench",
    "transnasbench101",
]

path = Path(__file__).parents[3] / "data"
encoding_path = Path(__file__).parents[3] / "data"
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

for ss in SS:
    # for dataset in datasets:
    ss_path = path / f"{ss}_simplify" if not ss.startswith("einspace") else path / f"{ss}_simplify" / "cifar10"
    encoding_path_ss = encoding_path / "chain_slim_v1"
    if not encoding_path_ss.exists():
        encoding_path_ss.mkdir(parents=True)
    if not ss_path.exists():
        msg = f"Path {ss_path} does not exist."
        raise FileNotFoundError(msg)

    data = {
        "onnx_encoding": [],
        "accuracy": [],
        "onnx_encoding_tokens": [],
        "dataset": [],
        "name": [],
    }

    print(f"Processing {ss} dataset...")
    for onnx_file in tqdm(list_onnx_files(ss_path), desc=f"Processing ONNX files in {ss}"):
        if ss == "nasbench301" and onnx_file.name == "1669.onnx":
            continue
        onnx_path = ss_path / onnx_file
        seed = onnx_file.name.split("/")[-2] if ss == "einspace" else ""
        onnx_name = onnx_file.name.split("/")[-1]
        converter = ONNXConverter(onnx_path, tokenizer)
        try:
            model_str, acc, token_count = converter.get_onnx_str(mode="chain_slim")
        except Exception as e:
            msg = f"Error processing {onnx_file}: {e}"
            raise RuntimeError(msg) from e
            # continue

        data["onnx_encoding"].append(model_str)
        data["accuracy"].append(acc)
        data["onnx_encoding_tokens"].append(token_count)
        data["dataset"].append("cifar10")
        if ss == "einspace":
            data["name"].append(f"{seed}_{onnx_name.split('.')[0]}")
        else:
            data["name"].append(onnx_name.split(".")[0])

    df = pd.DataFrame(data)
    df.to_csv(encoding_path_ss / f"{ss}.csv", index=False)
