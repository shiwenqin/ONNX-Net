import os
from collections import defaultdict

import onnx
import pandas as pd
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from utils import ONNXConverter, list_onnx_files

SS = ['nasbench201']
#datasets = ['addnist', 'chesseract', 'cifartile', 'geoclassing', 'gutenberg', 'isabella', 'language', 'multnist']

path = '../eintool/onnx'
encoding_path = '../eintool/encodings'
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

for ss in SS:
    #for dataset in datasets:
    ss_path = os.path.join(path, f"{ss}_simplify") if ss != 'einspace' else os.path.join(path, f"{ss}_simplify", "cifar10")
    encoding_path_ss = os.path.join(encoding_path, "chain_slim_input")
    if not os.path.exists(encoding_path_ss):
        os.makedirs(encoding_path_ss)
    if not os.path.exists(ss_path):
        raise FileNotFoundError(f"Path {ss_path} does not exist.")
    
    data = {
        'onnx_encoding': [],
        'accuracy': [],
        "onnx_encoding_tokens": [],
        'dataset': [],
        'name': [],
    }

    print(f"Processing {ss} dataset...")
    for onnx_file in tqdm(list_onnx_files(ss_path), desc=f"Processing ONNX files in {ss}"):
        
        onnx_path = os.path.join(ss_path, onnx_file)
        seed = onnx_file.split('/')[-2] if ss == 'einspace' else ''
        onnx_name = onnx_file.split('/')[-1]
        converter = ONNXConverter(onnx_path, tokenizer)
        try:
            model_str, acc, token_count = converter.get_onnx_str(mode = 'chain_slim_input')
        except Exception as e:
            print(f"Error processing {onnx_file}: {e}")
            #continue
        
        data['onnx_encoding'].append(model_str)
        data['accuracy'].append(acc)
        data['onnx_encoding_tokens'].append(token_count)
        data['dataset'].append('cifar10')
        if ss == 'einspace':
            data['name'].append(f'{seed}_{onnx_name.split(".")[0]}')
        else:
            data['name'].append(onnx_name.split('.')[0])
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(encoding_path_ss, f"{ss}.csv"), index=False)
