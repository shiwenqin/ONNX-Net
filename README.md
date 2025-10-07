# ONNX-Net: Towards Universal Representations and Instant Performance Prediction for Neural Architectures

This is the official codebase for our paper:

[ONNX-Net: Towards Universal Representations and Instant Performance Prediction for Neural Architectures](https://arxiv.org/abs/2510.04938)

Check out our [project page](https://shiwenqin.github.io/onnxnet/) and [dataset](https://huggingface.co/datasets/carlosqsw/ONNX-Bench) too!

## Environment Setup

```bash
pip install -r requirements.txt
```

## Important Folders

```
.
├── src/                             # Contains all main source code
│   ├── process/                     # Contains codes for text encoding generation
│   └── bert_inference.py            # Code for surrogate inference
│   └── bert_tuning.py               # Code for surrogate training
│   └── losses.py                    # Various loss functions tried
├── scripts/                         # Contains example scripts for running experiments
├── .gitignore         
├── requirements.txt
└── README.md          
```
