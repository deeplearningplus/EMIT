# EMIT: End-Motif Inspection via Transformer.

## Introduction
End-motif of plasma cell-free DNA (cfDNA) is a fragmentomic marker for cancer diagnosis. Here, we presented a self-supervised learning approach – end-motif inspection via transformer (EMIT) – that learns feature represenations of cfDNA end-motifs. We demonstrated that high classification performance in the identification of cancer via linear projection of features extracted from pretrained EMIT.   

## System requirements
- Operating systems: CentOS 7.
- [Python](https://docs.conda.io/en/latest/miniconda.html) (version == 3.7).
- [PyTorch](https://pytorch.org) (version == 1.13.1+cu116).
- [transformers](https://huggingface.co/docs/transformers/index) (version == 4.28.1).

This example was tested with the following environment. However, it should work on the other platforms. 

## Installation guide
- Following instruction from [miniconda](https://docs.conda.io/en/latest/miniconda.html) to install Python.
- Use the following command to install required packages.
```bash
# Install with GPU support. Check https://pytorch.org for more information. 
#+The following cmd install PyTorch compiled with cuda 118. 
pip install torch --index-url https://download.pytorch.org/whl/cu118

# If GPU not available, install the PyTorch compiled for CPU.
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install transformers, tokenizers and prettytable
pip install transformers==4.28.1 tokenizers==0.13.3 prettytable
```

- The installation process will take about an hour. This heavily depends on your network bandwidth.

## Demo
- Clone `EMIT` locally from Github.
```bash
git clone https://github.com/deeplearningplus/EMIT.git
```
- Instructions to run on data:
```bash
# Run on GPU
bash pretrain_gpu.sh

# Run on CPU
bash pretrain_cpu.sh
```

The pretrained model will be saved in `model-example` when the above command finishes running.
We uploaded a pretrained model in `model` for this tutorial.

- Linear projection from the pretrained model
```python
# Run on GPU
bash pretrain_gpu.sh

# Run on CPU
bash pretrain_cpu.sh
```

The outputs include log file `log.txt`, checkpoint of the linear classification at each epoch and prediction probabilities on the testing set.

## How to run on your own data
- Pretraining stage: prepare the pretraining data in the same format as `data/pretrained.trn.txt.gz` and run `pretrain.sh`.
- Linear projection stage: prepare the data in the same format as `data/targeted_BS_HCC_train_fold0.csv.gz` and run `linear_probe.sh`.




