#!/opt/software/install/miniconda38/bin/python
import h5py
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import pandas as pd
import json
import torch
import random
import glob
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, lr_scheduler
from transformers import (
  BertForSequenceClassification, 
  OPTForSequenceClassification, 
  PreTrainedTokenizerFast,
  BertModel,
  OPTModel,
  AutoConfig,
  BertConfig, 
  OPTConfig
)
import sys
from tqdm import tqdm
from sklearn.model_selection import KFold

seed = 123456
torch.set_num_threads(1)
random.seed(seed)

def mkdir(path):
    if os.path.exists(path):
        return None
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def initializeFeatureExtractor(featureExtractorPath):
    config = AutoConfig.from_pretrained(f"{featureExtractorPath}/config.json")
    if type(config) == BertConfig:
        featureExtractor = BertModel.from_pretrained(featureExtractorPath)
        index = 0
    elif type(config) == OPTConfig:
        featureExtractor = OPTModel.from_pretrained(featureExtractorPath)
        index = -1
    else:
        raise ValueError(f"Unsupported model {type(config)}")
    return featureExtractor, index

def encodeKmer(tokenizer, kmers, max_length):
  inputs = tokenizer(kmers, is_split_into_words=True, truncation=True, return_token_type_ids=False, 
                      max_length=max_length, padding="max_length", return_tensors='pt')
  return inputs

def train_one_epoch(epoch, featureExtractor, index, tokenizer, max_length, 
                    linearClassifier, optimizer, scheduler, criterion, 
                    trainDataset, kmerColumn, labelColumn, device, logStream):
    featureExtractor.eval()
    linearClassifier.train()
    total_loss = 0.0
    correct_k = 0.0
    trainDatasetRnd = trainDataset.sample(len(trainDataset))
    
    kmers = trainDatasetRnd[kmerColumn]
    labels = trainDatasetRnd[labelColumn]
    N = len(trainDatasetRnd)

    for i, label, kmer in zip(range(N), labels, kmers):
        a = kmer.split(',')
        x = encodeKmer(tokenizer, a, max_length)
        for k, v in x.items():x[k] = v.to(device)
        with torch.no_grad():
            z = featureExtractor(**x).last_hidden_state[:,index,:]

        logit = linearClassifier(z)
        label = torch.tensor(label).unsqueeze(0).to(device)
        loss = criterion(logit, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct_k += logit.argmax(1).eq(label).sum()

        if i % 1000 == 0 and i > 0:
            avg_loss = total_loss / (i + 1)
            acc = correct_k / (i + 1)
            print(f'Train, Epoch {epoch}, {i} / {N}, avg_loss: {avg_loss}, acc: {acc}, lr:{scheduler.get_last_lr()}', file=logStream)
            logStream.flush()

    trainAcc = correct_k / N
    trainLoss = total_loss / N

    return trainLoss, trainAcc

def evaluate(epoch, featureExtractor, index, tokenizer, max_length, 
             linearClassifier, criterion, testDataset, kmerColumn, labelColumn, device, logStream):
    correct_k = 0.0
    total_loss = 0.0
    featureExtractor.eval()
    linearClassifier.eval()
    probs = []

    labels = testDataset[labelColumn]
    kmers = testDataset[kmerColumn]
    N = len(testDataset)

    for i, label, kmer in zip(range(N), labels, kmers):
        a = kmer.split(',')
        x = encodeKmer(tokenizer, a, max_length)
        for k, v in x.items():x[k] = v.to(device)
        with torch.no_grad():
            z = featureExtractor(**x).last_hidden_state[:,index,:]
            logit = linearClassifier(z)

        label = torch.tensor(label).unsqueeze(0).to(device)
        loss = criterion(logit, label)
        total_loss += loss.item()
        correct_k += logit.argmax(1).eq(label).sum()

        probs.append(logit.softmax(1).flatten().tolist())

        if i % 1000 == 0 and i > 0:
            avg_loss = total_loss / (i + 1)
            acc = correct_k / (i + 1)
            print(f'Test, Epoch {epoch}, {i} / {N}, avg_loss: {avg_loss}, acc: {acc}', file=logStream)
            logStream.flush()

    valAcc = correct_k / N
    valLoss = total_loss / N

    return valLoss, valAcc, probs

def linearProjections(featureExtractorPath, trainDataset, testDataset, outdir, logStream, max_length=128, num_labels=2, 
                      epochs=8, device='cuda:0', lr=1e-4, wd=1e-2, kmerColumn='_4kmer', labelColumn='is_cancer'):

    tokenizer = PreTrainedTokenizerFast.from_pretrained(featureExtractorPath)
    featureExtractor, index = initializeFeatureExtractor(featureExtractorPath)
    featureExtractor = featureExtractor.to(device)
    featureExtractor.eval()

    ## Linear classifier
    hidden_size = featureExtractor.config.hidden_size
    linearClassifier = nn.Linear(hidden_size, num_labels, bias=False)
    linearClassifier = linearClassifier.to(device)

    print(featureExtractor, file=logStream)
    print(linearClassifier, file=logStream)

    ## Setup optimizer, scheduler and criterion
    optimizer = AdamW(linearClassifier.parameters(), lr=lr, weight_decay=wd)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    ## Train and evaluation code
    for epoch in range(epochs):
        trainLoss, trainAcc = train_one_epoch(epoch, 
                         featureExtractor, index, tokenizer, max_length, 
                         linearClassifier, optimizer, scheduler, criterion, 
                         trainDataset, kmerColumn, labelColumn, device, logStream)

        torch.save(linearClassifier, f'{outdir}/linearClassifier_e{epoch}.pt')
        torch.save(linearClassifier.state_dict(), f'{outdir}/linearClassifier_params_e{epoch}.pt')

        scheduler.step()

        valLoss, valAcc, probs = evaluate(epoch, 
                         featureExtractor, index, tokenizer, max_length, 
                         linearClassifier, criterion,
                         testDataset, kmerColumn, labelColumn, device, logStream)
        print(f'## Epoch {epoch}, trainLoss {trainLoss}, trainAcc {trainAcc}, valLoss {valLoss}, valAcc {valAcc}', file=logStream)

        probs_df = pd.DataFrame(probs)
        probs_df.columns = ['Control', 'Cancer']
        probs_df.to_csv(f'{outdir}/testingset_predictions_e{epoch}.csv.gz', index=False)
 

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="Linear probe", add_help=add_help)
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--epochs", default=6, type=int, metavar="N", help="number of total epochs to run (default: 6)")
    parser.add_argument("--num_labels", default=2, type=int, help="num_labels (default: 2)")
    parser.add_argument("--max_length", default=128, type=int, help="max_length (default: 128)")
    parser.add_argument("--pretrained_path", help="pretrained path")
    parser.add_argument("--lr", help="learning rate (default: 1e-4)", default=1e-4, type=float)
    parser.add_argument("--wd", help="weight decay (default: 1e-2)", default=1e-2, type=float)
    parser.add_argument("--train_file", help="trainingset file")
    parser.add_argument("--test_file", help="testingset file")
    parser.add_argument("--kmer_column", default='end_motif', help="kmer name in input file (default: end_motif)")
    parser.add_argument("--label_column", default='is_cancer', help="column name for class labels (default: is_cancer)")
    parser.add_argument("--outdir", default='results', help="output directory (default: results)")
    return parser

args = get_args_parser().parse_args()

mkdir(args.outdir)
logStream = open(f'{args.outdir}/log.txt', 'w')
print(args, file=logStream)

trainDataset = pd.read_csv(args.train_file)
testDataset = pd.read_csv(args.test_file)

linearProjections(
    featureExtractorPath=args.pretrained_path, 
    trainDataset=trainDataset, testDataset=testDataset, logStream=logStream,
    outdir=args.outdir, max_length=args.max_length, num_labels=args.num_labels, 
    epochs=args.epochs, device=args.device, lr=args.lr, wd=args.wd, 
    kmerColumn=args.kmer_column, labelColumn=args.label_column)

logStream.close()

