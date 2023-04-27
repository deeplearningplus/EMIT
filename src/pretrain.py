import datetime
import math
import os
import sys
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import time
import warnings

import torch
import torch.utils.data
import utils
from torch import nn
from torch.utils.data.dataloader import default_collate

from transformers import (
    AutoConfig, AutoModelForCausalLM,
    OPTForCausalLM, OPTConfig, 
    AutoTokenizer, PreTrainedTokenizerFast,
    GPT2LMHeadModel, GPT2Config,
    BertConfig, BertForMaskedLM, BertTokenizer,
    get_scheduler,
    default_data_collator,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq
)
import gzip
import random
from count_parameters import *

class EndMotifDataset(torch.utils.data.Dataset):
    def __init__(self, text_file, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lines = []

        if text_file.endswith('.gz'):
            ## open gz file in text mode, therefore, no need to apply s.decode()
            with gzip.open(text_file, 'rt') as f:
                self.lines = [s.strip().split()[0:self.max_length] for s in f]
        else:
            with open(text_file) as f:
                self.lines = [s.strip().split()[0:self.max_length] for s in f]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        tokens = self.lines[i]
        return self.tokenizer(
                    tokens, is_split_into_words=True, return_token_type_ids=False, 
                    truncation=True, max_length=self.max_length, padding="max_length"
                )

def train_one_epoch(model, lr_scheduler, optimizer, data_loader, device, epoch, args):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        for k, v in batch.items():batch[k] = v.to(device, non_blocking=True)
        output = model(**batch)
        loss = output.loss

        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        elif args.clip_value is not None:
            nn.utils.clip_grad_value_(model.parameters(), clip_value=args.clip_value)
        optimizer.step()
        lr_scheduler.step()

        batch_size = batch['input_ids'].shape[0]

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['perplexity'].update(loss.exp().item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        sys.stdout.flush()

def evaluate(model, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for batch in metric_logger.log_every(data_loader, print_freq, header):
            for k, v in batch.items():batch[k] = v.to(device, non_blocking=True)
            output = model(**batch)
            loss = output.loss

            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = batch['input_ids'].shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['perplexity'].update(loss.exp().item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    #print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    print(f"{header} Perplexity {metric_logger.perplexity.global_avg:.3f}")
    return metric_logger.perplexity.global_avg

def load_data(args):
    # Data loading code
    print("Loading data")
    st = time.time()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)

    dataset = EndMotifDataset(args.train_file, tokenizer, args.max_length)
    dataset_test = EndMotifDataset(args.val_file, tokenizer, args.max_length)
    print("Took", time.time() - st)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler, tokenizer


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    if args.json_file is not None:
        config = AutoConfig.from_pretrained(args.json_file)
        assert type(config) == BertConfig

    dataset, dataset_test, train_sampler, test_sampler, tokenizer = load_data(args)
    tokenizer.save_pretrained(args.output_dir)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    model = BertForMaskedLM(config)

    if config.vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer)) # if necessary
    
    if args.initial_weight:
        state_dict = torch.load(args.initial_weight, map_location="cpu")
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

    model.to(device)
    print(model.config)
    print(count_parameters(model))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=data_collator, drop_last=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, 
        num_workers=args.workers, pin_memory=True, collate_fn=data_collator
    )

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.95))

    ##lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    num_update_steps_per_epoch = math.ceil(len(data_loader))
    max_train_steps = args.epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type, optimizer=optimizer, 
        num_warmup_steps=len(data_loader)*args.warmup_epochs, 
        num_training_steps=max_train_steps
    )

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        evaluate(model, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, lr_scheduler, optimizer, data_loader, device, epoch, args)
        evaluate(model, data_loader_test, device=device)

        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }

            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

            if utils.is_main_process():
                model_without_ddp.save_pretrained(args.output_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Pretraining for end-motifs", add_help=add_help)

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=40, type=int, metavar="N", help="number of total epochs to run (default: 40)")
    parser.add_argument("--warmup_epochs", default=3, type=int, help="number of warmup epochs to (default: 3)")
    parser.add_argument(
        "-j", "--workers", default=1, type=int, metavar="N", help="number of data loading workers (default: 1)"
    )
    parser.add_argument("--lr", default=1e-4, type=float, help="initial learning rate (default: 1e-4)")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0.0,
        type=float,
        metavar="W",
        help="weight decay (default: 0.0)",
        dest="weight_decay",
    )
    parser.add_argument("--lr-scheduler-type", default="cosine", 
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        type=str, help="the lr scheduler (default: cosine)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--save-freq", default=-1, type=int, help="save frequency")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--initial_weight", default="", type=str, help="path of the inital weight")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training (e.g. 'tcp://localhost:10001')")
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--clip-value", default=None, type=float, help="the maximum value to clip (default None)")

    parser.add_argument("--max-length", default=64, type=int, help="max length (default: 64)")
    parser.add_argument("--tokenizer-dir", default=None, type=str, help="tokenizer directory or the name of transformers pretrained tokenizer")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--json-file", default=None, type=str, help="model json file. make sure that special tokens in this file is consistent with tokenizer")
    parser.add_argument("--train-file", default=None, type=str, help="training file")
    parser.add_argument("--val-file", default=None, type=str, help="validation file")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
