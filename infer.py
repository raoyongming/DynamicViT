# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from samplers import RASampler
from functools import partial


from models.dyvit import VisionTransformerDiffPruning
from models.dylvvit import LVViTDiffPruning
from models.dyconvnext import AdaConvNeXt
from models.dyswin import AdaSwinTransformer
import utils

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--model', default='deit_small', type=str, help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', type=utils.str2bool, default=True)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model_path', default='', help='resume from checkpoint')
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--base_rate', type=float, default=0.7)

    return parser


def main(args):

    cudnn.benchmark = True
    dataset_val, _ = build_dataset(is_train=False, args=args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    base_rate = args.base_rate
    KEEP_RATE1 = [base_rate, base_rate ** 2, base_rate ** 3]
    KEEP_RATE2 = [base_rate, base_rate - 0.2, base_rate - 0.4]

    print(f"Creating model: {args.model}")

    if args.model == 'deit-s':
        PRUNING_LOC = [3,6,9] 
        print('token_ratio =', KEEP_RATE1, 'at layer', PRUNING_LOC)
        model = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, 
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE1
            )
    elif args.model == 'deit-256':
        PRUNING_LOC = [3,6,9] 
        print('token_ratio =', KEEP_RATE1, 'at layer', PRUNING_LOC)
        model = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True, 
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE1
            )
    elif args.model == 'deit-b':
        PRUNING_LOC = [3,6,9] 
        print('token_ratio =', KEEP_RATE1, 'at layer', PRUNING_LOC)
        model = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, 
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE1
            )
    elif args.model == 'lvvit-s':
        PRUNING_LOC = [4,8,12] 
        print('token_ratio =', KEEP_RATE1, 'at layer', PRUNING_LOC)
        model = LVViTDiffPruning(
            patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
            p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE1
        )
    elif args.model == 'lvvit-m':
        PRUNING_LOC = [5,10,15] 
        print('token_ratio =', KEEP_RATE1, 'at layer', PRUNING_LOC)
        model = LVViTDiffPruning(
            patch_size=16, embed_dim=512, depth=20, num_heads=8, mlp_ratio=3.,
            p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE1
        )
    elif args.model == 'convnext-t':
        PRUNING_LOC = [1,2,3]
        print('token_ratio =', KEEP_RATE2, 'at layer', PRUNING_LOC)
        model = AdaConvNeXt(
            sparse_ratio=KEEP_RATE2, pruning_loc=PRUNING_LOC
        )
    elif args.model == 'convnext-s':
        PRUNING_LOC = [3,6,9]
        print('token_ratio =', KEEP_RATE2, 'at layer', PRUNING_LOC)
        model = AdaConvNeXt(
            sparse_ratio=KEEP_RATE2, pruning_loc=PRUNING_LOC, 
            depths=[3, 3, 27, 3]
        )
    elif args.model == 'convnext-b':
        PRUNING_LOC = [3,6,9]
        print('token_ratio =', KEEP_RATE2, 'at layer', PRUNING_LOC)
        model = AdaConvNeXt(
            sparse_ratio=KEEP_RATE2, pruning_loc=PRUNING_LOC, 
            depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]
        )
    elif args.model == 'swin-t':
        PRUNING_LOC = [1,2,3]
        print('token_ratio =', KEEP_RATE2, 'at layer', PRUNING_LOC)
        model = AdaSwinTransformer(
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_rate=0.0,
            pruning_loc=[1,2,3], sparse_ratio=KEEP_RATE2
        )
    elif args.model == 'swin-s':
        PRUNING_LOC = [2,4,6]
        print('token_ratio =', KEEP_RATE2, 'at layer', PRUNING_LOC)
        model = AdaSwinTransformer(
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_rate=0.0,
            drop_path_rate=args.drop_path,
            pruning_loc=[2,4,6], sparse_ratio=KEEP_RATE2
        )
    elif args.model == 'swin-b':
        PRUNING_LOC = [2,4,6]
        print('token_ratio =', KEEP_RATE2, 'at layer', PRUNING_LOC)
        model = AdaSwinTransformer(
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_rate=0.0,
            drop_path_rate=args.drop_path,
            pruning_loc=[2,4,6], sparse_ratio=KEEP_RATE2
        )
    else:
        raise NotImplementedError

    model_path = args.model_path
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    print('## model has been successfully loaded')

    model = model.cuda()

    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    validate(data_loader_val, model, criterion)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.eval()

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')


    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 20 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dynamic evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)