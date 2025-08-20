
import shutil
import numpy as np
import time
from pathlib import Path
import copy
from collections import defaultdict
import os
from tqdm import tqdm

import torch
from torch import Tensor
from torch import optim
import torch.nn as nn

from utils.common.utils import ssim_loss


def validate(model, val_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for _, data in enumerate(val_loader):
            mask, kspace, target, _, fnames, slices = data
        
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            
            target_is_valid = torch.is_tensor(target) and target.numel() > 1
            if target_is_valid:
                target = target.cuda(non_blocking=True)
            
            output = model(kspace, mask, use_grad_ckpt=False)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                
                if target_is_valid:
                    targets[fnames[i]][int(slices[i])] = target[i].cpu().numpy()
                else:
                    targets[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, time.perf_counter() - start


def train_epoch(model, epoch, train_loader, optimizer, criterion, scaler, fold, args):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(train_loader)
    total_loss = 0.

    accumulation_step = args.accumulation_step
    optimizer.zero_grad()

    for iter, data in enumerate(train_loader):
        mask, kspace, target, maximum, fnames, slices = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            output = model(kspace, mask, use_grad_ckpt=True)
            data_range = maximum

            # Check if criterion supports index-based weighting
            if hasattr(criterion, 'forward') and 'slice_indices' in criterion.forward.__code__.co_varnames:
                # For index-based loss functions, we need to estimate num_slices
                # This is a simplified approach - in practice you might want to pass this as metadata
                estimated_num_slices = 30  # Typical number of slices for brain/knee MRI
                loss = criterion(output, target, data_range, slice_indices=slices, num_slices=estimated_num_slices)
            else:
                # Standard loss functions
                loss = criterion(output, target, data_range) 
            loss = loss / accumulation_step


        loss.backward()
        # scaler.scale(loss).backward()

        if (iter + 1) % accumulation_step == 0 or (iter + 1) == len_loader:
            optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_step
        if (iter) % args.report_interval == 0:
            log_template = f'Epoch = [{epoch:3d}/{args.start_epoch + args.num_epochs:3d}]  ' + \
                (f'Fold = [{fold:2d}/{args.num_folds:2d}]  ' if args.k_fold else '') + \
                f'Iter = [{iter:4d}/{len_loader:4d}]  ' + \
                f'Loss = {loss.item() * accumulation_step:.5f}  ' + \
                f'Time = {time.perf_counter() - start_iter:.4f}s'
            print(log_template)
            start_iter = time.perf_counter()

    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch