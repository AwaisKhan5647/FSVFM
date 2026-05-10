# -*- coding: utf-8 -*-
# Author: Gaojian Wang@ZJUICSR
# --------------------------------------------------------
# This source code is licensed under the Attribution-NonCommercial 4.0 International License.
# You can find the license in the LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
import time
import datetime
from typing import Iterable, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy
from tqdm import tqdm

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import util.misc as misc
import util.lr_sched as lr_sched
from util.metrics import *


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    total_epochs = getattr(args, 'epochs', '?')
    is_main = misc.is_main_process()

    # tqdm bar on rank 0 only — one line per batch, updates in-place
    pbar = tqdm(
        total=len(data_loader),
        desc=f"Epoch {epoch+1}/{total_epochs} [train]",
        unit="batch",
        dynamic_ncols=True,
        disable=not is_main,
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    epoch_start = time.time()

    for data_iter_step, (samples, targets) in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples).to(device, non_blocking=True)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        max_lr = max(g["lr"] for g in optimizer.param_groups)
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

        # Update tqdm every batch with current loss and lr
        if is_main:
            pbar.set_postfix(loss=f"{loss_value:.4f}", lr=f"{max_lr:.2e}", refresh=False)
            pbar.update(1)

    pbar.close()

    epoch_secs = time.time() - epoch_start
    metric_logger.synchronize_between_processes()
    if is_main:
        print(f"  Epoch {epoch+1}/{total_epochs} train done — "
              f"avg_loss={metric_logger.meters['loss'].global_avg:.4f}  "
              f"time={str(datetime.timedelta(seconds=int(epoch_secs)))}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    """Alias kept for backward compatibility — delegates to evaluate_full."""
    return evaluate_full(data_loader, model, device)


@torch.no_grad()
def evaluate_full(data_loader, model, device):
    """
    Collect ALL predictions across the full dataset (gathering across DDP ranks),
    then compute global Accuracy, AUC, Real_Accuracy, Fake_Accuracy.
    Returns a dict with keys: accuracy, auc, real_accuracy, fake_accuracy, loss.
    """
    import torch.distributed as dist_mod

    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    local_probs: List[np.ndarray] = []
    local_labels: List[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0
    is_main = misc.is_main_process()

    pbar = tqdm(
        total=len(data_loader),
        desc="  [eval ]",
        unit="batch",
        dynamic_ncols=True,
        disable=not is_main,
        leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )

    for batch in data_loader:
        images = batch[0].to(device, non_blocking=True)
        target = batch[-1].to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            output = model(images).to(device, non_blocking=True)
            loss = criterion(output, target)
        total_loss += loss.item()
        n_batches += 1
        local_probs.append(F.softmax(output, dim=1)[:, 1].detach().cpu().numpy())
        local_labels.append(target.detach().cpu().numpy())
        if is_main:
            pbar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)
            pbar.update(1)

    pbar.close()

    local_probs_np = np.concatenate(local_probs) if local_probs else np.array([])
    local_labels_np = np.concatenate(local_labels) if local_labels else np.array([])
    avg_loss = total_loss / max(n_batches, 1)

    # Gather predictions and labels from all DDP ranks
    if dist_mod.is_available() and dist_mod.is_initialized():
        world_size = dist_mod.get_world_size()
        gathered = [None] * world_size
        dist_mod.all_gather_object(gathered, {"probs": local_probs_np, "labels": local_labels_np})
        all_probs = np.concatenate([g["probs"] for g in gathered])
        all_labels = np.concatenate([g["labels"] for g in gathered])
        loss_t = torch.tensor(avg_loss, device=device)
        dist_mod.all_reduce(loss_t, op=dist_mod.ReduceOp.SUM)
        avg_loss = (loss_t / world_size).item()
    else:
        all_probs = local_probs_np
        all_labels = local_labels_np

    if len(all_probs) == 0:
        return {"accuracy": 0.0, "auc": 0.0, "real_accuracy": 0.0, "fake_accuracy": 0.0, "loss": avg_loss}

    hard_preds = (all_probs >= 0.5).astype(int)
    accuracy = float((hard_preds == all_labels).mean() * 100)

    try:
        auc = float(roc_auc_score(all_labels, all_probs) * 100)
    except Exception:
        auc = 0.0

    real_mask = all_labels == 0
    fake_mask = all_labels == 1
    real_acc = float((hard_preds[real_mask] == 0).mean() * 100) if real_mask.sum() > 0 else 0.0
    fake_acc = float((hard_preds[fake_mask] == 1).mean() * 100) if fake_mask.sum() > 0 else 0.0

    print(
        f"* Accuracy={accuracy:.2f}%  AUC={auc:.2f}%  "
        f"Real_Acc={real_acc:.2f}%  Fake_Acc={fake_acc:.2f}%  loss={avg_loss:.4f}"
    )
    return {
        "accuracy": accuracy,
        "auc": auc,
        "real_accuracy": real_acc,
        "fake_accuracy": fake_acc,
        "loss": avg_loss,
    }


@torch.no_grad()
def test(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    frame_labels = np.array([])  # int label
    frame_preds = np.array([])  # pred logit
    frame_y_preds = np.array([])  # pred int

    # for batch in metric_logger.log_every(data_loader, print_freq=len(data_loader), header=header):
    for batch in data_loader:
        images = batch[0]  # torch.Size([BS, C, H, W])
        target = batch[1]  # torch.Size([BS])

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            # output = model(images)
            output = model(images).to(device, non_blocking=True)  # modified
            loss = criterion(output, target)

        frame_pred = (F.softmax(output, dim=1)[:, 1].detach().cpu().numpy())
        frame_preds = np.append(frame_preds, frame_pred)

        frame_y_pred = np.argmax(output.detach().cpu().numpy(), axis=1)
        frame_y_preds = np.append(frame_y_preds, frame_y_pred)

        frame_label = (target.detach().cpu().numpy())
        frame_labels = np.append(frame_labels, frame_label)

        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_logger.meters['frame_acc'].update(frame_level_acc(frame_labels, frame_y_preds))
    metric_logger.meters['frame_balanced_acc'].update(frame_level_balanced_acc(frame_labels, frame_y_preds))
    metric_logger.meters['frame_auc'].update(frame_level_auc(frame_labels, frame_preds))
    metric_logger.meters['frame_eer'].update(frame_level_eer(frame_labels, frame_preds))

    print('*[------FRAME-LEVEL------] \n'
          'Acc {frame_acc.global_avg:.3f} Balanced_Acc {frame_balanced_acc.global_avg:.3f} '
          'Auc {frame_auc.global_avg:.3f} EER {frame_eer.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(frame_acc=metric_logger.frame_acc, frame_balanced_acc=metric_logger.frame_balanced_acc,
                  frame_auc=metric_logger.frame_auc, frame_eer=metric_logger.frame_eer, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_binary_video_frames(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    frame_labels = np.array([])  # int label
    frame_preds = np.array([])  # pred logit
    frame_y_preds = np.array([])  # pred int
    video_names_list = list()

    # for batch in metric_logger.log_every(data_loader, print_freq=len(data_loader), header=header):
    for batch in data_loader:
        images = batch[0]  # torch.Size([BS, C, H, W])
        target = batch[1]  # torch.Size([BS])
        video_name = batch[-1]  # list[BS]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            # output = model(images)
            output = model(images).to(device, non_blocking=True)  # modified
            loss = criterion(output, target)

        frame_pred = (F.softmax(output, dim=1)[:, 1].detach().cpu().numpy())
        frame_preds = np.append(frame_preds, frame_pred)

        frame_y_pred = np.argmax(output.detach().cpu().numpy(), axis=1)
        frame_y_preds = np.append(frame_y_preds, frame_y_pred)

        frame_label = (target.detach().cpu().numpy())
        frame_labels = np.append(frame_labels, frame_label)

        video_names_list.extend(list(video_name))

        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # metric_logger.meters['frame_acc'].update(frame_level_acc(frame_labels, frame_y_preds))
    metric_logger.meters['frame_balanced_acc'].update(frame_level_balanced_acc(frame_labels, frame_y_preds))
    metric_logger.meters['frame_auc'].update(frame_level_auc(frame_labels, frame_preds))
    metric_logger.meters['frame_eer'].update(frame_level_eer(frame_labels, frame_preds))

    print('*[------FRAME-LEVEL------] \n'
          'Balanced_Acc {frame_balanced_acc.global_avg:.3f} '
          'Auc {frame_auc.global_avg:.3f} '
          'EER {frame_eer.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(
        frame_balanced_acc=metric_logger.frame_balanced_acc,
        frame_auc=metric_logger.frame_auc,
        frame_eer=metric_logger.frame_eer,
        losses=metric_logger.loss)
    )

    # video-level metrics:
    frame_labels_list = frame_labels.tolist()
    frame_preds_list = frame_preds.tolist()

    video_label_list, video_pred_list, video_y_pred_list = get_video_level_label_pred(frame_labels_list, video_names_list, frame_preds_list)
    # print(len(video_label_list), len(video_pred_list), len(video_y_pred_list))
    # metric_logger.meters['video_acc'].update(video_level_acc(video_label_list, video_y_pred_list))
    metric_logger.meters['video_balanced_acc'].update(video_level_balanced_acc(video_label_list, video_y_pred_list))
    metric_logger.meters['video_auc'].update(video_level_auc(video_label_list, video_pred_list))
    metric_logger.meters['video_eer'].update(frame_level_eer(video_label_list, video_pred_list))

    print('*[------VIDEO-LEVEL------] \n'
          'Balanced_Acc {video_balanced_acc.global_avg:.3f} '
          'Auc {video_auc.global_avg:.3f} '
          'EER {video_eer.global_avg:.3f}'
          .format(
        video_balanced_acc=metric_logger.video_balanced_acc,
        video_auc=metric_logger.video_auc,
        video_eer=metric_logger.video_eer)
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
