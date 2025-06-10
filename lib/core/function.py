# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate

import utils.distributed as dist

import os
from PIL import Image
import torchvision.transforms.functional as TF




def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, _, _ = batch
        images = images.to(device)



        labels = labels.long().to(device)

        # Forward
        losses, pred = model(images, labels)  # pred = logits [B, C, H, W]

        loss = losses.mean()

        if dist.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # Time and loss update
        batch_time.update(time.time() - tic)
        tic = time.time()
        ave_loss.update(reduced_loss.item())

        # Adjust LR
        lr = adjust_learning_rate(optimizer, base_lr, num_iters, i_iter + cur_iters)

        # Logging + DEBUG
        if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg = f'Epoch: [{epoch}/{num_epoch}] Iter:[{i_iter}/{epoch_iters}], Time: {batch_time.average():.2f}, ' \
                  f'lr: {[x["lr"] for x in optimizer.param_groups]}, Loss: {ave_loss.average():.6f}'
            logging.info(msg)

            with torch.no_grad():
                # --- LOGITS STATS PAR CLASSE ---
                print("[DEBUG/function.py] -> pred (logits) stats :")
                print("  shape        :", pred.shape)
                print("  classe 0 - min :", pred[:, 0, :, :].min().item(),
                    ", max :", pred[:, 0, :, :].max().item(),
                    ", mean :", pred[:, 0, :, :].mean().item(),
                    ", std :", pred[:, 0, :, :].std().item())
                print("  classe 1 - min :", pred[:, 1, :, :].min().item(),
                    ", max :", pred[:, 1, :, :].max().item(),
                    ", mean :", pred[:, 1, :, :].mean().item(),
                    ", std :", pred[:, 1, :, :].std().item())

                unique_label, count_label = np.unique(labels.cpu().numpy(), return_counts=True)
                print("[DEBUG/function.py] -> val in label :", dict(zip(unique_label, count_label)))

                pred_classes = torch.argmax(pred, dim=1)
                unique_pred, count_pred = np.unique(pred_classes.cpu().numpy(), return_counts=True)
                print("[DEBUG/function.py] -> val in pred  :", dict(zip(unique_pred, count_pred)), "\n\n")


    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            image = image.to(device)
            label = label.long().to(device)

            losses, pred = model(image, label)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            if idx % 10 == 0:
                print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array


def testval(config, test_dataset, testloader, model,
            sv_dir='', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name, *border_padding = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            # --- LOGITS STATS PAR CLASSE ---
            print("[DEBUG/function.py] -> pred (logits) stats :")
            print("  shape        :", pred.shape)
            print("  classe 0 - min :", pred[:, 0, :, :].min().item(),
                  ", max :", pred[:, 0, :, :].max().item(),
                  ", mean :", pred[:, 0, :, :].mean().item(),
                  ", std :", pred[:, 0, :, :].std().item())
            print("  classe 1 - min :", pred[:, 1, :, :].min().item(),
                  ", max :", pred[:, 1, :, :].max().item(),
                  ", mean :", pred[:, 1, :, :].mean().item(),
                  ", std :", pred[:, 1, :, :].std().item())


            # --- GROUND TRUTH LABEL ---
            unique_label, count_label = np.unique(label.cpu().numpy(), return_counts=True)
            print("[DEBUG/function.py] -> val in label :", dict(zip(unique_label, count_label)))

            # --- PREDICTED CLASSES ---
            pred_classes = torch.argmax(pred, dim=1)  # shape [B, H, W]
            unique_pred, count_pred = np.unique(pred_classes.cpu().numpy(), return_counts=True)
            print("[DEBUG/function.py] -> val in pred  :", dict(zip(unique_pred, count_pred)), "\n\n")




            if len(border_padding) > 0:
                border_padding = border_padding[0]
                pred = pred[:, :, 0:pred.size(2) - border_padding[0], 0:pred.size(3) - border_padding[1]]

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model,
         sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
