# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
from config import config


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        functions = [self._ce_forward] * \
            (len(weights) - 1) + [self._ohem_forward]
        return sum([
            w * func(x, target)
            for (w, x, func) in zip(weights, score, functions)
        ])


class TolerantDiceLoss(nn.Module):
    def __init__(self, ignore_label=-1, kernel_size=3, smooth=1.0):
        super(TolerantDiceLoss, self).__init__()
        self.ignore_label = ignore_label
        self.kernel_size = kernel_size
        self.smooth = smooth

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)

        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w),
                                  mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        # Softmax to get class probabilities (assumes binary segmentation: 2 classes)
        probs = F.softmax(score, dim=1)
        probs_fg = probs[:, 1, :, :]  # Foreground: chemin
        target_fg = (target == 1).float()

        # Mask ignore_label
        valid_mask = (target != self.ignore_label).float()

        # Dilate target for spatial tolerance
        kernel = torch.ones((1, 1, self.kernel_size, self.kernel_size),
                            device=score.device, dtype=torch.float)
        target_dilated = F.conv2d(target_fg.unsqueeze(1), kernel, padding=self.kernel_size // 2)
        target_dilated = torch.clamp(target_dilated, 0, 1).squeeze(1)

        # Apply valid mask
        probs_fg = probs_fg * valid_mask
        target_dilated = target_dilated * valid_mask

        intersection = (probs_fg * target_dilated).sum()
        union = probs_fg.sum() + target_dilated.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])


class DiceLoss(nn.Module):
    def __init__(self, ignore_label=-1, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.ignore_label = ignore_label
        self.smooth = smooth

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)

        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w),
                                  mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        # Softmax et extraction de la classe "chemin" (supposée en index 1)
        probs = F.softmax(score, dim=1)
        probs_fg = probs[:, 1, :, :]  # classe "chemin"
        target_fg = (target == 1).float()

        # Masque d'exclusion des pixels ignorés
        valid_mask = (target != self.ignore_label).float()

        # Application du masque
        probs_fg = probs_fg * valid_mask
        target_fg = target_fg * valid_mask

        # Calcul Dice
        intersection = (probs_fg * target_fg).sum()
        union = probs_fg.sum() + target_fg.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1. - dice

    def forward(self, score, target):
        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])

class DistanceAwareDiceLoss(nn.Module):
    def __init__(self, ignore_label=-1, smooth=1.0, lambda_distance=1.0):
        super(DistanceAwareDiceLoss, self).__init__()
        self.ignore_label = ignore_label
        self.smooth = smooth
        self.lambda_distance = lambda_distance

    def _forward(self, pred_logits, target, distance_map):
        ph, pw = pred_logits.size(2), pred_logits.size(3)
        h, w = target.size(1), target.size(2)

        if ph != h or pw != w:
            pred_logits = F.interpolate(pred_logits, size=(h, w),
                                        mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
            distance_map = F.interpolate(distance_map.unsqueeze(1), size=(h, w),
                                         mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS).squeeze(1)

        probs = F.softmax(pred_logits, dim=1)
        probs_fg = probs[:, 1, :, :]
        target_fg = (target == 1).float()
        valid_mask = (target != self.ignore_label).float()

        # Dice Loss
        probs_fg_dice = probs_fg * valid_mask
        target_fg_dice = target_fg * valid_mask
        intersection = (probs_fg_dice * target_fg_dice).sum()
        union = probs_fg_dice.sum() + target_fg_dice.sum()
        dice_loss = 1. - (2. * intersection + self.smooth) / (union + self.smooth)

        # Distance Penalty
        distance_map = distance_map * valid_mask
        probs_fg_penalty = probs_fg * valid_mask
        penalty = (probs_fg_penalty * distance_map).sum() / (probs_fg_penalty.sum() + 1e-6)

        return dice_loss + self.lambda_distance * penalty

    def forward(self, score, target, distance_map):
        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target, distance_map) for (w, x) in zip(weights, score)])
