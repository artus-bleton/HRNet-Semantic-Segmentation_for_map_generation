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
import torchvision.transforms as T



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



class DiceLoss(nn.Module):
    def __init__(self, weight, ignore_label=-1, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.ignore_label = ignore_label
        self.smooth = smooth
        self.weight = weight

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)

        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w),
                                  mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        # Softmax et extraction de la classe "chemin" (supposée en index 1)
        probs = F.softmax(score, dim=1)
        probs_fg = probs[:, 1, :, :]  # classe "chemin"
        #print(f"[DEBUG] probs_fg min: {probs_fg.min().item():.4f}, max: {probs_fg.max().item():.4f}")

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


        assert len(self.weight) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(self.weight, score)])


class TverskyLoss(nn.Module):
    def __init__(self, weight, alpha=0.4, beta=0.6, ignore_label=-1, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.ignore_label = ignore_label
        self.smooth = smooth
        self.weight = weight
        self.alpha = alpha
        self.beta = beta

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)

        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w),
                                  mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        probs = F.softmax(score, dim=1)
        probs_fg = probs[:, 1, :, :]  # classe "chemin"

        target_fg = (target == 1).float()

        valid_mask = (target != self.ignore_label).float()

        probs_fg = probs_fg * valid_mask
        target_fg = target_fg * valid_mask

        TP = (probs_fg * target_fg).sum()
        FP = (probs_fg * (1 - target_fg)).sum()
        FN = ((1 - probs_fg) * target_fg).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return 1. - tversky

    def forward(self, score, target):
        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        assert len(self.weight) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(self.weight, score)])

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight, alpha=0.4, beta=0.6, gamma=1.33, ignore_label=-1, smooth=1.0, class_idx=1):
        super(FocalTverskyLoss, self).__init__()
        self.ignore_label = ignore_label
        self.smooth = smooth
        self.weight = weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.class_idx = class_idx  # index de la classe cible (ex: chemin)

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)

        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w),
                                  mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        probs = F.softmax(score, dim=1)
        probs_fg = probs[:, self.class_idx, :, :]  # ta classe cible

        target_fg = (target == self.class_idx).float()
        valid_mask = (target != self.ignore_label).float()

        probs_fg = probs_fg * valid_mask
        target_fg = target_fg * valid_mask

        TP = (probs_fg * target_fg).sum()
        FP = (probs_fg * (1 - target_fg)).sum()
        FN = ((1 - probs_fg) * target_fg).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        # LF_T = (1 - Tversky)^(1/gamma), sur ta classe uniquement
        focal_tversky = torch.pow((1 - tversky), 1.0 / self.gamma)

        return focal_tversky

    def forward(self, score, target):
        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        assert len(self.weight) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(self.weight, score)])
