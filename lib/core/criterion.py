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


# ------------------------------------------------------------------------------
# Adapted for GPS path segmentation from HRNet semantic segmentation
# ------------------------------------------------------------------------------


def get_gaussian_kernel2d(kernel_size=11, sigma=2.0):
    """Crée un noyau 2D gaussien normalisé pour flouter."""
    ax = torch.arange(kernel_size).float() - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def soft_distance(tensor, kernel_size=5, sigma=2.0):
    """
    Approximation différentiable d'une distance transform.
    tensor: [B, H, W]
    retourne: [B, H, W]
    """
    tensor = 1. - tensor  # inverser foreground / background

    B, H, W = tensor.shape
    kernel = get_gaussian_kernel2d(kernel_size, sigma).to(tensor.device)
    kernel = kernel.expand(1, 1, kernel_size, kernel_size)  # pas besoin de batch

    blurred = F.conv2d(tensor.unsqueeze(1), kernel, padding=kernel_size // 2, groups=1)
    return blurred.squeeze(1)  # [B, H, W]



class HausdorffLoss(nn.Module):
    def __init__(self, weight = [0.4], ignore_label=-1):
        super(HausdorffLoss, self).__init__()
        self.ignore_label = ignore_label
        self.weight = weight

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)

        if ph != h or pw != w:
            score = F.interpolate(score, size=(h, w), mode='bilinear',
                                  align_corners=config.MODEL.ALIGN_CORNERS)

        probs = F.softmax(score, dim=1)
        probs_fg = probs[:, 1, :, :]  # classe "chemin"

        target_fg = (target == 1).float()
        valid_mask = (target != self.ignore_label).float()

        probs_fg = probs_fg * valid_mask
        target_fg = target_fg * valid_mask

        dist_target = soft_distance(target_fg)
        dist_pred = soft_distance(probs_fg)

        diff = torch.abs(probs_fg - target_fg)
        loss = torch.mean(diff * (dist_target + dist_pred))

        return loss

    def forward(self, score, target):
        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        assert len(self.weight) == len(score)
        return sum([w * self._forward(x, target) for (w, x) in zip(self.weight, score)])


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.7, hausdorff_weight=0.3,
                 balance_weights=[1.0], ignore_label=-1):
        super(CombinedLoss, self).__init__()
        self.dice = DiceLoss(weight=balance_weights, ignore_label=ignore_label)
        self.hausdorff = HausdorffLoss(weight=balance_weights, ignore_label=ignore_label)
        self.dice_weight = dice_weight
        self.hausdorff_weight = hausdorff_weight

    def forward(self, score, target):
        loss_dice = self.dice(score, target)
        loss_haus = self.hausdorff(score, target)
        return self.dice_weight * loss_dice + self.hausdorff_weight * loss_haus
