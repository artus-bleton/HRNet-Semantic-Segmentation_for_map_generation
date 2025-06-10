# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import logging
from config import config
import torchvision.transforms as T
import gudhi as gd
import math




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


# On suppose que tu gardes tes fonctions existantes :
# - compute_dgm_force
# - getCriticalPoints
# - getTopoLoss (version fonctionnelle que je vais intégrer ici dans _forward)

def compute_dgm_force(lh_dgm, gt_dgm, pers_thresh=0.03, pers_thresh_perfect=0.99, do_return_perfect=False):
    """
    Compute the persistent diagram of the image

    Args:
        lh_dgm: likelihood persistent diagram.
        gt_dgm: ground truth persistent diagram.
        pers_thresh: Persistent threshold, which also called dynamic value, which measure the difference.
        between the local maximum critical point value with its neighouboring minimum critical point value.
        The value smaller than the persistent threshold should be filtered. Default: 0.03
        pers_thresh_perfect: The distance difference between two critical points that can be considered as
        correct match. Default: 0.99
        do_return_perfect: Return the persistent point or not from the matching. Default: False

    Returns:
        force_list: The matching between the likelihood and ground truth persistent diagram
        idx_holes_to_fix: The index of persistent points that requires to fix in the following training process
        idx_holes_to_remove: The index of persistent points that require to remove for the following training
        process

    """
    lh_pers = abs(lh_dgm[:, 1] - lh_dgm[:, 0])
    if (gt_dgm.shape[0] == 0):
        gt_pers = None;
        gt_n_holes = 0;
    else:
        gt_pers = gt_dgm[:, 1] - gt_dgm[:, 0]
        gt_n_holes = gt_pers.size  # number of holes in gt

    if (gt_pers is None or gt_n_holes == 0):
        idx_holes_to_fix = list();
        idx_holes_to_remove = list(set(range(lh_pers.size)))
        idx_holes_perfect = list();
    else:
        # check to ensure that all gt dots have persistence 1
        tmp = gt_pers > pers_thresh_perfect

        # get "perfect holes" - holes which do not need to be fixed, i.e., find top
        # lh_n_holes_perfect indices
        # check to ensure that at least one dot has persistence 1; it is the hole
        # formed by the padded boundary
        # if no hole is ~1 (ie >.999) then just take all holes with max values
        tmp = lh_pers > pers_thresh_perfect  # old: assert tmp.sum() >= 1
        lh_pers_sorted_indices = np.argsort(lh_pers)[::-1]
        if np.sum(tmp) >= 1:
            lh_n_holes_perfect = tmp.sum()
            idx_holes_perfect = lh_pers_sorted_indices[:lh_n_holes_perfect];
        else:
            idx_holes_perfect = list();

        # find top gt_n_holes indices
        idx_holes_to_fix_or_perfect = lh_pers_sorted_indices[:gt_n_holes];

        # the difference is holes to be fixed to perfect
        idx_holes_to_fix = list(
            set(idx_holes_to_fix_or_perfect) - set(idx_holes_perfect))

        # remaining holes are all to be removed
        idx_holes_to_remove = lh_pers_sorted_indices[gt_n_holes:];

    # only select the ones whose persistence is large enough
    # set a threshold to remove meaningless persistence dots
    pers_thd = pers_thresh
    idx_valid = np.where(lh_pers > pers_thd)[0]
    idx_holes_to_remove = list(
        set(idx_holes_to_remove).intersection(set(idx_valid)))

    force_list = np.zeros(lh_dgm.shape)

    # push each hole-to-fix to (0,1)
    force_list[idx_holes_to_fix, 0] = 0 - lh_dgm[idx_holes_to_fix, 0]
    force_list[idx_holes_to_fix, 1] = 1 - lh_dgm[idx_holes_to_fix, 1]

    # push each hole-to-remove to (0,1)
    force_list[idx_holes_to_remove, 0] = lh_pers[idx_holes_to_remove] / \
                                         math.sqrt(2.0)
    force_list[idx_holes_to_remove, 1] = -lh_pers[idx_holes_to_remove] / \
                                         math.sqrt(2.0)


    if (do_return_perfect):
        return force_list, idx_holes_to_fix, idx_holes_to_remove, idx_holes_perfect

    return force_list, idx_holes_to_fix, idx_holes_to_remove

def getCriticalPoints(likelihood):
    """
    Compute the critical points of the image (Value range from 0 -> 1)

    Args:
        likelihood: Likelihood image from the output of the neural networks

    Returns:
        pd_lh:  persistence diagram.
        bcp_lh: Birth critical points.
        dcp_lh: Death critical points.
        Bool:   Skip the process if number of matching pairs is zero.

    """
    lh = 1 - likelihood
    lh_vector = np.asarray(lh).flatten()

    lh_cubic = gd.CubicalComplex(
        dimensions=[lh.shape[0], lh.shape[1]],
        top_dimensional_cells=lh_vector
    )

    Diag_lh = lh_cubic.persistence(homology_coeff_field=2, min_persistence=0)
    pairs_lh = lh_cubic.cofaces_of_persistence_pairs()

    # If the paris is 0, return False to skip
    if (len(pairs_lh[0])==0): return 0, 0, 0, False

    # return persistence diagram, birth/death critical points
    pd_lh = np.array([[lh_vector[pairs_lh[0][0][i][0]], lh_vector[pairs_lh[0][0][i][1]]] for i in range(len(pairs_lh[0][0]))])
    bcp_lh = np.array([[pairs_lh[0][0][i][0]//lh.shape[1], pairs_lh[0][0][i][0]%lh.shape[1]] for i in range(len(pairs_lh[0][0]))])
    dcp_lh = np.array([[pairs_lh[0][0][i][1]//lh.shape[1], pairs_lh[0][0][i][1]%lh.shape[1]] for i in range(len(pairs_lh[0][0]))])

    return pd_lh, bcp_lh, dcp_lh, True

class TopoLoss(nn.Module):
    def __init__(self, weight, ignore_label=-1, smooth=1.0, topo_size=80):
        super(TopoLoss, self).__init__()
        self.ignore_label = ignore_label
        self.smooth = smooth
        self.weight = weight
        self.topo_size = topo_size

    def _get_topo_loss(self, likelihood_tensor, gt_tensor):
        likelihood = torch.sigmoid(likelihood_tensor).clone()
        gt = gt_tensor.clone()

        likelihood = torch.squeeze(likelihood).cpu().detach().numpy()
        gt = torch.squeeze(gt).cpu().detach().numpy()

        topo_cp_weight_map = np.zeros(likelihood.shape)
        topo_cp_ref_map = np.zeros(likelihood.shape)

        for y in range(0, likelihood.shape[0], self.topo_size):
            for x in range(0, likelihood.shape[1], self.topo_size):
                lh_patch = likelihood[y:min(y + self.topo_size, likelihood.shape[0]),
                                      x:min(x + self.topo_size, likelihood.shape[1])]
                gt_patch = gt[y:min(y + self.topo_size, gt.shape[0]),
                              x:min(x + self.topo_size, gt.shape[1])]

                if (np.min(lh_patch) == 1 or np.max(lh_patch) == 0): continue
                if (np.min(gt_patch) == 1 or np.max(gt_patch) == 0): continue

                pd_lh, bcp_lh, dcp_lh, pairs_lh_pa = getCriticalPoints(lh_patch)
                pd_gt, bcp_gt, dcp_gt, pairs_lh_gt = getCriticalPoints(gt_patch)

                if not (pairs_lh_pa): continue
                if not (pairs_lh_gt): continue
                if (len(pd_lh.shape) != 2): continue

                force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(pd_lh, pd_gt, pers_thresh=0.03)

                if (len(idx_holes_to_fix) > 0 or len(idx_holes_to_remove) > 0):
                    for hole_indx in idx_holes_to_fix:
                        # birth point
                        if 0 <= bcp_lh[hole_indx][0] < likelihood.shape[0] and 0 <= bcp_lh[hole_indx][1] < likelihood.shape[1]:
                            topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 1
                            topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 0
                        # death point
                        if 0 <= dcp_lh[hole_indx][0] < likelihood.shape[0] and 0 <= dcp_lh[hole_indx][1] < likelihood.shape[1]:
                            topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 1
                            topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 1
                    for hole_indx in idx_holes_to_remove:
                        # birth point
                        if 0 <= bcp_lh[hole_indx][0] < likelihood.shape[0] and 0 <= bcp_lh[hole_indx][1] < likelihood.shape[1]:
                            topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 1
                            if 0 <= dcp_lh[hole_indx][0] < likelihood.shape[0] and 0 <= dcp_lh[hole_indx][1] < likelihood.shape[1]:
                                topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = \
                                    lh_patch[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])]
                            else:
                                topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 1
                        # death point
                        if 0 <= dcp_lh[hole_indx][0] < likelihood.shape[0] and 0 <= dcp_lh[hole_indx][1] < likelihood.shape[1]:
                            topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 1
                            if 0 <= bcp_lh[hole_indx][0] < likelihood.shape[0] and 0 <= bcp_lh[hole_indx][1] < likelihood.shape[1]:
                                topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = \
                                    lh_patch[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])]
                            else:
                                topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 0

        topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float).cuda()
        topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float).cuda()

        # MSE Loss on critical points
        loss_topo = (((likelihood_tensor * topo_cp_weight_map) - topo_cp_ref_map) ** 2).sum()
        return loss_topo

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)

        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w),
                                  mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        # Softmax et extraction de la classe "chemin" (index 1)
        probs = F.softmax(score, dim=1)
        probs_fg = probs[:, 1, :, :]  # classe "chemin"

        target_fg = (target == 1).float()
        valid_mask = (target != self.ignore_label).float()

        probs_fg = probs_fg * valid_mask
        target_fg = target_fg * valid_mask

        # On suppose batch size 1 pour getTopoLoss actuel
        #assert probs_fg.shape[0] == 1, "TopoLoss actuelle ne supporte que batch_size=1"

        # Calcul TopoLoss
        loss_topo = 0
        for i in range(probs.shape[0]):
            print("loss :", i ,"/32")
            loss_topo = loss_topo + self._get_topo_loss(probs_fg[i], target_fg[i])
        return loss_topo / probs.shape[0]

    def forward(self, score, target):
        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        assert len(self.weight) == len(score)
        return sum([w * self._forward(x, target) for (w, x) in zip(self.weight, score)])



class Ce_Tl(nn.Module):
    def __init__(self, weight, lbd=0.5, ignore_label=-1, smooth=1.0):
        super(Ce_Tl, self).__init__()

        self.lamda = lbd

        self.ce = CrossEntropy()
        self.tl = TopoLoss(weight=weight)

    def forward(self, score, target):
        return self.ce(score, target) + self.lamda * self.tl(score, target)

class Di_Tl(nn.Module):
    def __init__(self, weight, lbd=0.5, ignore_label=-1, smooth=1.0):
        super(Di_Tl, self).__init__()

        self.lamda = lbd

        self.di = DiceLoss(weight=weight)
        self.tl = TopoLoss(weight=weight)

    def forward(self, score, target):
        return self.di(score, target) + self.lamda * self.tl(score, target)
