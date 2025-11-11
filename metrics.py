import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ZR(nn.Module):
    def __init__(self, a=200, b=1.6):
        self.a = torch.tensor(a).float()
        self.b = torch.tensor(b).float()
    def to_dbz(self, img_rain):
        img_dbz = 10 * torch.log10(self.a * torch.pow(img_rain, self.b))
        img_dbz[img_dbz < 0] = 0     # set dbz=0 if rain is too small.
        return img_dbz
    def to_rain(self, img_dbz):
        img_rain = torch.pow(10, (0.1*img_dbz - torch.log10(self.a)) / self.b)
        return img_rain


def binarize(gts, preds, threshold):
    """
    Compute binarized ground truths and predictions.
    
    Inputs: 
        gts:   arbitrary tensor shape
        preds: arbitrary tensor shape
        threshold: a value to binarize the predictions and ground truths
    
    Return: binarized tensors, gts_bin and preds_bin
    """
    gts_bin = (gts >= threshold).float()
    preds_bin = (preds >= threshold).float()

    return gts_bin, preds_bin


def compute_hits_misses_fas(gts, preds, threshold):
    gts_bin, preds_bin = binarize(gts, preds, threshold)
    hits = torch.sum(gts_bin * preds_bin)
    misses = torch.sum(gts_bin * (1 - preds_bin))
    fas = torch.sum((1 - gts_bin) * preds_bin)
    return torch.stack((hits, misses, fas))

def compute_4confusion(gts, preds, threshold):
    gts_bin, preds_bin = binarize(gts, preds, threshold)
    hits = torch.sum(gts_bin * preds_bin)
    misses = torch.sum(gts_bin * (1 - preds_bin))
    fas = torch.sum((1 - gts_bin) * preds_bin)
    corneg = torch.sum((1 - gts_bin) * (1 - preds_bin)) # correctnegetives (True Negative (TF))
    return torch.stack((hits, misses, fas, corneg))


def compute_csi(confusion_components):
    """
    Compute Critical Success Index (CSI) for binary classification.
    
    Inputs: 
        confusion_components = torch.tensor(hits, misses, fas) 
                             = compute_hits_misses_fas(gts, preds, threshold)

    Return: CSI value
    """
    hits = confusion_components[0]
    misses = confusion_components[1]
    false_alarms = confusion_components[2]

    csi = hits / (hits + misses + false_alarms + 1e-6)
    return csi


def compute_bias(confusion_components):
    
    hits = confusion_components[0]
    misses = confusion_components[1]
    false_alarms = confusion_components[2]

    bias = (hits + false_alarms) / (hits + misses + 1e-6)
    return bias


def compute_pooled_confusion(gts, preds, threshold, pool_size=4, mode='avg'):
    """
    Compute Pooled Critical Success Index (CSI)

    Inputs:
        - gts:   tensor of shape (B, C, H, W)
        - preds: tensor of shape (B, C, H, W)
        - threshold: a value to binarize the predictions and ground truths
        - pool_size: size of the pooling window
        - mode: 'max' or 'avg', pooling method

    Return: CSI value
    """

    if mode == 'max':
        pool_fn = F.max_pool2d
    elif mode == 'avg':
        pool_fn = F.avg_pool2d
    else:
        raise ValueError("mode must be 'max' or 'avg'")
    
    stride = math.ceil(pool_size / 4)
    pooled_gts = pool_fn(gts, kernel_size=pool_size, stride=stride)
    pooled_preds = pool_fn(preds, kernel_size=pool_size, stride=stride)

    pooled_confusion = compute_hits_misses_fas(pooled_gts, pooled_preds, threshold)

    return pooled_confusion


def compute_pooled_4confusion(gts, preds, threshold, pool_size=4, mode='avg'):
    if mode == 'max':
        pool_fn = F.max_pool2d
    elif mode == 'avg':
        pool_fn = F.avg_pool2d
    else:
        raise ValueError("mode must be 'max' or 'avg'")
    
    stride = math.ceil(pool_size / 4)
    pooled_gts = pool_fn(gts, kernel_size=pool_size, stride=stride)
    pooled_preds = pool_fn(preds, kernel_size=pool_size, stride=stride)

    pooled_confusion = compute_4confusion(pooled_gts, pooled_preds, threshold)

    return pooled_confusion


def compute_hss(confusion_components):
    
    hits = confusion_components[0]          # TP
    misses = confusion_components[1]        # FN
    false_alarms = confusion_components[2]  # FP
    corneg = confusion_components[3]        # TN

    hss = 2 * (hits*corneg - misses*false_alarms) / ((hits+misses)*(misses+corneg) + (hits+false_alarms)*(false_alarms+corneg) + 1e-6)
    return hss

