import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FACL(nn.Module):
    def __init__(self, total_step, const_ratio=0.4, prob_init=1, prob_end=0, include_sigmoid=False):
        super(FACL, self).__init__()
        const_step = int(total_step*const_ratio)
        self.prob_init = prob_init
        self.prob_end = prob_end
        self.prob_thres = torch.linspace(prob_init, prob_end, int(total_step-const_step))
        self.step = 0
        self.out = 0
        self.include_sigmoid = include_sigmoid

    def get_thres(self): ## default micro_batch = 1
        prob = self.prob_thres[self.step] if self.step < len(self.prob_thres) else self.prob_thres[-1] ## init(=1) to end(=0)
        self.step += 1
        # return self.out
        return 1-prob ## from 1-init to 1-end

    def fal(self, fft_pred, fft_gt):
        return nn.MSELoss()(fft_pred.abs(), fft_gt.abs())

    def fcl(self, fft_pred, fft_gt):
        conj_pred = torch.conj(fft_pred)
        numerator = (conj_pred*fft_gt).sum().real
        denominator = torch.sqrt(((fft_gt).abs()**2).sum()*((fft_pred).abs()**2).sum())
        return 1. - numerator/denominator
    
    def forward(self, pred, gt):
        if self.include_sigmoid:
            pred = F.sigmoid(pred)
            gt = F.sigmoid(gt)

        fft_pred = torch.fft.fftn(pred, dim=[-1,-2], norm='ortho')
        fft_gt = torch.fft.fftn(gt, dim=[-1,-2], norm='ortho')
        prob = self.get_thres()
        
        H,W = pred.shape[-2:]
        weight = np.sqrt(H*W)
        loss = prob*self.fal(fft_pred, fft_gt) + (1-prob)*self.fcl(fft_pred, fft_gt)
        loss = loss * weight
        return loss
