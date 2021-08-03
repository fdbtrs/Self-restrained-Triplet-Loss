import math
import re
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Module

import numpy as np



class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))

def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()
def custom_cosine(u,v,ln):
    uv = torch.sum(u * v,dim=1,keepdim=True)
    uu = torch.sum(u * u,dim=1,keepdim=True)
    vv = torch.sum(v * v,dim=1,keepdim=True)
    uu_vv=uu*vv
    dist = uv / torch.sqrt(uu_vv.clamp(1e-8))
    dist=dist.clamp(max=1)
    # Return absolute value to avoid small negative value due to rounding
    return torch.abs(1.0 - (dist.sum(dim=1,keepdim=True)/ln).mean())




def average(u,weights,dim=1):
    if(weights !=None):
        uw=u*weights
    else:
        uw=u
    return torch.sum(uw,dim=dim,keepdim=True) /torch.sum(weights,dim=dim,keepdim=True)

def weighted_cosine(u, v,w, eps=1e-8,centered=False):
    if(centered):
        umu = average(u, weights=w)
        vmu = average(v, weights=w)
        u = u - umu
        v = v - vmu
    uv = average(u * v, weights=w)
    uu = average(u*u, weights=w)
    vv = average(v*v, weights=w)
    dist =  uv / torch.sqrt(uu * vv)
    # Return absolute value to avoid small negative value due to rounding
    return torch.abs(1.0-dist.mean())

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample 0002
    """
    """
    margin 2.0
    """
    def __init__(self, margin = 2.0, alpha=0.95, distance="cosine"):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.alpha=alpha
        self.distance=distance
        self.tripletMargin=torch.nn.TripletMarginLoss(margin=1.0,swap=True,reduction='mean')
    def l2_norm(self,input, axis=1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)
        return output
    def norm_l2_distnace(self,emb1,emb2,dim=1,centered=False):
        if (centered):
            umu = torch.sum(emb1,dim=dim,keepdim=True)
            vmu = torch.sum(emb2,dim=dim,keepdim=True)
            emb1 = emb1 - umu
            emb2 = emb2 - vmu
        uv = torch.sum(emb1 * emb2,dim=1, keepdim=True)
        norm1 =  torch.sum(emb1 * emb1,dim=1,keepdim=True)
        norm2 = torch.sum(emb2 * emb2,dim=1, keepdim=True)
        dist = torch.div(uv, torch.sqrt( norm1)* torch.sqrt(norm2))
        return 2.0*( 1.0- dist)

    def sum(self,emb,dim=1):
        return torch.sum(emb,dim=dim,keepdim=True)
    def cosine(self, emb1, emb2):
        uv = torch.sum(emb1 * emb2,dim=1)
        uu = torch.sum(emb2 * emb2,dim=1)
        vv = torch.sum(emb1 * emb1,dim=1)
        dist =torch.div( uv , torch.sqrt(uu * vv))
        return dist
    def l2(self,emb1,emb2,dim=1):
        sub=torch.sub(emb1,emb2).pow(2)
        sm=torch.sum(sub,dim=1)
        return sm
    def dis(self,anchor, positive):
        pos_dist = torch.sum((torch.sub(anchor, positive).pow(2)), 1)
        return pos_dist

    def forward(self, anchor, positive, negative, size_average=True):
        if(self.distance=="Triplet" or self.distance=="TTriplet"  ):
            self.margin = 1.0
            if(self.distance=="TTriplet"):
                self.margin=1
                losses=self.tripletMargin(anchor,positive,negative)
                return losses, losses, losses
            positive = F.normalize(positive, p=2, dim=1)
            anchor = F.normalize(anchor, p=2, dim=1)
            negative = F.normalize(negative, p=2, dim=1)

            positive_loss = self.dis(anchor, positive)
            distance_negative = self.dis(anchor, negative)
            distance_p_n = self.dis(positive, negative)
            losses = F.relu(positive_loss + self.margin - distance_negative).mean()
            return (losses), (positive_loss.mean()), (distance_negative.mean()), distance_p_n.mean()
        elif(self.distance=="SRT"):
            self.margin=2.0
            positive=F.normalize(positive,p=2,dim=1)
            anchor=F.normalize(anchor,p=2,dim=1)
            negative=F.normalize(negative,p=2,dim=1)

            positive_loss = self.dis(anchor, positive)
            distance_negative = self.dis(anchor, negative)
            distance_p_n = self.dis(positive, negative)

            cond = distance_negative.mean() >= distance_p_n.mean()  # + 0.5*self.margin

            ls = torch.where(cond, (positive_loss + self.margin - distance_p_n.mean()),
                             (positive_loss + self.margin - distance_negative))
            losses = F.relu(ls).mean()
            return (losses), (positive_loss.mean()), (distance_negative.mean()),distance_p_n.mean()
        else:
            positive_loss = torch.abs(1.0- self.cosine(anchor,positive).mean())
            distance_negative =self.cosine(anchor,negative)
            negative_loss=F.relu( distance_negative-self.margin).mean()
            losses = self.alpha* positive_loss +(1.0-self.alpha)*negative_loss
            return (losses ),(positive_loss),(negative_loss)
