#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn


class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.005,
                 reduction='mean',
                 lb_ignore=255,
                 ):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)

        loss = -torch.sum(logs*label, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            loss = loss
        return loss


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR decayed by 5 every 10 epochs"""
    # lr = args.lr * (0.2 ** (epoch // 10))
    if epoch <= 20:
        lr = learning_rate
    elif epoch > 20 and epoch < 35:
        lr = learning_rate * 0.1
    elif epoch >= 35 and epoch < 40:
        lr = learning_rate * 0.1 * 0.1
    elif epoch >= 40:
        lr = learning_rate * 0.1 * 0.1 *0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr