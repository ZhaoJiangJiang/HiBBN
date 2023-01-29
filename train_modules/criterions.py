import torch
import torch.nn.functional as F
import numpy as np


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.5, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001

    def forward(self, logits, labels):
        pt = torch.sigmoid(logits)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * labels * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - labels) * torch.log(1 - pt)
        return torch.mean(loss)

class ClassificationLoss(torch.nn.Module):
    def __init__(self, loss_type="bce"):
        super(ClassificationLoss, self).__init__()
        self.loss_type = loss_type
        self.focal_loss_fn = FocalLoss()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        device = logits.device
        if self.loss_type == "bce":
            loss = self.loss_fn(logits, targets)
        elif self.loss_type == "focal":
            loss = self.focal_loss_fn(logits, targets)
        else:
            loss = 0
        return loss


class ContrastLoss(torch.nn.Module):
    def __init__(self, loss_type="bce"):
        super(ContrastLoss, self).__init__()
        self.loss_type = loss_type
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, targets, mask):
        device = logits[0][0].device
        targets = targets.to(device)
        mask = mask.to(device)
        logits = torch.mul(logits, mask)
        targets = torch.mul(targets, mask)
        loss = self.loss_fn(logits, targets)
        return loss