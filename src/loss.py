
import torch
import torch.nn as nn
import torch.nn.functional as F


class VaeLoss(nn.Module):
    def __init__(self, rec_weight, kld_weight):
        super(VaeLoss, self).__init__()
        self.rec_weight = rec_weight
        self.kld_weight = kld_weight

    def forward(self, x_true, x_pred, mean, log_var, z, label_true=None):
        rec = F.mse_loss(x_pred, x_true, reduction='mean')
        kld = torch.mean(-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1), dim=0)
        loss = rec * self.rec_weight + kld * self.kld_weight
        return loss, rec, kld


class VaeLossExtended(VaeLoss):
    def __init__(self, rec_weight, kld_weight, ce_weights, lspace_size, n_labels):
        super(VaeLossExtended, self).__init__(rec_weight, kld_weight)
        self.ce_weights = ce_weights
        self.fc = nn.Linear(lspace_size, n_labels)

    def forward(self, x_true, x_pred, mean, log_var, z, label_true):
        _loss, rec, kld = super(VaeLossExtended, self).forward(x_true, x_pred, mean, log_var)
        logits = self.fc(z)
        ce = F.cross_entropy(logits, label_true, reduction='mean')
        loss = _loss + ce * self.ce_weights
        return loss, rec, kld, ce


