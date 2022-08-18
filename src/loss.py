
import torch
import torch.nn as nn
import torch.nn.functional as F


class VaeLoss(nn.Module):
    def __init__(self, rec_weight, kld_weight, **kwargs):
        super(VaeLoss, self).__init__()
        self.rec_weight = rec_weight
        self.kld_weight = kld_weight

    def forward(self, x_true, x_pred, mean, log_var, z, label_true=None):
        rec = F.mse_loss(x_pred, x_true, reduction='mean')
        kld = torch.mean(-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1), dim=0)
        loss = rec * self.rec_weight + kld * self.kld_weight
        return loss, rec, kld


class VaeLossClasses(VaeLoss):
    def __init__(self, rec_weight, kld_weight, ce_weight, lspace_size, n_classes, **kwargs):
        super(VaeLossClasses, self).__init__(rec_weight, kld_weight)
        self.ce_weight = ce_weight
        self.fc_classes = nn.Linear(lspace_size, n_classes)

    def forward(self, x_true, x_pred, mean, log_var, z, label_true):
        _loss, rec, kld = super(VaeLossClasses, self).forward(x_true, x_pred, mean, log_var, z, label_true)
        if label_true is not None and len(label_true) == x_true.shape[0]:
            logits = self.fc_classes(z)
            ce = F.cross_entropy(logits, label_true, reduction='mean')
        else:
            ce = torch.Tensor([0.0]).to(x_true.device)
        loss = _loss + ce * self.ce_weight
        return loss, rec, kld, ce

