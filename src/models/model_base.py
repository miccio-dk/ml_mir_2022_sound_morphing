import numpy as np
import torch
import torch.nn as nn

class VaeModelBase(nn.Module):
    def __init__(self, loss):
        super(VaeModelBase, self).__init__()
        self.loss = loss


    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu


    def forward(self, x, x_clean=None, label=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decode(z)
        if label is not None:
            if x_clean is None:
                x_clean = x
            losses = self.loss(x_clean, x_reconst, mu, logvar, z, label)
            return x_reconst, mu, logvar, z, losses
        return x_reconst, mu, logvar, z


    def get_infos(self):
        model_blocks = {
            "Backbone": self.resnet,
            "Loss": self.loss,
            "Total": self,
        }
        info_str = "Model size:\n"
        for name, block in model_blocks.items():
            if block is not None:
                n_params = sum([np.prod(p.size()) for p in block.parameters() if p.requires_grad]) / 1e6
                info_str += f"- {name:<8}:{n_params:>6.3f}M\n"
        return info_str
