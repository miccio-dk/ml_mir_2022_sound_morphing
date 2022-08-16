import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import (resnet18, resnet34,
                                shufflenet_v2_x1_0, mobilenetv3)
from torchvision.models import (ResNet18_Weights, ResNet34_Weights,
                                ShuffleNet_V2_X1_0_Weights, MobileNet_V3_Small_Weights)


class VaeModel(nn.Module):
    def __init__(self, loss, fc_hidden1=512, fc_hidden2=256, CNN_embed_dim=256, freeze_model=False):
        super(VaeModel, self).__init__()
        self.loss = loss

        # Model Parameters
        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # ENCODER
        # Load model backbone with pre-trained weights
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Change number of channels for first layer
        backbone.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                                   kernel_size=(7, 7), stride=(2, 2),
                                   padding=(3, 3), bias=False)

        # Freeze Backbone
        if freeze_model:
            for params in backbone.parameters():
                params.requires_grad = False

            # Make first conv2d layer trainable => required if using frozen model
            for name, params in backbone.named_parameters():
                if "conv1" in name:
                    params.requires_grad = True

        # Remove last layer from model
        modules = list(backbone.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(backbone.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)

        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # # DECODER - Using Transposed Conv2d
        # self.convTrans6 = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
        #     nn.BatchNorm2d(32, momentum=0.01),
        #     nn.ReLU(inplace=True),
        # )
        # self.convTrans7 = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(3, 3), stride=(2, 2)),
        #     nn.BatchNorm2d(8, momentum=0.01),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.convTrans8 = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(3, 3), stride=(2, 2)),
        #     nn.BatchNorm2d(1, momentum=0.01),
        #     nn.Sigmoid()
        # )

        # DECODER - Using PixelShuffle Layers
        self.convTrans6 = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(16, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(4, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(1, momentum=0.01),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(80, 251), mode='bilinear')

        return x

    def forward(self, x, label=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decode(z)
        if label is not None:
            losses = self.loss(x, x_reconst, mu, logvar, z, label)
            return x_reconst, losses
        return x_reconst


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