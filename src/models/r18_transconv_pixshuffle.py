# ResNet18 model with TransposedConv2D + PixelShuffle Output
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (resnet18, resnet34,
                                shufflenet_v2_x1_0, mobilenetv3)
from torchvision.models import (ResNet18_Weights, ResNet34_Weights,
                                ShuffleNet_V2_X1_0_Weights, MobileNet_V3_Small_Weights)
from models.model_base import VaeModelBase


class VaeModel(VaeModelBase):
    def __init__(self, loss, fc_hidden1=512, fc_hidden2=256, fc_hidden3=1024, lspace_size=256, freeze_model=False):
        super(VaeModelBase, self).__init__(loss)
        # Model Parameters
        assert fc_hidden3 % 64 == 0
        self.fc_hidden1, self.fc_hidden2, self.fc_hidden3, self.lspace_size = fc_hidden1, fc_hidden2, fc_hidden3, lspace_size

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
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.lspace_size)
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.lspace_size)

        # Sampling vector
        self.fc4 = nn.Linear(self.lspace_size, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, self.fc_hidden3)
        self.fc_bn5 = nn.BatchNorm1d(self.fc_hidden3)
        self.relu = nn.ReLU(inplace=True)

        # DECODER - Using Transposed Conv2d
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        
        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=4, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(4, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans9 = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(1, momentum=0.01),
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

    def decode(self, z):
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = self.relu(self.fc_bn5(self.fc5(x))).view(x.shape[0], -1, 4, 8)      # ([32, 256, 4, 4]) => ([32, 128, 4, 8]) => ([32, 64, 4, 16])  => 131,072
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = self.convTrans9(x)
        x = F.interpolate(x, size=(80, 251), mode='bilinear')
        return x



if __name__ == "__main__":
    from loss import VaeLoss

    x = torch.rand((32,1,80,251))
    print("input: ", x.shape)

    loss = VaeLoss(rec_weight=1, kld_weight=0.01)
    model = VaeModel(loss, fc_hidden1=512, fc_hidden2=1024, fc_hidden3=4096, lspace_size=256, freeze_model=False)

    x_reconst, mu, logvar, z = model(x)
    print("output: ", x_reconst.shape)