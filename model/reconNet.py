"""
Channel-wise Image Reconstruction using Convolutional Neural Networks
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder(nn.Module):
    '''
        Encoder for the Reconstruction network
    '''

    def __init__(self, batchNorm_momentum):
        super(encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.ReLU(True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(True),

            nn.Conv2d(512, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024, momentum=batchNorm_momentum),
            nn.ReLU(True)
        )

    def forward(self, input):
        output = self.main(input)
        return output

class decoder(nn.Module):
    '''
        Decoder for the Reconstruction Network
    '''

    def __init__(self, batchNorm_momentum):
        super(decoder, self).__init__()

        self.CT1 = nn.ConvTranspose2d(1024, 512, 4, 1, 0, bias=False),
        self.BN1 = nn.BatchNorm2d(512, momentum=batchNorm_momentum),
        self.D1 = nn.Dropout2d(),
        self.R1 = nn.ReLU(True),

        self.CT2 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
        self.BN2 = nn.BatchNorm2d(256, momentum=batchNorm_momentum),
        self.D2 = nn.Dropout2d(),
        self.R2 = nn.ReLU(True),

        self.CT3 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
        self.BN3 = nn.BatchNorm2d(128, momentum=batchNorm_momentum),
        self.D3 = nn.Dropout2d(),
        self.R3 = nn.ReLU(True),

        self.CT4 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
        self.BN4 = nn.BatchNorm2d(64, momentum=batchNorm_momentum),
        self.R4 = nn.ReLU(True),

        self.CT5A = nn.ConvTranspose2d(64, 255, 4, 2, 1, bias=False),
        self.CT5B = nn.ConvTranspose2d(64, 255, 4, 2, 1, bias=False),
        self.CT5B = nn.ConvTranspose2d(64, 255, 4, 2, 1, bias=False),
        self.SM = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.CT1(input)
        x = self.BN1(x)
        x = self.D1(x)
        x = self.R1(x)

        x = self.CT2(x)
        x = self.BN2(x)
        x = self.D2(x)
        x = self.R2(x)

        x = self.CT3(x)
        x = self.BN3(x)
        x = self.D3(x)
        x = self.R3(x)

        x = self.CT4(x)
        x = self.BN4(x)
        x = self.R4(x)

        R = self.SM(self.CT5A(x))
        G = self.SM(self.CT5B(x))
        B = self.SM(self.CT5C(x))

        return R, G, B

class ReconNet(nn.Module):
    """ReconNet network."""

    def __init__(self, batchNorm_momentum):
        """Init fields."""
        super(ReconNet, self).__init__()
        self.encoder = encoder(batchNorm_momentum)
        self.decoder = decoder(batchNorm_momentum)

    def forward(self, x):
        latent = self.encoder(x)
        R, G, B = self.decoder(latent)

        return R, G, B
