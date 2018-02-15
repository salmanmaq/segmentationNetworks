"""
PyTorch implementation of
SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation

https://arxiv.org/abs/1511.00561
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder(nn.Module):
    '''
        Encoder for the Segmentation network
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
        Decoder for the Segmentation Network
    '''

    def __init__(self, batchNorm_momentum, num_classes=19):
        super(decoder, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 19, 4, 2, 1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        output = self.main(input)
        return output

class SegNet(nn.Module):
    """Segnet network."""

    def __init__(self, batchNorm_momentum, num_classes):
        """Init fields."""
        super(SegNet, self).__init__()
        self.encoder = encoder(batchNorm_momentum)
        self.decoder = decoder(batchNorm_momentum, num_classes)

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)

        return output
