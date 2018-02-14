"""
PyTorch implementation of
SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation

https://arxiv.org/abs/1511.00561
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

# this may be unstable sometimes.Notice set the size_average
def CrossEntropy2d(input, target, weight=None, size_average=False):
    # input:(n, c, h, w) target:(n, h, w)
    n, c, h, w = input.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

    target_mask = target >= 0
    target = target[target_mask]
    #loss = F.nll_loss(F.log_softmax(input), target, weight=weight, size_average=False)
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= target_mask.sum().data[0]

    return loss

class encoder(nn.Module):
    '''
        Encoder for the Segmentation network
    '''

    def __init__(self, batchNorm_momentum):
        super(encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
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
        self.lossFunction = CrossEntropyLoss2d()

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)

        return output

    def loss(self, generated, groundtruth):
        loss = self.lossFunction(generated, groundtruth)
        return loss
