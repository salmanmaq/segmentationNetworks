"""
This is not an implementation for
SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation

I'll change the file names later.

CNN architeture for combined segmentation and classification.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder(nn.Module):
    '''
        Encoder for the Segmentation plus Classification Network
    '''

    def __init__(self, batchNorm_momentum):
        super(encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, dilation=1, bias=True),
            # nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, 2, 1, dilation=1, bias=True),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.ReLU(True),

            nn.Conv2d(128, 256, 4, 2, 1, dilation=1, bias=True),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(True),

            nn.Conv2d(256, 512, 4, 2, 1, dilation=1, bias=True),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(True),

            nn.Conv2d(512, 1024, 4, 1, 0, dilation=1, bias=True),
            nn.BatchNorm2d(1024, momentum=batchNorm_momentum),
            nn.ReLU(True)
        )

    def forward(self, input):
        output = self.main(input)
        return output

class decoder(nn.Module):
    '''
        Decoder for the Segmentation and Classification Network
    '''

    def __init__(self, batchNorm_momentum, img_size, num_classes=19):
        super(decoder, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.Dropout2d(),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.Dropout2d(),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.Dropout2d(),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, num_classes, 4, 2, 1, bias=False)
        )

        self.classifier = nn.Conv2d(num_classes, 7, img_size, bias=True)
        self.smax = nn.Softmax(dim=1)

    def forward(self, input):
        seq_output = self.main(input)
        classified = self.classifier(seq_output)
        segmented = self.smax(seq_output)
        return classified, segmented

class segnetPlusClass(nn.Module):
    '''
        Segnet network
    '''

    def __init__(self, batchNorm_momentum, img_size, num_classes):
        super(segnetPlusClass, self).__init__()
        self.encoder = encoder(batchNorm_momentum)
        self.decoder = decoder(batchNorm_momentum, img_size, num_classes)

    def forward(self, x):
        latent = self.encoder(x)
        classified, segmented = self.decoder(latent)

        return classified, segmented

    def dice_loss(self, output, target, weights=None, ignore_index=None):
        '''
            output : NxCxHxW Variable
            target :  NxHxW LongTensor
            weights : C FloatTensor
            ignore_index : int index to ignore from loss
        '''
        eps = 0.0001

        encoded_target = output.detach() * 0
        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)
