'''
Image Reconstruction using SegNet
Code adapted from: https://github.com/chengyangfu/pytorch-vgg-cifar10
'''

import argparse
import os
import shutil
import time

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from segnet import SegNet
from miccaiDataLoader import miccaiDataset

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
            help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
            help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
            help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N',
            help='mini-batch size (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
            metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
            help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
            metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int, metavar='N',
            help='print frequency (default:1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
            help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
            help='evaluate model on validation set')
parser.add_argument('--pre-trained', dest='pretrained', action='store_true',
            help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
            help='use half-precision(16-bit)')
parser.add_argument('--save-dir', dest='save_dir',
            help='The directory used to save the trained models',
            default='save_temp', type=str)

best_prec1 = np.inf
use_gpu = torch.cuda.is_available()

def main():
    global args, best_prec1
    args = parser.parse_args()

    # Check if the save directory exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = SegNet(3, 3)

    #model.features = torch.nn.DataParallel(model.features)
    if use_gpu:
        model.cuda()

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.Scale(256),
    #         transforms.RandomSizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    #     ]),
    #     'val': transforms.Compose([
    #         transforms.Scale(256),
    #         transforms.CenterCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    #     ]),
    # }

    data_transforms = {
        'train': transforms.Compose([
            transforms.Scale((224, 224)),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Scale((224, 224)),
            transforms.ToTensor(),
        ]),
    }

    data_dir = '/media/salman/DATA/NUST/MS RIME/Thesis/MICCAI Dataset/miccai_all_images'

    image_datasets = {x: miccaiDataset(os.path.join(data_dir, x), data_transforms[x])
                    for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=args.workers)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # Define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    #optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    if args.evaluate:
        validate(dataloaders['val'], model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        #adjust_learning_rate(optimizer, epoch)

        # Train for one epoch
        train(dataloaders['train'], model, criterion, optimizer, epoch)

        # Evaulate on validation set
        prec1 = validate(dataloaders['val'], model, criterion)
        prec1 = prec1.cpu().data.numpy()

        # Remember best prec1 and save checkpoint
        print(prec1)
        print(best_prec1)
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            #'optimizer': optimizer.state_dict(),
        }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))

def train(train_loader, model, criterion, optimizer, epoch):
    '''
        Run one training epoch
    '''

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to train mode
    model.train()

    end = time.time()

    for i, input in enumerate(train_loader):
        # Measure Data loading time
        data_time.update(time.time() - end)
        target = input.cuda()
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)
        if args.half:
            input_var = input_var.half()

        # Compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        if i % args.print_freq == 0:
            displaySamples(target_var, output)

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        # Measure accuracy and record loss
        #prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        #top1.update(prec1[0], input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))

def validate(val_loader, model, criterion):
    '''
        Run evaluation
    '''

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    for i, input in enumerate(val_loader):
        target = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True)
        if args.half:
            input_var = input_var.half()

        # Compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        if i % args.print_freq == 0:
            displaySamples(target_var, output)

        # Measure accuracy and record loss
        #prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        #top1.update(prec1[0], input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      i, len(val_loader), batch_time=batch_time,
                      loss=losses))

    return loss

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    '''
        Save the training model
    '''
    torch.save(state, filename)

class AverageMeter(object):
    '''
        Computes and stores the average and current value
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    '''
        Sets the learning rate to the initial LR decayed by a factor of 10
        every 30 epochs
    '''

    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    '''
        Computes the precision@k for the specified values of k
    '''

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def displaySamples(input, output):
    ''' Display the original and the reconstructed image.
        If a batch is used, it displays only the first image in the batch.

        Args:
            input image, output image
    '''
    if use_gpu:
        input = input.cpu()
        output = output.cpu()

    unNorm = UnNormalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

    input = input.data.numpy()
    input = np.transpose(np.squeeze(input[0,:,:,:]), (1,2,0))
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    #input = unNorm(input)

    output = output.data.numpy()
    output = np.transpose(np.squeeze(output[0,:,:,:]), (1,2,0))
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    #output = unNorm(output)

    cv2.namedWindow('Input Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Reconstructed Image', cv2.WINDOW_NORMAL)

    cv2.imshow('Input Image', input)
    cv2.imshow('Reconstructed Image', output)
    cv2.waitKey(1)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        """
        Args:
            image (Image): Numpy ndarray of size (H, W, C) to be normalized.
        Returns:
            Numpy ndarray: Normalized image.
        """
        im = np.zeros_like(image)
        for i in range(image.shape[2]):
            im[i,:,:] = np.maximum(np.zeros_like(image[i,:,:]),
            np.minimum(np.ones_like(image[i,:,:]),
            (image[i,:,:] * self.mean[i]) + self.std[i]))
            # The normalize code -> t.sub_(m).div_(s)
        return im

if __name__ == '__main__':
    main()
