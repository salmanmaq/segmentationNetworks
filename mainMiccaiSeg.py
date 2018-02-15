'''
Image Segmentation using SegNet
'''

import argparse
import os
import shutil

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
import torchvision.transforms as transforms

import utils
from model.segnet import SegNet
from datasets.miccaiSegDataLoader import miccaiSegDataset

parser = argparse.ArgumentParser(description='PyTorch SegNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
            help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
            help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
            help='manual epoch number (useful on restarts)')
parser.add_argument('--batchSize', default=4, type=int,
            help='Mini-batch size (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
            metavar='LR', help='initial learning rate')
parser.add_argument('--bnMomentum', default=0.1, type=float,
            help='Batch Norm Momentum (default: 0.1)')
parser.add_argument('--imageSize', default=256, type=int,
            help='height/width of the input image to the network')
parser.add_argument('--resizedImageSize', default=224, type=int,
            help='height/width of the resized image to the network')
parser.add_argument('--print-freq', '-p', default=1, type=int, metavar='N',
            help='print frequency (default:1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
            help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
            help='evaluate model on validation set')
parser.add_argument('--pre-trained', dest='pretrained', action='store_true',
            help='use pre-trained model')
parser.add_argument('--save-dir', dest='save_dir',
            help='The directory used to save the trained models',
            default='save_temp', type=str)
parser.add_argument('--saveTest', default='False', type=str,
            help='Saves the validation/test images if True')

best_prec1 = np.inf
use_gpu = torch.cuda.is_available()

def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.saveTest == 'True':
        args.saveTest = True
    elif args.saveTest == 'False':
        args.saveTest = False

    # Check if the save directory exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

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

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((args.imageSize, args.imageSize), interpolation=Image.NEAREST),
            transforms.TenCrop(args.resizedImageSize),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
            #transforms.RandomResizedCrop(224, interpolation=Image.NEAREST),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            #transforms.ToTensor(),
        ]),
        'trainval': transforms.Compose([
            transforms.Resize((args.imageSize, args.imageSize), interpolation=Image.NEAREST),
            transforms.TenCrop(args.resizedImageSize),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
            #transforms.RandomResizedCrop(224, interpolation=Image.NEAREST),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            #transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize((args.imageSize, args.imageSize), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ]),
    }

    # Data Loading
    data_dir = '/home/salman/pytorch/segmentationNetworks/datasets/miccaiSegRefined'
    # json path for class definitions
    json_path = '/home/salman/pytorch/segmentationNetworks/datasets/miccaiSegClasses.json'

    image_datasets = {x: miccaiSegDataset(os.path.join(data_dir, x), data_transforms[x],
                        json_path) for x in ['train', 'trainval', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=args.batchSize,
                                                  shuffle=True,
                                                  num_workers=args.workers)
                  for x in ['train', 'trainval', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'trainval', 'test']}

    # Get the dictionary for the id and RGB value pairs for the dataset
    classes = image_datasets['train'].classes
    key = utils.disentangleKey(classes)
    num_classes = len(key)

    # Initialize the model
    model = SegNet(args.bnMomentum, classes)

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    if use_gpu:
        model.cuda()
        criterion.cuda()

    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    if args.evaluate:
        validate(dataloaders['test'], model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        #adjust_learning_rate(optimizer, epoch)

        # Train for one epoch
        print('>>>>>>>>>>>>>>>>>>>>>>>Training<<<<<<<<<<<<<<<<<<<<<<<')
        train(dataloaders['train'], model, criterion, optimizer, epoch, key)

        # Evaulate on validation set

        print('>>>>>>>>>>>>>>>>>>>>>>>Testing<<<<<<<<<<<<<<<<<<<<<<<')
        prec1 = validate(dataloaders['test'], model, criterion, epoch, key)
        # prec1 = prec1.cpu().data.numpy()
        #
        # # Remember best prec1 and save checkpoint
        # print(prec1)
        # print(best_prec1)
        # is_best = prec1 < best_prec1
        # best_prec1 = min(prec1, best_prec1)
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'best_prec1': best_prec1,
        #     #'optimizer': optimizer.state_dict(),
        # }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))

def train(train_loader, model, criterion, optimizer, epoch, key):
    '''
        Run one training epoch
    '''

    # Switch to train mode
    model.train()

    for i, (img, gt) in enumerate(train_loader):

        # For TenCrop Data Augmentation
        img = img.view(args.batchSize*10,3,args.resizedImageSize,args.resizedImageSize)
        gt = gt.view(args.batchSize*10,3,args.resizedImageSize,args.resizedImageSize)

        # Process the network inputs and outputs
        gt_temp = gt * 255
        label = utils.generateLabel4CE(gt_temp, key)

        img, label = Variable(img), Variable(label)

        if use_gpu:
            img = img.cuda()
            label = label.cuda()

        # Compute output
        seg = model(img)
        loss = criterion(seg, label)

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('[%d/%d][%d/%d] Loss: %.4f'
              % (epoch, args.epochs-1, i, len(train_loader)-1, loss.mean().data[0]))

        utils.displaySamples(img, seg, gt, use_gpu, key, False, epoch,
                             args.save_dir)


def validate(val_loader, model, criterion, epoch, key):
    '''
        Run evaluation
    '''

    # Switch to evaluate mode
    model.eval()

    for i, (img, gt) in enumerate(val_loader):

        # Process the network inputs and outputs
        gt_temp = gt * 255
        label = utils.generateLabel4CE(gt_temp, key)

        img, label = Variable(img), Variable(label)

        if use_gpu:
            img = img.cuda()
            label = label.cuda()

        # Compute output
        seg = model(img)
        #loss = model.loss(seg, label)
        loss = criterion(seg, label)

        print('[%d/%d][%d/%d] Loss: %.4f'
              % (epoch, args.epochs-1, i, len(val_loader)-1, loss.mean().data[0]))

        utils.displaySamples(img, seg, gt, use_gpu, key, args.saveTest, epoch,
                             args.save_dir)

    return loss

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    '''
        Save the training model
    '''
    torch.save(state, filename)

def adjust_learning_rate(optimizer, epoch):
    '''
        Sets the learning rate to the initial LR decayed by a factor of 10
        every 30 epochs
    '''

    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
