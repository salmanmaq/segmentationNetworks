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
from model.reconNet import ReconNet
from datasets.miccaiDataLoader import miccaiDataset

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

best_prec1 = np.inf
use_gpu = torch.cuda.is_available()

def main():
    global args, best_prec1
    args = parser.parse_args()

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
            #transforms.TenCrop(args.resizedImageSize),
            #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
            #transforms.RandomResizedCrop(224, interpolation=Image.NEAREST),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize((args.imageSize, args.imageSize), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ]),
    }

    # Data Loading
    data_dir = '/media/salman/DATA/NUST/MS RIME/Thesis/MICCAI Dataset/m2cai16-tool/train_dataset'

    image_datasets = {x: miccaiDataset(os.path.join(data_dir, x), data_transforms[x])
                        for x in ['train', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=args.batchSize,
                                                  shuffle=True,
                                                  num_workers=args.workers)
                  for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    # Initialize the model
    model = ReconNet(args.bnMomentum)

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
        train(dataloaders['train'], model, criterion, optimizer, epoch)

        # Evaulate on validation set

        print('>>>>>>>>>>>>>>>>>>>>>>>Testing<<<<<<<<<<<<<<<<<<<<<<<')
        prec1 = validate(dataloaders['test'], model, criterion, epoch)
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

def train(train_loader, model, criterion, optimizer, epoch):
    '''
        Run one training epoch
    '''

    # Switch to train mode
    model.train()

    for i, img in enumerate(train_loader):

        # For TenCrop Data Augmentation
        #img = img.view(args.batchSize*10,3,args.resizedImageSize,args.resizedImageSize)

        # Process the network inputs and outputs
        img_temp = img * 255
        R_label, G_label, B_label = utils.generateLabels4ReconCE(img_temp)

        img = Variable(img)
        R_label = Variable(R_label)
        G_label = Variable(G_label)
        B_label = Variable(B_label)

        if use_gpu:
            img = img.cuda()
            R_label = R_label.cuda()
            G_label = G_label.cuda()
            B_label = B_label.cuda()

        # Compute output
        R_gen, G_gen, B_gen = model(img)
        R_loss = criterion(R_gen, R_label)
        G_loss = criterion(G_gen, G_label)
        B_loss = criterion(B_gen, B_label)

        loss = R_loss + G_loss + B_loss
        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('[%d/%d][%d/%d] R-Loss: %.4f | G-Loss: %.4f | B-Loss: %.4f | Total-Loss: %.4f'
              % (epoch, args.epochs-1, i, len(train_loader)-1, R_loss.mean().data[0],
              G_loss.mean().data[0], B_loss.mean().data[0], loss.mean().data[0]))

        utils.displayReconSamples(img, R_gen, G_gen, B_gen, use_gpu)

def validate(val_loader, model, criterion, epoch, key):
    '''
        Run evaluation
    '''

    # Switch to evaluate mode
    model.eval()

    for i, img in enumerate(val_loader):

        # For TenCrop Data Augmentation
        img = img.view(args.batchSize*10,3,args.resizedImageSize,args.resizedImageSize)

        # Process the network inputs and outputs
        img_temp = img * 255
        R_label, G_label, B_label = utils.generateLabels4ReconCE(img_temp)

        img = Variable(img)
        R_label = Variable(R_label)
        G_label = Variable(G_label)
        B_label = Variable(B_label)

        if use_gpu:
            img = img.cuda()
            R_label = R_label.cuda()
            G_label = G_label.cuda()
            B_label = B_label.cuda()

        # Compute output
        R_gen, G_gen, B_gen = model(img)
        R_loss = criterion(R_gen, R_label)
        G_loss = criterion(G_gen, G_label)
        B_loss = criterion(B_gen, B_label)

        loss = R_loss + G_loss + B_loss

        print('[%d/%d][%d/%d] R-Loss: %.4f | G-Loss: %.4f | B-Loss: %.4f | Total-Loss: %.4f'
              % (epoch, args.epochs-1, i, len(val_loader)-1, R_loss.mean().data[0],
              G_loss.mean().data[0], B_loss.mean().data[0], loss.mean().data[0]))

        utils.displayReconSamples(img, R_gen, G_gen, B_gen, use_gpu)

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
