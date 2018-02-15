'''
Utilites for data visualization and manipulation.
'''

import torch
import numpy as np
import cv2
import math
import os

def displaySamples(img, generated, gt, use_gpu, key, save, epoch, imageNum,
    save_dir):
    ''' Display the original, generated, and the groundtruth image.
        If a batch is used, it displays only the first image in the batch.

        Args:
            input image, output image, groundtruth segmentation,
            use_gpu, class-wise key, save or not?, epoch, image number,
            save directory
    '''

    if use_gpu:
        img = img.cpu()
        generated = generated.cpu()

    gt = gt.numpy()
    gt = np.transpose(np.squeeze(gt[0,:,:,:]), (1,2,0))
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

    generated = generated.data.numpy()
    generated = reverseOneHot(generated, key)
    generated = np.squeeze(generated[0,:,:,:]).astype(np.uint8)
    generated = cv2.cvtColor(generated, cv2.COLOR_BGR2RGB) / 255

    img = img.data.numpy()
    img = np.transpose(np.squeeze(img[0,:,:,:]), (1,2,0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    stacked = np.concatenate((img, generated, gt), axis = 1)

    if save:
        file_name = 'epoch_%d_img_%d.png' %(epoch, imageNum)
        save_path = os.path.join(save_dir, file_name)
        cv2.imwrite(save_path, stacked*255)

    cv2.namedWindow('Input | Gen | GT' %(epoch, imageNum),
                    cv2.WINDOW_NORMAL)
    cv2.imshow('Input | Gen | GT', stacked)

    cv2.waitKey(1)

def disentangleKey(key):
    '''
        Disentangles the key for class and labels obtained from the
        JSON file
        Returns a python dictionary of the form:
            {Class Id: RGB Color Code as numpy array}
    '''
    dKey = {}
    for i in range(len(key)):
        class_id = int(key[i]['id'])
        c = key[i]['color']
        c = c.split(',')
        c0 = int(c[0][1:])
        c1 = int(c[1])
        c2 = int(c[2][:-1])
        color_array = np.asarray([c0,c1,c2])
        dKey[class_id] = color_array

    return dKey

def generateLabel4CE(gt, key):
    '''
        Generates the label for Cross Entropy Loss from a batch of groundtruth
        segmentation images.
    '''

    batch = gt.numpy()
    # Iterate over all images in a batch
    for i in range(len(batch)):
        img = batch[i,:,:,:]
        img = np.transpose(img, (1,2,0))
        catMask = np.zeros((img.shape[0], img.shape[1]))

        # Iterate over all the key-value pairs in the class Key dict
        for k in range(len(key)):
            rgb = key[k]
            mask = np.where(np.all(img == rgb, axis = 2))
            catMask[mask] = k

        catMaskTensor = torch.from_numpy(catMask).unsqueeze(0)
        if 'label' in locals():
            label = torch.cat((label, catMaskTensor), 0)
        else:
            label = catMaskTensor

    return label.long()

def reverseOneHot(batch, key):
    '''
        Generates the segmented image from the output of a segmentation network.
        Takes a batch of numpy oneHot encoded tensors and returns a batch of
        numpy images in RGB (not BGR).
    '''

    # Iterate over all images in a batch
    for i in range(len(batch)):
        vec = batch[i,:,:,:]
        idxs = np.argmax(vec, axis=0)

        segSingle = np.zeros([idxs.shape[0], idxs.shape[1], 3])

        # Iterate over all the key-value pairs in the class Key dict
        for k in range(len(key)):
            rgb = key[k]
            mask = idxs == k
            #mask = np.where(np.all(idxs == k, axis=-1))
            segSingle[mask] = rgb

        # print(segSingle[120:130,120:130,:])
        segMask = np.expand_dims(segSingle, axis=0)
        if 'generated' in locals():
            generated = np.concatenate((generated, segMask), axis=0)
        else:
            generated = segMask

    return generated

def generatePresenceVector(batch, key):
    '''
        Generate a vector with dimensions of classes equal to the number of
        classes. Each elements corresponds to the presence of a particular
        class in the image: It is the fraction of pixels a particular category
        in an image, and is 0 if the class is absent from that image.
    '''
    batch = batch.numpy()
    # Iterate over all images in a batch
    for i in range(len(batch)):
        img = batch[i,:,:,:]
        imgSize = img.shape[1] * img.shape[2]
        img = np.transpose(img, (1,2,0))
        presence = np.zeros(len(key) + 1) # +1 for the background class

        # Iterate over all the key-value pairs in the class Key dict
        for k in range(len(key)):
            rgb = key[k]
            mask = np.where(np.all(img == rgb, axis = -1))
            presence[k] = len(mask[0])/imgSize

        # Check for background pixels [0,0,0]
        rgb = np.array([0,0,0])
        mask = np.where(np.all(img == rgb, axis = -1))
        presence[19] = len(mask[0])/imgSize

        presence = torch.from_numpy(presence).unsqueeze(0)

        if 'label' in locals():
            label = torch.cat((label, presence), 0)
        else:
            label = presence

    return label

def generateOneHot(gt, key):
    '''
        Generates the one-hot encoded tensor for a batch of images based on
        their class.
    '''

    batch = gt.numpy()
    # Iterate over all images in a batch
    for i in range(len(batch)):
        img = batch[i,:,:,:]
        img = np.transpose(img, (1,2,0))
        catMask = np.zeros((img.shape[0], img.shape[1]))

        # Iterate over all the key-value pairs in the class Key dict
        for k in range(len(key)):
            catMask = catMask * 0
            rgb = key[k]
            mask = np.where(np.all(img == rgb, axis = -1))
            catMask[mask] = 1

            catMaskTensor = torch.from_numpy(catMask).unsqueeze(0)
            if 'oneHot' in locals():
                oneHot = torch.cat((oneHot, catMaskTensor), 0)
            else:
                oneHot = catMaskTensor

    label = oneHot.view(len(batch),len(key),img.shape[0],img.shape[1])
    return label

def generateGTmask(batch, key):
    '''
        Generates the category-wise encoded vector for the segmentation classes
        for a batch of images.
        Returns a tensor of size: [batchSize, imgSize**2, 1]
    '''
    batch = batch.numpy()
    # Iterate over all images in a batch
    for i in range(len(batch)):
        img = batch[i,:,:,:]
        img = np.transpose(img, (1,2,0))
        cat_mask = np.ones((img.shape[0], img.shape[1]))
        # Multiply by 19 since 19 is considered label for the background class
        cat_mask = cat_mask * 19

        # Iterate over all the key-value pairs in the class Key dict
        for k in range(len(key)):
            rgb = key[k]
            mask = np.where(np.all(img == rgb, axis = -1))
            cat_mask[mask] = k

        cat_mask = torch.from_numpy(cat_mask).view(-1,1).unsqueeze(0)

        if 'label' in locals():
            label = torch.cat((label, cat_mask), 0)
        else:
            label = cat_mask
        #print('img copy masked')
        #print(img_copy)

    label = torch.squeeze(label, dim=2)
    return label

def labelToImage(label, key):
    '''
        Generates the image from the output label.
        Basically the inverse process of the generateGTmask function.
    '''

    img_dim = int(math.sqrt(label.shape[1]))
    label = label[0,:]
    label = np.around(label).astype(int)
    #print(label)
    #print(np.min(label))
    gen = np.ones((label.shape[0], 3)) * 255

    for k in range(len(key) + 1):
        if k == 19:
            rgb = [0, 0, 0]
        else:
            rgb = key[k]
        mask = label == k
        gen[mask] = rgb

    # print(gen)

    gen = np.reshape(gen, (img_dim, img_dim, 3))

    return gen

#################### Reconstruction Utilities ######################

def generateLabels4ReconCE(batch):
    '''
        Generates the label for Cross Entropy Loss from a batch of images.
        Also separates into the three RGB channels (for channel-wise loss).

        Input: PyTorch Tensor
        Output: 3 x PyTorch Tensors corrsponding to the RGB channels
    '''

    # Iterate over all images in a batch
    for i in range(len(batch)):
        img = batch[i,:,:,:]

        R = img[0,:,:]
        G = img[1,:,:]
        B = img[2,:,:]

        if 'R_label' in locals():
            R_label = torch.cat((R_label, R), 0)
        else:
            R_label = R
        if 'G_label' in locals():
            G_label = torch.cat((G_label, G), 0)
        else:
            G_label = G
        if 'B_label' in locals():
            B_label = torch.cat((B_label, B), 0)
        else:
            B_label = B

    return R_label.long(), G_label.long(), B_label.long()

def displayReconSamples(img, R_gen, G_gen, B_gen, use_gpu):
    ''' Display the original and the reconstructed image.
        If a batch is used, it displays only the first image in the batch.

        Args:
            input image, R-channel output, G-channel output, B-channel output,
            use_gpu
    '''

    if use_gpu:
        img = img.cpu()
        R_gen = R_gen.cpu()
        G_gen = G_gen.cpu()
        B_gen = B_gen.cpu()

    R_gen = R_gen.data.numpy()
    G_gen = G_gen.data.numpy()
    B_gen = B_gen.data.numpy()
    generated = reverseReconOneHot(R_gen, G_gen, B_gen)
    generated = np.squeeze(generated[0,:,:,:]).astype(np.uint8)
    generated = cv2.cvtColor(generated, cv2.COLOR_BGR2RGB) / 255

    img = img.data.numpy()
    img = np.transpose(np.squeeze(img[0,:,:,:]), (1,2,0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    stacked = np.concatenate((img, generated), axis = 1)

    cv2.namedWindow('Input | Generated', cv2.WINDOW_NORMAL)
    cv2.imshow('Input | Generated', stacked)

    cv2.waitKey(1)

def reverseReconOneHot(R_batch, G_batch, B_batch):
    '''
        Generates the reconstructed image from the output of a Reconstruction
        network.
        Takes a batch of numpy one-hot tensors on individual RGB channels
        and returns a batch of numpy images in RGB (not BGR).
    '''
    batchSize = len(R_batch)
    # Iterate over all images in a batch
    for i in range(batchSize):
        R = np.expand_dims(np.argmax(R_batch[i,:,:], axis=0), axis=0)
        G = np.expand_dims(np.argmax(G_batch[i,:,:], axis=0), axis=0)
        B = np.expand_dims(np.argmax(B_batch[i,:,:], axis=0), axis=0)

        concatenated = np.concatenate((R, G, B), axis=0)
        if 'img' in locals():
            img = np.concatenate((img, concatenated), axis=0)
        else:
            img = concatenated

    return img
