'''
Utilites for data visualization and manipulation.
'''

import torch
import numpy as np
import cv2
import math

def displaySamples(img, generated, gt, use_gpu, key):
    ''' Display the original and the reconstructed image.
        If a batch is used, it displays only the first image in the batch.

        Args:
            input image, output image, groundtruth segmentation,
            use_gpu, class-wise key
    '''

    # OH = generateOneHot(gt, key)
    # revOH = reverseOneHot(OH, key)

    if use_gpu:
        img = img.cpu()
        generated = generated.cpu()
        # gt = gt.cpu()

    #unNorm = UnNormalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

    # seg_mask = seg_mask.numpy()
    # seg_mask = seg_mask[0,:,:,:]
    # if seg_mask.shape[2] == 1:
    #     real_depth = real.shape[1]
    #     real_height = real.shape[2]
    #     real_width = real.shape[3]
    #     seg_mask = np.reshape(seg_mask, (real_depth, real_height, real_width))
    # seg_mask = np.transpose(seg_mask, (1,2,0))
    # seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2RGB)

    gt = gt.numpy()
    gt = np.transpose(np.squeeze(gt[0,:,:,:]), (1,2,0))
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

    #generated = generated.data.numpy()
    #generated = labelToImage(generated, key)
    #generated = generated * 255

    generated = generated.data.numpy()
    generated = reverseOneHot(generated, key)
    generated = np.squeeze(generated[0,:,:,:]).astype(np.uint8)
    generated = cv2.cvtColor(generated, cv2.COLOR_BGR2RGB) / 255

    img = img.data.numpy()
    img = np.transpose(np.squeeze(img[0,:,:,:]), (1,2,0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #real = unNorm(real)

    # revOH = np.squeeze(revOH[0,:,:,:]) / 255

    stacked = np.concatenate((img, generated, gt), axis = 1)

    cv2.namedWindow('Input | Gen | GT', cv2.WINDOW_NORMAL)
    cv2.imshow('Input | Gen | GT', stacked)

    # cv2.namedWindow('Real Image', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Reconstructed Image', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Reconstructed Image', cv2.WINDOW_NORMAL)
    #
    # cv2.imshow('Real Image', real)
    # cv2.imshow('Reconstructed Image', output)
    # cv2.imshow('Segmentation Mask', seg_mask)

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
        Takes a batch of tensors and returns a batch of numpy images in RGB (not BGR).
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
        class in the image: It is 1 if a certain class is present, or 0 if it
        is absent.
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
