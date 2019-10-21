import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

from saliency_model import *
from initialize_dataset import createDataset, createDatasetViolence
import random
from ViolenceDatasetV2 import ViolenceDatasetVideos
from operator import itemgetter
from transforms import createTransforms
from loss import Loss
import argparse
import os
from torchvision.utils import save_image

def normalize(img):
    print('normalize:', img.size() )
    _min = torch.min(img)
    _max = torch.max(img)
    print('min:', _min.item(), ', max:', _max.item())
    return (img - _min) / (_max - _min)

def imshow(img):
    img = img / 2 + 0.5    
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def print_max_min(img):
    _min = torch.min(img)
    _max = torch.max(img)
    print('min:', _min.item(), ', max:', _max.item())

def plot_sample(img1, img2, img3, title, save=False):
    fig2 = plt.figure(figsize=(12, 12))
    fig2.suptitle(title, fontsize=16)
    img1 = img1 / 2 + 0.5    
    npimg1 = img1.numpy()
    ax1 = plt.subplot(311)
    # ax1.set_title(title)
    plt.imshow(np.transpose(npimg1, (1, 2, 0)))

    img2 = img2 / 2 + 0.5  
    # img2 = normalize(img2)  
    npimg2 = img2.numpy()
    ax2 = plt.subplot(312)
    plt.imshow(np.transpose(npimg2, (1, 2, 0)))

    img3 = img3 / 2 + 0.5    
    npimg3 = img3.numpy()
    ax3 = plt.subplot(313)
    plt.imshow(np.transpose(npimg3, (1, 2, 0)))
    plt.show()
    print(title)
    if save:
        fig2.savefig(os.path.join('/media/david/datos/Violence DATA/HockeyFights/saliencyResults',title+'.png'))
    

def subplot(img1, img2, img3, img4, img5, title):
    fig2 = plt.figure(figsize=(12, 12))
    fig2.suptitle(title, fontsize=16)
    img1 = img1 / 2 + 0.5    
    npimg1 = img1.numpy()
    ax1 = plt.subplot(231)
    # ax1.set_title(title)
    plt.imshow(np.transpose(npimg1, (1, 2, 0)))

    img2 = img2 / 2 + 0.5    
    npimg2 = img2.numpy()
    ax2 = plt.subplot(232)
    plt.imshow(np.transpose(npimg2, (1, 2, 0)))

    img3 = img3 / 2 + 0.5    
    npimg3 = img3.numpy()
    ax3 = plt.subplot(233)
    plt.imshow(np.transpose(npimg3, (1, 2, 0)))

    img4 = img4 / 2 + 0.5    
    npimg4 = img4.numpy()
    ax4 = plt.subplot(234)
    plt.imshow(np.transpose(npimg4, (1, 2, 0)))

    img5 = img5 / 2 + 0.5    
    npimg5 = img5.numpy()
    ax5 = plt.subplot(235)
    plt.imshow(np.transpose(npimg5, (1, 2, 0)))
    plt.show()


# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trainset = torchvision.datasets.CIFAR10(root='data/', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=num_workers)

# testset = torchvision.datasets.CIFAR10(root='data/', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=num_workers)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def init(batch_size, num_workers, debugg_mode, numDiPerVideos, dataset_source, data_transforms, interval_duration, avgmaxDuration, shuffle = False):
    hockey_path_violence = "/media/david/datos/Violence DATA/HockeyFights/frames/violence"
    hockey_path_noviolence = "/media/david/datos/Violence DATA/HockeyFights/frames/nonviolence"
    datasetAll, labelsAll, numFramesAll = createDatasetViolence(hockey_path_violence) #ordered
    # combined = list(zip(datasetAll, labelsAll, numFramesAll))
    # random.shuffle(combined)
    # datasetAll[:], labelsAll[:], numFramesAll[:] = zip(*combined) 
    print(len(datasetAll), len(labelsAll), len(numFramesAll))

    image_datasets = {
        # "train": ViolenceDatasetVideos( dataset=datasetAll, labels=labelsAll, spatial_transform=data_transforms["train"], source=dataset_source,
        #     interval_duration=interval_duration,difference=3, maxDuration=avgmaxDuration, nDynamicImages=numDiPerVideos, debugg_mode=debugg_mode, ),
        "test": ViolenceDatasetVideos( dataset=datasetAll, labels=labelsAll, spatial_transform=data_transforms["test"], source=dataset_source,
            interval_duration=interval_duration, difference=3, maxDuration=avgmaxDuration, nDynamicImages=numDiPerVideos, debugg_mode=debugg_mode, )
    }
    dataloaders_dict = {
        # "train": torch.utils.data.DataLoader( image_datasets["train"], batch_size=batch_size, shuffle=False, num_workers=num_workers, ),
        "test": torch.utils.data.DataLoader( image_datasets["test"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, ),
    }
    return datasetAll, image_datasets, dataloaders_dict

def test(saliency_model_file, num_classes, dataloaders_dict, datasetAll, input_size, saliency_config, numDiPerVideos):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = saliency_model(num_classes=num_classes)
    net = net.to(device)
    net = torch.load(saliency_model_file)
    net.eval()
    padding = 20
    # cuda0 = torch.device('cuda:0')
    
    ones = torch.ones(1, input_size, input_size)
    zeros = torch.zeros(1, input_size, input_size)
    ones = ones.to(device)
    zeros = zeros.to(device)
    # plt.figure(1)
    for i, data in enumerate(dataloaders_dict['test'], 0): #inputs, labels:  <class 'torch.Tensor'> torch.Size([3, 3, 224, 224]) <class 'torch.Tensor'> torch.Size([3])
        print('-'*150)
        print('video: ',i,' ',datasetAll[i])
        inputs, labels = data  #dataset load [bs,ndi,c,w,h]
        print('inputs, labels: ',type(inputs),inputs.size(), type(labels), labels.size())
        # print('dataset element: ',inputs_r.shape)
        if numDiPerVideos>1: 
            inputs = inputs.permute(1, 0, 2, 3, 4)
            inputs = torch.squeeze(inputs, 0)  #get one di [bs,c,w,h]
        
        rgb_dinamyc_image = inputs[0].clone()
        print_max_min(inputs)
        rgb_dinamyc_image = rgb_dinamyc_image.cpu()
        # # rgb_dinamyc_image = rgb_dinamyc_image / 2 + 0.5    
        # # rgb_dinamyc_image = rgb_dinamyc_image.numpy()
        # save_image(rgb_dinamyc_image,os.path.join('/media/david/datos/Violence DATA/HockeyFights/di_images',str(i+1)+'.png'))

        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        masks, _ = net(inputs, labels)
        y = torch.where(masks > masks.view(masks.size(0), masks.size(1), -1).mean(2)[:, :, None, None], ones, zeros)
        binary_imgs = torchvision.utils.make_grid(y.cpu().data, padding=padding)
        # print('inputs size:',inputs.size())
        # print('mask size:', masks.size())
        # print('central size', central_fr.size())
        #Original Image
        di_images = torchvision.utils.make_grid(inputs.cpu().data, padding=padding)
        # ct_images = torchvision.utils.make_grid(central_fr.cpu().data, padding=padding)
        #Mask
        # masks = torch.zeros(masks.size(),device=cuda0)
        # masks = normalize(masks)
        mask_images = torchvision.utils.make_grid(masks.cpu().data, padding=padding)
        #Image Segmented
        # segmented = torchvision.utils.make_grid((inputs*masks).cpu().data, padding=padding)
        
        # segmented_central = torchvision.utils.make_grid((central_fr * y).cpu().data, padding=padding)
        # segmented = torchvision.utils.make_grid((inputs * y).cpu().data, padding=padding)  #apply binary mask
        
        segmented = torchvision.utils.make_grid((inputs * masks).cpu().data, padding=padding)  #apply soft mask
        # save_image((inputs * y).cpu(),os.path.join('saliencyResults',str(i+1)+'.png'))
        # imshow(ct_images)
        # subplot(ct_images, di_images, mask_images, segmented, segmented_central, datasetAll[i])
        # mask_images = normalize(mask_images)
        plot_sample(di_images, mask_images, segmented, 'video_'+str(i+1)+'_'+saliency_config)

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saliencyModelFile", type=str)
    parser.add_argument("--batchSize", type=int, default=3)
    # parser.add_argument("--numEpochs", type=int, default=10)
    parser.add_argument("--numWorkers", type=int, default=1)
    args = parser.parse_args()
    interval_duration = 0
    avgmaxDuration = 0
    numDiPerVideos = 1
    input_size = 224
    num_classes = 2
    data_transforms = createTransforms(input_size)
    dataset_source = 'frames'
    debugg_mode = False
    batch_size = args.batchSize
    num_workers = args.numWorkers
    saliency_model_file = args.saliencyModelFile
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    numDiPerVideos = 1

    saliency_model_file = args.saliencyModelFile
    saliency_model_config = saliency_model_file
    saliency_model_file = os.path.join('saliencyModels',saliency_model_file)
    datasetAll, image_datasets, dataloaders_dict = init(batch_size, num_workers, debugg_mode, numDiPerVideos, dataset_source, data_transforms, interval_duration, avgmaxDuration)
    test(saliency_model_file, num_classes, dataloaders_dict, datasetAll, input_size, saliency_model_config, numDiPerVideos)

__main__()
    
    
   