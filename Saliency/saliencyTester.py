
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import saliencyModel
import random
from operator import itemgetter
from loss import Loss
import os
from torchvision.utils import save_image
import constants


def test(saliency_model_file, num_classes, dataloader, datasetAll, input_size, saliency_config, numDiPerVideos):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = saliencyModel.build_saliency_model(num_classes=num_classes)
    net = net.to(device)
    net = torch.load(saliency_model_file)
    net.eval()
    padding = 10
    # cuda0 = torch.device('cuda:0')
    
    ones = torch.ones(1, input_size, input_size)
    zeros = torch.zeros(1, input_size, input_size)
    ones = ones.to(device)
    zeros = zeros.to(device)
    # plt.figure(1)
    for i, data in enumerate(dataloader, 0): #inputs, labels:  <class 'torch.Tensor'> torch.Size([3, 3, 224, 224]) <class 'torch.Tensor'> torch.Size([3])
        print('-'*150)
        # print('video: ',i,' ',datasetAll[i])
        inputs, labels, video_name = data  #dataset load [bs,ndi,c,w,h]
        print('inputs, labels: ', type(inputs), inputs.size(), type(labels), labels.size())
        if numDiPerVideos > 1:
            inputs = torch.squeeze(inputs, 0)  #get one di [ndi,c,w,h]

        # if numDiPerVideos>1: 
        #     inputs = inputs.permute(1, 0, 2, 3, 4)
        #     inputs = torch.squeeze(inputs, 0)  #get one di [bs,c,w,h]
        
        # rgb_dinamyc_image = inputs[0].clone()
        # print_max_min(inputs)
        # rgb_dinamyc_image = rgb_dinamyc_image.cpu()
        # # save_image(rgb_dinamyc_image,os.path.join('/media/david/datos/Violence DATA/HockeyFights/di_images',str(i+1)+'.png'))

        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        masks, _ = net(inputs, labels)
        # y = torch.where(masks > masks.view(masks.size(0), masks.size(1), -1).mean(2)[:, :, None, None], ones, zeros)
        # binary_imgs = torchvision.utils.make_grid(y.cpu().data, padding=padding)
        # # print('inputs size:',inputs.size())
        # # print('mask size:', masks.size())
        # # print('central size', central_fr.size())
        # #Original Image
        # Tensor2Numpy(masks.clone())
        di_images = torchvision.utils.make_grid(inputs.cpu().data, padding=padding)
        # # ct_images = torchvision.utils.make_grid(central_fr.cpu().data, padding=padding)
        # #Mask
        mask_images = torchvision.utils.make_grid(masks.cpu().data, padding=padding)
        # #Image Segmented
        # # segmented = torchvision.utils.make_grid((inputs*masks).cpu().data, padding=padding)
        # # segmented_central = torchvision.utils.make_grid((central_fr * y).cpu().data, padding=padding)
        # # segmented = torchvision.utils.make_grid((inputs * y).cpu().data, padding=padding)  #apply binary mask
        
        segmented = torchvision.utils.make_grid((inputs * masks).cpu().data, padding=padding)  #apply soft mask
        # # save_image((inputs * y).cpu(),os.path.join('saliencyResults',str(i+1)+'.png'))
        # # imshow(ct_images)
        # # subplot(ct_images, di_images, mask_images, segmented, segmented_central, datasetAll[i])
        # # mask_images = normalize(mask_images)
        plot_sample(di_images, mask_images, segmented, video_name)



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
    print('tensor: ', type(img1), img1.size())

    fig2 = plt.figure(figsize=(12, 12))
    fig2.suptitle(title, fontsize=16)
    img1 = img1 / 2 + 0.5    
    npimg1 = img1.numpy()
    ax1 = plt.subplot(3,1,1)
    # ax1.set_title(title)
    plt.imshow(np.transpose(npimg1, (1, 2, 0)))

    img2 = img2 / 2 + 0.5  
    # img2 = normalize(img2)  
    npimg2 = img2.numpy()
    ax2 = plt.subplot(3,1,2)
    plt.imshow(np.transpose(npimg2, (1, 2, 0)))

    img3 = img3 / 2 + 0.5    
    npimg3 = img3.numpy()
    ax3 = plt.subplot(3,1,3)
    plt.imshow(np.transpose(npimg3, (1, 2, 0)))
    plt.show()
    print(title)
    if save:
        fig2.savefig(os.path.join('/media/david/datos/Violence DATA/HockeyFights/saliencyResults',title+'.png'))


def grid_plot(images, rows, cols):
    fig2 = plt.figure(figsize=(12, 12))
    # fig2.suptitle(title, fontsize=16)
    for i in range(1, len(images)+1):
        ax1 = plt.subplot(rows,cols,i)
        plt.imshow(images[i-1])
    plt.show()


def Tensor2Numpy(tensor):
    tensor = tensor.cpu().data
    tensor = tensor / 2 + 0.5
    tensor = tensor.numpy()
    tensor = np.squeeze(tensor)
    print('Tensor2Numpy: ', type(tensor), tensor.shape)
    grid_plot(tensor,1,3)

    
    
   