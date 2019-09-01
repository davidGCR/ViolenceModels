import torch 
import torchvision

import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import os 
import glob 
import cv2 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models 
import torch 
from torchvision import transforms 
from PIL import Image 
from torch.autograd import Variable 
# from tensorboardcolab import TensorBoardColab 
import time
from torch.optim import lr_scheduler
import numpy as np

import sys
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/MyCode')
from AlexNet import *
from ViolenceDatasetV2 import *
from trainer import *
from kfolds import *
from operator import itemgetter
import random
from initializeModel import *
from util import *
from  verifyParameters import *


# #Create dataset
# path_violence = '/content/drive/My Drive/VIOLENCE DATASETS/HockeyFightsFrames/Fights'
# path_noviolence = '/content/drive/My Drive/VIOLENCE DATASETS/HockeyFightsFrames/noFights'
# datasetAll, labelsAll, numFramesAll = createDataset(path_violence,path_noviolence)
# combined = list(zip(datasetAll, labelsAll, numFramesAll))
# random.shuffle(combined)
# datasetAll[:], labelsAll[:], numFramesAll[:] = zip(*combined)
# print(len(datasetAll), len(labelsAll), len(numFramesAll))

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Setup the loss fxn
criterion = nn.CrossEntropyLoss()
input_size = 224

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# tb = TensorBoardColab()

train_lost = []
train_acc = []
test_lost = []
test_acc = []

foldidx = 0
best_acc_test = 0.0
avgmaxDuration = 1.66


modelType = 'alexnetv1'
interval_duration = 0.3
numDiPerVideos = 2
dataset_source = 'frames'
debugg_mode = False
num_workers = 4
batch_size = 64
num_epochs = 15
feature_extract = True
path_models = '/media/david/datos/Violence DATA/violentflows/Models/'
path_results = '/media/david/datos/Violence DATA/violentflows/Results'

gpath = '/media/david/datos/Violence DATA/violentflows/movies Frames'

debugg_mode = False

for dataset_train, dataset_train_labels,dataset_test,dataset_test_labels   in k_folds_from_folders(gpath, 5):
#   print(dataset_test)
    
    image_datasets = {
        'train':ViolenceDatasetVideos(
            dataset= dataset_train,
            labels=dataset_train_labels,
            spatial_transform = data_transforms['train'],
            source = dataset_source,
            interval_duration = interval_duration,
            difference = 3,
            maxDuration = avgmaxDuration,
            nDynamicImages = numDiPerVideos,
            debugg_mode = debugg_mode
        ),
        'val': ViolenceDatasetVideos(
            dataset= dataset_test,
            labels=dataset_test_labels,
            spatial_transform = data_transforms['val'],
            source = dataset_source,
            interval_duration = interval_duration,
            difference=3,
            maxDuration = avgmaxDuration,
            nDynamicImages = numDiPerVideos,
            debugg_mode = debugg_mode
        )
    }
    dataloaders_dict = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    }
    
    model = None
    
    model, input_size = initialize_model(model_name = modelType, num_classes = 2, feature_extract=feature_extract, numDiPerVideos=numDiPerVideos, use_pretrained=True)
    model.to(device)
    params_to_update = verifiParametersToTrain(model)
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    ### trainer
    trainer = Trainer(model,dataloaders_dict,criterion,optimizer,exp_lr_scheduler,device,num_epochs)
    
    for epoch in range(1, num_epochs + 1):
      # Train and evaluate
      epoch_loss_train, epoch_acc_train  = trainer.train_epoch(epoch)
      epoch_loss_test, epoch_acc_test = trainer.test_epoch(epoch)
      
      train_lost.append(epoch_loss_train)
      train_acc.append(epoch_acc_train)
      test_lost.append(epoch_loss_test)
      test_acc.append(epoch_acc_test)
      
    #   tb.save_value("trainLoss", "train_loss", foldidx*num_epochs + epoch, epoch_loss_train)
    #   tb.save_value("trainAcc", "train_acc", foldidx*num_epochs + epoch, epoch_acc_train)
    #   tb.save_value("testLoss", "test_loss", foldidx*num_epochs + epoch, epoch_loss_test)
    #   tb.save_value("testAcc", "test_acc", foldidx*num_epochs + epoch, epoch_acc_test)

    #   tb.flush_line('train_loss')
    #   tb.flush_line('train_acc')
    #   tb.flush_line('test_loss')
    #   tb.flush_line('test_acc')
     
#     filepath = path_models+str(modelType)+'('+str(numDiPerVideos)+'di)-fold-'+str(foldidx)+'.pt'
#     torch.save({
#         'kfold': foldidx,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict()
#         }, filepath)
     
    foldidx = foldidx + 1
    
print('saving loss and acc history...')  
saveList(path_results,modelType,'train_lost', numDiPerVideos, dataset_source, train_lost)
saveList(path_results,modelType,'train_acc', numDiPerVideos, dataset_source, train_acc)
saveList(path_results,modelType,'test_lost', numDiPerVideos, dataset_source, test_lost)
saveList(path_results,modelType,'test_acc', numDiPerVideos, dataset_source, test_acc)

# import torch
# print(torch.cuda.current_device())
# print(torch.cuda.device(0))
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.is_available())
# videos2ImagesFromKfols('/media/david/datos/Violence DATA/violentflows/movies','/media/david/datos/Violence DATA/violentflows/movies Frames')

# saveList(path_results,modelType,'pruebas', numDiPerVideos, dataset_source, [4,2,1,3])