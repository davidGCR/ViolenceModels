import sys
sys.path.insert(1, "/media/david/datos/PAPERS-SOURCE_CODE/MyCode")
import torch 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import torch.optim as lr_scheduler
import os 
import glob 
import cv2 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models 
import torch 
from torch.autograd import Variable 
import time
import numpy as np

from AlexNet import *
from ViolenceDatasetV2 import *
from trainer import *
from kfolds import *
from operator import itemgetter
import random
from initializeModel import *
from util import *
from verifyParameters import *

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import cross_val_score

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Setup the loss fxn
criterion = nn.CrossEntropyLoss()
input_size = 224

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

train_lost = []
train_acc = []
test_lost = []
test_acc = []

foldidx = 0
best_acc_test = 0.0
avgmaxDuration = 1.66

modelType = "alexnetv2"
interval_duration = 0.3
numDiPerVideos = 8
dataset_source = "frames"
debugg_mode = False
num_workers = 6
batch_size = 1000
num_epochs = 40
feature_extract = True
joinType = "tempMaxPool"
debugg_mode = False

# Create dataset HockeyFights
path_violence = '/media/david/datos/Violence DATA/HockeyFights/frames/violence'
path_noviolence = '/media/david/datos/Violence DATA/HockeyFights/frames/nonviolence'
datasetAll, labelsAll, numFramesAll = createDataset(path_violence, path_noviolence)
combined = list(zip(datasetAll, labelsAll, numFramesAll))
random.shuffle(combined)
datasetAll[:], labelsAll[:], numFramesAll[:] = zip(*combined)
print(len(datasetAll), len(labelsAll), len(numFramesAll))

dataset_train = datasetAll
dataset_train_labels =  labelsAll 

# dataset_test = list(itemgetter(*te_idx)(datasetAll)) 
# dataset_test_labels =  list(itemgetter(*te_idx)(labelsAll))

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
    )
}
dataloaders_dict = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
#     'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True, num_workers=num_workers)
}

model = None
model, input_size = initialize_model(model_name = modelType, num_classes = 2, feature_extract=True, numDiPerVideos=numDiPerVideos, joinType = joinType ,use_pretrained=True)
model.to(device)
params_to_update = verifiParametersToTrain(model)
# Observe that all parameters are being optimized
optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

lista = []
dataset_labels = None

for inputs, labels in dataloaders_dict["train"]:
  print('==== dataloader size:' ,inputs.size())
  inputs = inputs.permute(1, 0, 2, 3, 4)
  inputs = inputs.to(device)
  dataset_labels = labels
  # zero the parameter gradient
  optimizer.zero_grad()
  # forward
  # track history if only in train
  with torch.set_grad_enabled(True):
    outputs = model.getFeatureVector(inputs)
    outputs = outputs.cpu()
    lista.append(outputs.numpy())

dataset = np.array(lista)
dataset = dataset.squeeze()

path_results = '/media/david/datos/Violence DATA/HockeyFights/CNN+SVM data'
saveList(
    path_results,
    modelType,
    "dataset",
    numDiPerVideos,
    dataset_source,
    feature_extract,
    joinType,
    dataset,
)
saveList(
    path_results,
    modelType,
    "dataset_labels",
    numDiPerVideos,
    dataset_source,
    feature_extract,
    joinType,
    dataset_labels,
)

path_results = '/media/david/datos/Violence DATA/HockeyFights/CNN+SVM data'
dataset = loadArray(path_results,
    modelType,
    "dataset",
    numDiPerVideos,
    dataset_source,
    feature_extract,
    joinType
    )
dataset_labels = loadArray(path_results,
    modelType,
    "dataset_labels",
    numDiPerVideos,
    dataset_source,
    feature_extract,
    joinType
    )
labels = dataset_labels.numpy()

# print('dataset_paths shape:', type(dataset_paths), len(dataset_paths))
print('dataset shape:', type(dataset), dataset.shape)
print('labels shape:', type(dataset_labels), dataset_labels.shape)


# print(dataset)
# print(dataset_paths[0:10])
# print(labels)

# # Grid Search
# # Parameter Grid
# param_grid = {'C': [0.1, 1, 10, 100]}
# # Make grid search classifier
# clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
# # Train the classifier
# clf_grid.fit(dataset, labels)
# # clf = grid.best_estimator_()
# print("Best Parameters:", clf_grid.best_params_)
# print("Best Estimators:", clf_grid.best_estimator_)

clf = svm.SVC(kernel='linear', C=1)
# clf = svm.SVC(kernel='rbf', C = 10.0, gamma=0.1)
scores = cross_val_score(clf, dataset, labels, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# clf2 = svm.SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
#     kernel='rbf', max_iter=-1, probability=False, random_state=None,
#     shrinking=True, tol=0.001, verbose=False)
# scores2 = cross_val_score(clf2, dataset, labels, cv=5)
# print(scores2)