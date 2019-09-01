import sys
# sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/MyCode')
from AlexNet import *
from ViolenceDatasetV2 import *
from trainer import *
from kfolds import *
from operator import itemgetter
import random
from initializeModel import *
from util import *


##Create dataset
path_violence = '/content/drive/My Drive/VIOLENCE DATASETS/HockeyFightsFrames/Fights'
path_noviolence = '/content/drive/My Drive/VIOLENCE DATASETS/HockeyFightsFrames/noFights'

# path_violence = '/content/drive/My Drive/VIOLENCE DATASETS/HockeyFightsVideos/Fights'
# path_noviolence = '/content/drive/My Drive/VIOLENCE DATASETS/HockeyFightsVideos/NoFights'

datasetAll, labelsAll, numFramesAll = createDataset(path_violence,path_noviolence)
combined = list(zip(datasetAll, labelsAll, numFramesAll))
random.shuffle(combined)
datasetAll[:], labelsAll[:], numFramesAll[:] = zip(*combined)

print(len(datasetAll), len(labelsAll), len(numFramesAll))


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

tb = TensorBoardColab()
train_lost = []
train_acc = []
test_lost = []
test_acc = []

foldidx = 0
best_acc_test = 0.0
avgmaxDuration = 1.66
