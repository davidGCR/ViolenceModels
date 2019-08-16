#import numpy as nump

# load additional module
# import pickle

# # define a list of places
# # hist = [0.12452, 0.12452, 0.12452, 0.12452]
# data_file = 'history.txt'
# # with open(data_file, 'wb') as filehandle:
# #     # store the data as binary data stream
# #     pickle.dump(hist, filehandle)

# with open(data_file, 'rb') as filehandle:
#     # read the data as binary data stream
#     hist = pickle.load(filehandle)
# print(type(hist))
# print(hist)


import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/MyCode')
from ViolenceAlexNet import *
from ViolenceDataset import createDataset, ViolenceDatasetVideos
from trainer import Trainer
from kfolds import k_folds
from operator import itemgetter

##Create dataset
path_violence = '/content/drive/My Drive/VIOLENCE DATASETS/HockeyFightsFrames/Fights'
path_noviolence = '/content/drive/My Drive/VIOLENCE DATASETS/HockeyFightsFrames/noFights'

datasetAll, labelsAll, numFramesAll = createDataset(path_violence,path_noviolence)

print(len(datasetAll), len(labelsAll), len(numFramesAll))


seqLen = 20
batch_size = 64

""" Alexnet
"""
model = None
model = ViolenceModel2(2)
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

### trainer
trainer = Trainer(model,dataloaders,criterion,optimizer)

# # # # training
n_epoch = 0

for train_idx, test_idx in k_folds(n_splits = 5, subjects = len(datasetAll)):

    dataset_train = list(itemgetter(*train_idx)(datasetAll)) 
    dataset_train_labels =  list(itemgetter(*train_idx)(labelsAll)) 
    
    dataset_test = list(itemgetter(*test_idx)(datasetAll)) 
    dataset_test_labels =  list(itemgetter(*test_idx)(labelsAll))
    
    image_datasets = {
        'train':ViolenceDatasetVideos(dataset_train,dataset_train_labels,data_transforms['train'],seqLen),
        'val': ViolenceDatasetVideos(dataset_test,dataset_test_labels,data_transforms['val'],seqLen)
    }
    dataloaders_dict = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True, num_workers=4)
    }
    n_epoch = n_epoch + 1
    # Train and evaluate
    trainer.train_epoch(model,dataloaders_dict,criterion,optimizer)

    trainer.tb.flush_line('train_loss')
    trainer.tb.flush_line('train_acc')