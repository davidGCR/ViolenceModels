import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from scipy import misc
# from model import saliency_model
# from resnet import resnet
# from loss import Loss
from saliency_model import *
from util import createDataset
import random
from tqdm import tqdm
from ViolenceDatasetV2 import ViolenceDatasetVideos
from operator import itemgetter
from transforms import createTransforms
from loss import Loss
from torch.optim import lr_scheduler
import argparse
import os

def save_checkpoint(state, filename='sal.pth.tar'):
    print('save in: ',filename)
    torch.save(state, filename)

def load_checkpoint(net,optimizer,filename='small.pth.tar'):
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return net,optimizer

# dataset_train = list(itemgetter(*train_idx)(datasetAll))
# dataset_train_labels = list(itemgetter(*train_idx)(labelsAll))
# dataset_test = list(itemgetter(*test_idx)(datasetAll))
# dataset_test_labels = list(itemgetter(*test_idx)(labelsAll))

# interval_duration = 0
# avgmaxDuration = 0
# numDiPerVideos = 1
# input_size = 224
# data_transforms = createTransforms(input_size)
# dataset_source = 'frames'
# debugg_mode = False
# batch_size = 8
# num_workers = 4
# num_epochs = 10
# num_classes = 2

def init(batch_size, num_workers, interval_duration, data_transforms, dataset_source, debugg_mode, numDiPerVideos, avgmaxDuration):
    hockey_path_violence = "/media/david/datos/Violence DATA/HockeyFights/frames/violence"
    hockey_path_noviolence = "/media/david/datos/Violence DATA/HockeyFights/frames/nonviolence"
    datasetAll, labelsAll, numFramesAll = createDataset(hockey_path_violence, hockey_path_noviolence) #ordered
    combined = list(zip(datasetAll, labelsAll, numFramesAll))
    random.shuffle(combined)
    datasetAll[:], labelsAll[:], numFramesAll[:] = zip(*combined) 
    print(len(datasetAll), len(labelsAll), len(numFramesAll))
    image_datasets = {
    "train": ViolenceDatasetVideos( dataset=datasetAll, labels=labelsAll, spatial_transform=data_transforms["train"], source=dataset_source,
        interval_duration=interval_duration,difference=3, maxDuration=avgmaxDuration, nDynamicImages=numDiPerVideos, debugg_mode=debugg_mode, ),
    # "val": ViolenceDatasetVideos( dataset=datasetAll, labels=labelsAll, spatial_transform=data_transforms["val"], source=dataset_source,
    #     interval_duration=interval_duration, difference=3, maxDuration=avgmaxDuration, nDynamicImages=numDiPerVideos, debugg_mode=debugg_mode, )
    }
    dataloaders_dict = {
        "train": torch.utils.data.DataLoader( image_datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers, ),
        # "val": torch.utils.data.DataLoader( image_datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers, ),
    }
    return image_datasets, dataloaders_dict

def train(num_classes, num_epochs, regularizers, device, checkpoint_path, dataloaders_dict, black_box_file, numDynamicImages=1):
    # trainloader,testloader,classes = cifar10()

    net = saliency_model(num_classes=num_classes)
    net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    # params_to_update = net.parameters()
    # optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters())
    # scheduler_type = "OnPlateau"
    # if scheduler_type == "StepLR":
    #     exp_lr_scheduler = lr_scheduler.StepLR( optimizer, step_size=7, gamma=0.1 )
    # elif scheduler_type == "OnPlateau":
    #     exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
                
    # black_box_func = resnet(pretrained=True)
    # black_box_func = torch.load('/media/david/datos/Violence DATA/HockeyFights/checkpoints/resnet18-frames-Finetuned:False-3di-tempMaxPool-OnPlateau.tar')
    black_box_func = torch.load(black_box_file)
    black_box_func = black_box_func.cuda()
    loss_func = Loss(num_classes=num_classes, regularizers=regularizers)
    best_loss = 1000.0
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("----- Epoch {}/{}".format(epoch+1, num_epochs))
        running_loss = 0.0
        running_loss_train = 0.0
        # running_corrects = 0.0
        
        for i, data in tqdm(enumerate(dataloaders_dict['train'], 0)):
            # get the inputs
            inputs, labels = data #dataset load [bs,ndi,c,w,h]
            # print('dataset element: ',inputs_r.shape) #torch.Size([8, 1, 3, 224, 224])
            if numDynamicImages > 1:
                inputs = inputs.permute(1, 0, 2, 3, 4)
                inputs = torch.squeeze(inputs, 0) #get one di [bs,c,w,h]
            # print('inputs shape:',inputs.shape)
            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            mask, out = net(inputs, labels)
            # print('mask shape:', mask.shape)
            # print('inputs shape:',inputs.shape)
            # print('labels shape:', labels.shape)
            # print(labels)

            # inputs_r = Variable(inputs_r.cuda())
            loss = loss_func.get(mask,inputs,labels,black_box_func)
            # running_loss += loss.data[0]
            running_loss += loss.item()
            running_loss_train += loss.item()*inputs.size(0)
            # if(i%10 == 0):
            #     print('Epoch = %f , Loss = %f '%(epoch+1 , running_loss/(batch_size*(i+1))) )
        
            loss.backward()
            optimizer.step()
        # exp_lr_scheduler.step(running_loss)

        epoch_loss = running_loss / len(dataloaders_dict["train"].dataset)
        epoch_loss_train = running_loss_train / len(dataloaders_dict["train"].dataset)
        print("{} RawLoss: {:.4f} Loss: {:.4f}".format('train', epoch_loss, epoch_loss_train))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # self.best_model_wts = copy.deepcopy(self.model.state_dict())
            print('SAving entire model...')
            save_checkpoint(net,checkpoint_path)

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchSize", type=int, default=8)
    parser.add_argument("--numEpochs", type=int, default=10)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--areaL", type=float, default=None)
    parser.add_argument("--smoothL", type=float, default=None)
    parser.add_argument("--preserverL", type=float, default=None)
    parser.add_argument("--areaPowerL", type=float, default=None)
    parser.add_argument("--saliencyModelFolder",type=str)
    parser.add_argument("--blackBoxFile",type=str) #areaL-9.0_smoothL-0.3_epochs-20
    
    # parser.add_argument("--areaL", type=float, default=8)
    # parser.add_argument("--smoothL", type=float, default=0.5)
    # parser.add_argument("--preserverL", type=float, default=0.3)
    # parser.add_argument("--areaPowerL", type=float, default=0.3)
    # parser.add_argument("--checkpointInfo",type=str)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    interval_duration = 0
    avgmaxDuration = 0
    numDiPerVideos = 1
    input_size = 224
    data_transforms = createTransforms(input_size)
    dataset_source = 'frames'
    debugg_mode = False
    batch_size = args.batchSize
    num_workers = args.numWorkers
    num_epochs = args.numEpochs
    black_box_file = args.blackBoxFile
    saliency_model_folder = args.saliencyModelFolder
    num_classes = 2
    # regularizers = {'area_loss_coef': args.areaL, 'smoothness_loss_coef': args.smoothL, 'preserver_loss_coef': args.preserverL, 'area_loss_power': args.areaPowerL}
    # checkpoint_info = args.checkpointInfo
    checkpoint_info = ''
    areaL, smoothL, preserverL, areaPowerL = None,None,None,None
    
    if args.areaL == None:
        areaL = 8
    else:
        areaL = args.areaL
        checkpoint_info += '_areaL-'+str(args.areaL)
    
    if args.smoothL == None:
        smoothL = 0.5
    else:
        smoothL = args.smoothL
        checkpoint_info += '_smoothL-' + str(args.smoothL)
    
    if args.preserverL == None:
        preserverL = 0.3
    else:
        preserverL = args.preserverL
        checkpoint_info += '_preserverL-' + str(args.preserverL)
    
    if args.areaPowerL == None:
        areaPowerL = 0.3
    else:
        areaPowerL = args.areaPowerL
        checkpoint_info += '_areaPowerL-' + str(args.areaPowerL)

    print('areaL, smoothL, preserverL, _areaPowerL=',areaL, smoothL, preserverL, areaPowerL)
    
    regularizers = {'area_loss_coef': areaL, 'smoothness_loss_coef': smoothL, 'preserver_loss_coef': preserverL, 'area_loss_power': areaPowerL}

    checkpoint_path = os.path.join(saliency_model_folder, 'saliency_model' + checkpoint_info + '_epochs-' + str(num_epochs) + '.tar')
    
    image_datasets, dataloaders_dict = init(batch_size, num_workers, interval_duration, data_transforms, dataset_source, debugg_mode, numDiPerVideos, avgmaxDuration)
    train(num_classes, num_epochs, regularizers, device, checkpoint_path, dataloaders_dict, black_box_file)
    
__main__()