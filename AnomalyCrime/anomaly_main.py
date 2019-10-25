import sys
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/MyCode')
from anomaly_dataset import AnomalyDataset
import os
import re
from util import video2Images2, saveList, get_model_name
import csv
import pandas as pd
import numpy as np
import cv2
from ViolenceDatasetV2 import ViolenceDatasetVideos
from transforms import createTransforms
import torch.nn as nn
import torch
from initializeModel import initialize_model
from parameters import verifiParametersToTrain
import torch.optim as optim
from torch.optim import lr_scheduler
from trainer import Trainer
import random
import util
import initialize_dataset
import constants
import glob
import argparse

def extractMetadata(path='/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/videos'):
    paths = os.listdir(path)
    paths.sort()
    # r = re.compile("([a-zA-Z]+)([0-9]+)")
    # labels = [r.match(string).groups() for string in paths] # (Robbery,089)
    # names = [str(tup[0])+str(tup[1]) for tup in labels] #Robbery089
    # labels = [tup[0] for tup in labels]  #Robbery
    names = [string[:-9] for string in paths]
    labels = [string[:-12] for string in paths]
    return names, labels, paths

def videos2frames(path_videos, path_frames):
#   listViolence = os.listdir(path_videos)
#   listViolence.sort()
    names, _, paths = extractMetadata(path_videos)
    # print(paths)
    # print(names)
    for idx,video in enumerate(paths):
        path_video = os.path.join(path_videos, video) #/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/videos/Vandalism050_x264.mp4
        # print('in: ',path_video)
        # path_frames_out = os.path.join(path_frames, str(idx + 1)) #/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/frames/violence/100
        path_frames_out = os.path.join(path_frames, names[idx])
        # print(path_frames_out)
        if not os.path.exists(path_frames_out):
            os.makedirs(path_frames_out)
        dirContents = os.listdir(path_frames_out)
        if len(dirContents) == 0:
            video2Images2(path_video, path_frames_out)

## process the Temporal_Anomaly_Annotation_for_Testing_Videos.txt        
def cutVideo(path):
    data = pd.read_csv(path, sep='  ') #name anomaly  start1  end1  start2  end2
    print(data.head())
    videos = data["name"].values
    anomaly = data["anomaly"].values
    start1 = data["start1"].values
    end1 = data["end1"].values
    start2 = data["start2"].values
    end2 = data["end2"].values
    # videos = [video.split("_")[0] for video in videos]
    print(videos)
    print(len(videos))
    
    return videos, anomaly, start1,end1, start2, end2

# def plotTemporalDetection(video_path, annotation_txt='/media/david/datos/Violence DATA/AnomalyCRIME/Temporal_Anomaly_Annotation_for_Testing_Videos.txt'):
    
    

def plotBoundingBox(video_path, bdx_file_path):
    data = []
    with open(bdx_file_path, 'r') as file:
        for row in file:
            data.append(row.split())
    data = np.array(data)
    # print(data.shape)
    # print(data[:, 5])
    vid = cv2.VideoCapture(video_path)
    index_frame = 0
    while(True):
        ret, frame = vid.read()
        if not ret:
            print('Houston there is a problem...')
            break
        index_frame += 1
        
        if index_frame < data.shape[0]:
            if int(data[index_frame,6]) == 0:
                frame = cv2.rectangle(frame, (int(data[index_frame, 1]), int(data[index_frame, 2])), (int(data[index_frame, 3]), int(data[index_frame, 4])), (0, 255, 0))
                cv2.putText(frame,str(index_frame),(int(data[index_frame, 1]), int(data[index_frame, 2])), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
        cv2.imshow('frame',frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
        
def createAnomalyDataset(path_frames):
    Dataset = []    
    classes = {'Normal_Videos': 0, 'Arrest': 1, 'Assault': 2, 'Burglary': 3, 'Robbery': 4, 'Stealing': 5, 'Vandalism': 6}
    names, labels, paths = extractMetadata()
    labels_int = [classes[label] for label in labels]
    # print(paths)
    # print(labels_int)
    for name in names:
        d = os.path.join(path_frames, name)
        Dataset.append(d)
    NumFrames = [len(glob.glob1(os.path.join(path_frames, names[i]), "*.jpg")) for i in range(len(Dataset))]
    return Dataset, labels_int, NumFrames
        
def train_test_videos(train_file, test_file, g_path):
    train_names = []
    train_labels = []
    test_names = []
    test_labes = []
    classes = {'Normal_Videos': 0, 'Arrest': 1, 'Assault': 2, 'Burglary': 3, 'Robbery': 4, 'Stealing': 5, 'Vandalism': 6}
    with open(train_file, 'r') as file:
        for row in file:
            train_names.append(os.path.join(g_path,row[:-1]))
            train_labels.append(row[:-4])

    with open(test_file, 'r') as file:
        for row in file:
            test_names.append(os.path.join(g_path,row[:-1]))
            test_labes.append(row[:-4])
    
    train_labels = [classes[label] for label in train_labels]
    test_labes = [classes[label] for label in test_labes]
    return train_names, train_labels, test_names, test_labes

def test_loader(dataloaders):
    #     inputs :  <class 'list'> 5
    # --> 1 torch.float32 torch.Size([28, 3, 224, 224])
    # --> 2 torch.float32 torch.Size([81, 3, 224, 224])
    # --> 3 torch.float32 torch.Size([94, 3, 224, 224])
    # --> 4 torch.float32 torch.Size([117, 3, 224, 224])
    # --> 5 torch.float32 torch.Size([72, 3, 224, 224])
    for inputs, labels in dataloaders["train"]:
        # inputs = inputs.permute(1, 0, 2, 3, 4)
        print('inputs : ', type(inputs), len(inputs))
        # print('inputs : ', inputs.size())
        # print('labels : ',labels.size())
        for idx,input in enumerate(inputs):
            print('-->',str(idx+1),input.dtype,input.size())
        print()

def normalVideoNormalize(num_avg_frames, video):
    path_frames_out = os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED,video[:-9])#folder
    if not os.path.exists(path_frames_out):
        os.makedirs(path_frames_out)
    vid = cv2.VideoCapture(os.path.join(constants.PATH_UCFCRIME2LOCAL_VIDEOS,video))
    index_frame = 0
    while(True):
        ret, frame = vid.read()
        if not ret:
            print('Houston there is a problem...')
            break
        index_frame += 1
        if index_frame < num_avg_frames:
            
            cv2.imwrite(os.path.join(path_frames_out, 'frame' + str("{0:03}".format(index_frame)) + '.jpg'), frame)

def createReducedDataset():
    """Create videos with only frames annotated with bounding box"""
    # videos = os.listdir(constants.PATH_UCFCRIME2LOCAL_VIDEOS)
    lista_videos = os.listdir(constants.PATH_UCFCRIME2LOCAL_VIDEOS)
    lista_videos.sort()
    NUM_AVG_FRAMES = 385

    for video in lista_videos:
        if video[0:6] == 'Normal':
            normalVideoNormalize(NUM_AVG_FRAMES,video)
            continue
        # print(video)
        bdx_file_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_BBOX_ANNOTATIONS, video[:-9]+'.txt')
        # print(bdx_file_path)
        path_frames_out = os.path.join(constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED,video[:-9])
        if not os.path.exists(path_frames_out):
            os.makedirs(path_frames_out)
        data = []
        with open(bdx_file_path, 'r') as file:
            for row in file:
                data.append(row.split())
        data = np.array(data)
        vid = cv2.VideoCapture(os.path.join(constants.PATH_UCFCRIME2LOCAL_VIDEOS,video))
        index_frame = 0
        frames_numbers = []
        
        while(True):
            ret, frame = vid.read()
            if not ret:
                # print('Houston there is a problem...')
                break
            index_frame += 1
            if index_frame < data.shape[0]:
                if int(data[index_frame, 6]) == 0:
                    frames_numbers.append(index_frame)
                    # cv2.imwrite(os.path.join(path_frames_out, 'frame' + str("{0:03}".format(index_frame)) + '.jpg'), frame)
                    # frame = cv2.rectangle(frame,(int(data[index_frame,1]),int(data[index_frame,2])),(int(data[index_frame,3]),int(data[index_frame,4])),(0,255,0))
        
        print('-'*20,video, len(frames_numbers))
        for idx in range(len(frames_numbers)):
            if (idx + 1) < len(frames_numbers):
                # print(frames_numbers[idx + 1], frames_numbers[idx])
                if int(frames_numbers[idx + 1]) - int(frames_numbers[idx]) < 4:
                    vid.set(cv2.CAP_PROP_POS_FRAMES, int(frames_numbers[idx]))
                    ret, frame = vid.read()
                    if not ret:
                        print('Houston there is a problem...')
                        break
                    cv2.imwrite(os.path.join(path_frames_out, 'frame' + str("{0:03}".format(frames_numbers[idx])) + '.jpg'), frame)
                    # print(os.path.join(path_frames_out, 'frame' + str("{0:03}".format(frames_numbers[idx])) + '.jpg'))
                else:
                    break
        # print('frames for videos: ', len(frames_numbers))
    
def numFramesMean(path=constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED):
    lista_videos_folders = os.listdir(path)
    lista_videos_folders.sort()
    total_videos = len(lista_videos_folders)
    total = 0
    for i, vid_folder in enumerate(lista_videos_folders):
        numFrames = len(glob.glob1(os.path.join(path, vid_folder), "*.jpg"))
        total += numFrames
        print(os.path.join(path, vid_folder), numFrames)
    avg = int(total / total_videos) #385
    print(avg)

def __main_plot__():
    # file_anomaly_train = 'AnomalyCrime/anomaly_videos_train.txt'
    # file_anomaly_test = 'AnomalyCrime/anomaly_videos_test.txt'
    # data = []
    # with open(file_anomaly_train, 'r') as file:
    #     for row in file:
    #         data.append(row[:-1])
    # print(len(data), data)
    # for video in data:
    video = 'Arrest002'
    print('*'*20, video)
    plotBoundingBox(constants.PATH_UCFCRIME2LOCAL_VIDEOS +'/'+ str(video) + '_x264.mp4',
                    constants.PATH_UCFCRIME2LOCAL_README +'/Txt annotations/'+str(video)+'.txt')

def __main__():
    # print(train_names)
    # print(train_labels)
    # print(len(train_names), len(test_names))

    parser = argparse.ArgumentParser()
    parser.add_argument("--pathDataset", type=str, default=constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED, help="Directory containing results")
    parser.add_argument("--pathLearningCurves", type=str, default=constants.PATH_ANOMALY_LEARNING_CURVES, help="Directory containing results")
    parser.add_argument("--checkpointPath", type=str, default=constants.PATH_ANOMALY_CHECKPOINTS)
    parser.add_argument("--modelType",type=str,default="alexnet",help="model")
    parser.add_argument("--numEpochs",type=int,default=30)
    parser.add_argument("--batchSize",type=int,default=64)
    parser.add_argument("--featureExtract",type=lambda x: (str(x).lower() == 'true'), default=False, help="to fine tunning")
    parser.add_argument("--schedulerType",type=str,default="OnPlateau",help="learning rate scheduler")
    parser.add_argument("--debuggMode", type=bool, default=False, help="show prints")
    parser.add_argument("--ndis", type=int, help="num dyn imgs")
    parser.add_argument("--joinType", type=str, default="tempMaxPool", help="show prints")
    parser.add_argument("--plotSamples", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--numWorkers", type=int, default=4)

    args = parser.parse_args()
    path_dataset = args.pathDataset
    path_learning_curves = args.pathLearningCurves
    modelType = args.modelType
    batch_size = args.batchSize
    num_epochs = args.numEpochs
    feature_extract = args.featureExtract
    joinType = args.joinType
    scheduler_type = args.schedulerType
    debugg_mode = args.debuggMode
    numDiPerVideos = args.ndis
    path_checkpoints = args.checkpointPath
    plot_samples = args.plotSamples
    dataset_source = "frames"
    debugg_mode = False
    avgmaxDuration = 1.66
    interval_duration = 0.3
    num_workers = args.numWorkers
    input_size = 224
    numFrames = 0

    transforms = createTransforms(input_size)
    num_classes = 7 #{'Normal_Videos': 0, 'Arrest': 1, 'Assault': 2, 'Burglary': 3, 'Robbery': 4, 'Stealing': 5, 'Vandalism': 6}

    train_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_split_AD.txt')
    test_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_split_AD.txt')
    
    train_names, train_labels, test_names, test_labels = train_test_videos(train_videos_path, test_videos_path, path_dataset)
    combined = list(zip(train_names, train_labels))
    random.shuffle(combined)
    train_names[:], train_labels[:] = zip(*combined)
    combined = list(zip(test_names, test_labels))
    random.shuffle(combined)
    test_names[:], test_labels[:] = zip(*combined)
    print(len(train_names), len(train_labels), len(test_names), len(test_labels))
    util.print_dataset_balance(train_labels,test_labels)

    # dataset, labels, spatial_transform, source='frames', interval_duration=0.0, nDynamicImages=0, debugg_mode = False
    image_datasets = {
        "train": AnomalyDataset( dataset=train_names, labels=train_labels, spatial_transform=transforms["train"], source=dataset_source,
            numFrames=numFrames, nDynamicImages=numDiPerVideos, debugg_mode=debugg_mode, sequenceLength=40),
        "test": AnomalyDataset( dataset=test_names, labels=test_labels, spatial_transform=transforms["test"], source=dataset_source,
            numFrames=numFrames, nDynamicImages=numDiPerVideos, debugg_mode=debugg_mode, sequenceLength=40)
    }
    dataloaders_dict = {
        "train": torch.utils.data.DataLoader( image_datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "test": torch.utils.data.DataLoader( image_datasets["test"], batch_size=batch_size, shuffle=True, num_workers=num_workers),
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # # test_loader(dataloaders_dict)

    model, input_size = initialize_model( model_name=modelType, num_classes=num_classes, feature_extract=feature_extract, numDiPerVideos=numDiPerVideos, joinType=joinType, use_pretrained=True)
    model.to(device)
    MODEL_NAME = util.get_model_name(modelType, scheduler_type, numDiPerVideos, dataset_source, feature_extract, joinType, num_epochs)
    print(MODEL_NAME)
    params_to_update = verifiParametersToTrain(model, feature_extract)
    # print(model)

    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    if scheduler_type == "StepLR":
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif scheduler_type == "OnPlateau":
        exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer, patience=5, verbose=True )
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(model, dataloaders_dict, criterion, optimizer, exp_lr_scheduler, device, num_epochs, os.path.join(path_checkpoints,MODEL_NAME),numDiPerVideos, plot_samples)
    train_lost = []
    train_acc = []
    test_lost = []
    test_acc = []
    for epoch in range(1, num_epochs + 1):
        print("----- Epoch {}/{}".format(epoch, num_epochs))
        # Train and evaluate
        epoch_loss_train, epoch_acc_train = trainer.train_epoch(epoch)
        epoch_loss_test, epoch_acc_test = trainer.test_epoch(epoch)
        exp_lr_scheduler.step(epoch_loss_test)
        train_lost.append(epoch_loss_train)
        train_acc.append(epoch_acc_train)
        test_lost.append(epoch_loss_test)
        test_acc.append(epoch_acc_test)
    
    print("saving loss and acc history...")
    util.saveLearningCurve(os.path.join(path_learning_curves,MODEL_NAME+"-train_lost.txt"), train_lost)
    util.saveLearningCurve(os.path.join(path_learning_curves,MODEL_NAME+"-train_acc.txt"), train_acc)
    util.saveLearningCurve(os.path.join(path_learning_curves,MODEL_NAME+"-test_lost.txt"), test_lost)
    util.saveLearningCurve(os.path.join(path_learning_curves,MODEL_NAME+"-test_acc.txt"),test_acc)
    # print("saving loss and acc history...")
    # saveList(path_results, modelType, scheduler_type, "train_lost", numDiPerVideos, dataset_source, feature_extract, joinType, train_lost,)
    # saveList(path_results, modelType, scheduler_type,"train_acc",numDiPerVideos, dataset_source, feature_extract, joinType, train_acc, )
    # saveList( path_results, modelType, scheduler_type, "test_lost", numDiPerVideos, dataset_source, feature_extract, joinType, test_lost, )
    # saveList( path_results, modelType, scheduler_type, "test_acc", numDiPerVideos, dataset_source, feature_extract, joinType, test_acc, )

# __main_plot__()
__main__()
# numFramesMean()
# createReducedDataset()