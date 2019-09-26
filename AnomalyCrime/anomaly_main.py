import sys
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/MyCode')
from dataset import *
import os
import re
from util import video2Images2
import csv
import pandas as pd
import numpy as np
import cv2

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
                frame = cv2.rectangle(frame,(int(data[index_frame,1]),int(data[index_frame,2])),(int(data[index_frame,3]),int(data[index_frame,4])),(0,255,0))
        cv2.imshow('frame',frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
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
    
    # print('Dataset: ',Dataset)
    # imagesNoF = []
    # list_no_violence = os.listdir(path_noviolence)
    # list_no_violence.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    # for target in list_no_violence:
    #     d = os.path.join(path_noviolence, target)
    #     imagesNoF.append(d)

    # Dataset = imagesF + imagesNoF
    # Labels = list([1] * len(imagesF)) + list([0] * len(imagesNoF))
    NumFrames = [len(glob.glob1(os.path.join(path_frames, names[i]), "*.jpg")) for i in range(len(Dataset))]
    return Dataset, labels_int, NumFrames

def __main__():
    # Dataset, Labels, NumFrames =
    # videos2frames('/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/videos', '/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/frames')
    names, labels, paths = extractMetadata('/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/videos')
    # print('names: ', names)
    # print('labels: ', labels)
    # print('paths: ',paths)
    Dataset, Labels, NumFrames = createAnomalyDataset('/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/frames')
    print('Dataset: ', Dataset)
    print('Labels: ', Labels)
    print('NumFrames: ',NumFrames)
    # cutVideo('/media/david/datos/Violence DATA/AnomalyCRIME/Temporal_Anomaly_Annotation_for_Testing_Videos.txt')
    # dataset_path = '/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local'
    # video_name = 'Stealing009'
    # plotBoundingBox(os.path.join(dataset_path,'videos/'+video_name+'_x264.mp4'),os.path.join(dataset_path,'readme/Txt annotations/'+video_name +'.txt'))

__main__()