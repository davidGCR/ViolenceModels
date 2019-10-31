
import sys
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/MyCode')
import cv2
import argparse
import numpy as np
import constants
import os
import matplotlib.pyplot as plt
import re

def imshow(img):
    img = img / 2 + 0.5    
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

def plotBoundingBoxVideo(video_path, bdx_file_path):
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
            # print(data[index_frame])
            if int(data[index_frame,6]) == 0:
                frame = cv2.rectangle(frame, (int(data[index_frame, 1]), int(data[index_frame, 2])), (int(data[index_frame, 3]), int(data[index_frame, 4])), (0, 255, 0))
            # cv2.putText(frame,str(index_frame),(int(data[index_frame, 1]), int(data[index_frame, 2])), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,0),0.7,cv2.LINE_AA)
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

def plotBoundingBoxFrames(folder_path_video, bdx_file_path):
    data = []
    print('from Frames...')
    with open(bdx_file_path, 'r') as file:
        for row in file:
            data.append(row.split())
    data = np.array(data)
    # print(data.shape)
    # print(data[:, 5])
    frames = os.listdir(folder_path_video)
    # frames.sort()
    frames.sort(key=natural_keys)
    index_frame = 0
    for frame in frames:
        index_frame = int(frame[5:-4])
        # print(data[index_frame],index_frame)
        # index_frame = frames
        # data[index_frame, 5]
        frame = cv2.imread(os.path.join(folder_path_video, frame))
        if index_frame < data.shape[0]:
            if int(data[index_frame, 6]) == 0:
                frame = cv2.rectangle(frame, (int(data[index_frame, 1]), int(data[index_frame, 2])), (int(data[index_frame, 3]), int(data[index_frame, 4])), (0, 255, 0))
                # cv2.putText(frame,str(index_frame),(int(data[index_frame, 1]), int(data[index_frame, 2])), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,0),0.7,cv2.LINE_AA)
        cv2.imshow('frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

def __main__():

    # file_anomaly_train = 'AnomalyCrime/anomaly_videos_train.txt'
    # file_anomaly_test = 'AnomalyCrime/anomaly_videos_test.txt'
    # data = []
    # with open(file_anomaly_train, 'r') as file:
    #     for row in file:
    #         data.append(row[:-1])
    # print(len(data), data)
    # for video in data:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fromFrames", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--videoName", type=str)
    args = parser.parse_args()
    from_frames = args.fromFrames
    video = args.videoName
    print('*'*20, video)
    if from_frames:
        plotBoundingBoxFrames(constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED +'/'+ str(video),
                    constants.PATH_UCFCRIME2LOCAL_README + '/Txt annotations/' + str(video) + '.txt')
    else:
        plotBoundingBoxVideo(constants.PATH_UCFCRIME2LOCAL_VIDEOS +'/'+ str(video) + '_x264.mp4',
                    constants.PATH_UCFCRIME2LOCAL_README + '/Txt annotations/' + str(video) + '.txt')

__main__()