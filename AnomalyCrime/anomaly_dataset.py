from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import cv2
import torch
import glob
import time
from dynamicImage import *
import random


class AnomalyDataset(Dataset):
    def __init__(self, dataset, labels, numFrames, spatial_transform, source, nDynamicImages,
                    videoSegmentLength, maxNumFramesOnVideo, positionSegment):
        self.spatial_transform = spatial_transform
        self.images = dataset
        self.labels = labels
        self.numFrames = numFrames #num frames to resume
        self.nDynamicImages = nDynamicImages
        self.source = source
        self.maxNumFramesOnVideo = maxNumFramesOnVideo # to use only some frames
        self.videoSegmentLength = videoSegmentLength
        self.positionSegment = positionSegment

    def __len__(self):
        return len(self.images)

    def getRandomSegment(self, sequences, idx):
        random_segment = None
        label = int(self.labels[idx])
        # print('label: ', label)
        if self.nDynamicImages == 1:
            random_segment_idx = random.randint(0, len(sequences) - 1)
            random_segment = sequences[random_segment_idx]
            if self.numFrames[idx] > self.videoSegmentLength:
                if label == 0:
                    cut = int((len(sequences)*35)/100)
                    # print('Trimed sequence: ', len(sequences), cut)
                    sequences = sequences[cut:]
                while len(random_segment) != self.videoSegmentLength:
                    random_segment_idx = random.randint(0, len(sequences) - 1)
                    random_segment = sequences[random_segment_idx]
        # print('random sequence:', random_segment_idx, len(random_segment))
        return random_segment
     
    def getCentralSegment(self, sequences, idx):
        segment = None
        # label = int(self.labels[idx])
        # print('label: ', label)
        if self.nDynamicImages == 1:
            central_segment_idx = int(len(sequences)/2) 
            segment = sequences[central_segment_idx]
            # print('central segment: ', len(sequences), central_segment_idx)
        # print('random sequence:', random_segment_idx, len(random_segment))
        return segment

    def getSequences(self, vid_name, idx):
        frames_list = os.listdir(vid_name)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        
        seqLen = 0 #number of frames for each segment
        if self.videoSegmentLength > 0:
            if self.numFrames[idx] <= self.videoSegmentLength:
                # print('hereeeeeeeeeeeeeee....')
                seqLen = self.numFrames[idx]
            else:
                seqLen = self.videoSegmentLength
            num_frames_on_video = self.numFrames[idx] 
            sequences = [frames_list[x:x + seqLen] for x in range(0, num_frames_on_video, seqLen)]
            random_segment = self.getRandomSegment(sequences, idx) if self.positionSegment == 'random' else self.getCentralSegment(sequences,idx)
            sequences = []
            sequences.append(random_segment)
        else:
            if self.maxNumFramesOnVideo == 0:
                num_frames_on_video = self.numFrames[idx]
            else:
                num_frames_on_video = self.maxNumFramesOnVideo if self.numFrames[idx] >= self.maxNumFramesOnVideo else self.numFrames[idx]
            seqLen = num_frames_on_video // self.nDynamicImages
            sequences = [frames_list[x:x + seqLen] for x in range(0, num_frames_on_video, seqLen)]
            if len(sequences) > self.nDynamicImages:
                diff = len(sequences) - self.nDynamicImages
                sequences = sequences[: - diff]
        
        # if len(sequences) < self.nDynamicImages:
        #     print('-->len(sequences)',len(sequences))
        return sequences, seqLen

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        # print(vid_name)
        label = self.labels[idx]
        dinamycImages = []
        if self.source == 'frames':
            sequences, seqLen = self.getSequences(vid_name, idx)
            # if len(sequences) < self.nDynamicImages:
            #     print('NOOOOO cumple numero de sequeencias...', vid_name, self.numFrames[idx])
            for seq in sequences:
                # if len(seq) == seqLen:
                frames = []
                for frame in seq:
                    img_dir = str(vid_name) + "/" + frame
                    img = Image.open(img_dir).convert("RGB")
                    img = np.array(img)
                    frames.append(img)
                imgPIL, img = getDynamicImage(frames)
                imgPIL = self.spatial_transform(imgPIL.convert("RGB"))
                dinamycImages.append(imgPIL)               
                    # print(imgPIL.size())
        
        dinamycImages = torch.stack(dinamycImages, dim=0)
        if self.nDynamicImages == 1:
            dinamycImages = dinamycImages.squeeze(dim=0)
            # print(dinamycImages.size()) #torch.Size([ndi, ch, h, w])
        return dinamycImages, label, vid_name

        #####################################################################################################################################

def labels_2_binary(multi_labels):
    binary_labels = multi_labels.copy()
    for idx,label in enumerate(multi_labels):
        if label == 0:
            binary_labels[idx]=0
        else:
            binary_labels[idx]=1
    return binary_labels

def train_test_videos(train_file, test_file, g_path):
    """ load train-test split from original dataset """
    train_names = []
    train_labels = []
    test_names = []
    test_labels = []
    classes = {'Normal_Videos': 0, 'Arrest': 1, 'Assault': 2, 'Burglary': 3, 'Robbery': 4, 'Stealing': 5, 'Vandalism': 6}
    with open(train_file, 'r') as file:
        for row in file:
            train_names.append(os.path.join(g_path,row[:-1]))
            train_labels.append(row[:-4])

    with open(test_file, 'r') as file:
        for row in file:
            test_names.append(os.path.join(g_path,row[:-1]))
            test_labels.append(row[:-4])
    # for idx,label in enumerate(train_labels):
    #     if label == 'Normal_Videos':
    #         train_labels[idx]=0
    #     else:
    #         train_labels[idx]=1
    # for idx,label in enumerate(test_labels):
    #     if label == 'Normal_Videos':
    #         test_labels[idx]=0
    #     else:
    #         test_labels[idx]=1       
    train_labels = [classes[label] for label in train_labels]
    test_labels = [classes[label] for label in test_labels]
    # for i in range(len(train_names)):
    #      print(train_names[i])
    NumFrames_train = [len(glob.glob1(train_names[i], "*.jpg")) for i in range(len(train_names))]
    NumFrames_test = [len(glob.glob1(test_names[i], "*.jpg")) for i in range(len(test_names))]
    return train_names, train_labels, NumFrames_train, test_names, test_labels, NumFrames_test



