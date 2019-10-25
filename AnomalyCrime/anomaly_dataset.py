from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import cv2
import torch
import glob
import time
from dinamycImage import *


class AnomalyDataset(Dataset):
    def __init__( self, dataset, labels, spatial_transform, source='frames', numFrames=0, nDynamicImages=0, debugg_mode = False, sequenceLength=0):
        self.spatial_transform = spatial_transform
        self.images = dataset
        self.labels = labels
        self.numFrames = numFrames #num frames to resume
        self.nDynamicImages = nDynamicImages
        self.source = source
        self.debugg_mode = debugg_mode
        self.sequenceLength = sequenceLength

    def __len__(self):
        return len(self.images)
    
    def getSequences(self, vid_name):
        frames_list = os.listdir(vid_name)
        frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        if self.sequenceLength == 0: num_frames_on_video = len(frames_list)
        else: num_frames_on_video = self.sequenceLength
        if self.nDynamicImages > 0:
            seqLen = int(num_frames_on_video / self.nDynamicImages)
        else: seqLen = self.numFrames
        sequences = [frames_list[x:x + seqLen] for x in range(0, num_frames_on_video, seqLen)]
        if len(sequences) > self.nDynamicImages:
            diff = len(sequences) - self.nDynamicImages
            sequences = sequences[: - diff]
        # if len(sequences) != self.nDynamicImages:
        # print('-->len(sequences)',len(sequences))
        return sequences, seqLen

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        # print(vid_name)
        label = self.labels[idx]
        dinamycImages = []
        if self.source == 'frames':
            sequences, seqLen = self.getSequences(vid_name)
            for seq in sequences:
                if len(seq) == seqLen:
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
        print(dinamycImages.size()) #torch.Size([ndi, ch, h, w])
        return dinamycImages, label, vid_name

        #####################################################################################################################################




