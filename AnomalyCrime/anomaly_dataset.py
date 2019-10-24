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
    def __init__( self, dataset, labels, spatial_transform, source='frames', numFrames=0, nDynamicImages=0, debugg_mode = False ):
        self.spatial_transform = spatial_transform
        self.images = dataset
        self.labels = labels
        self.numFrames = numFrames
        self.nDynamicImages = nDynamicImages
        self.source = source
        self.debugg_mode = debugg_mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        # print(vid_name)
        label = self.labels[idx]
        dinamycImages = []
        if self.source == 'frames':
            frames_list = os.listdir(vid_name)
            frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
            total_frames = len(frames_list)
            if self.nDynamicImages > 0:
                seqLen = int(total_frames / self.nDynamicImages)
            else: seqLen = self.numFrames
            sequences = [ frames_list[x : x + seqLen] for x in range(0, total_frames, seqLen) ]
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
        # print(dinamycImages.size())
        return dinamycImages, label, vid_name

        #####################################################################################################################################




