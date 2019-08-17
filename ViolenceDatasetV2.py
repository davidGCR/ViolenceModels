from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import cv2
import torch
import glob
import time
from dinamycImage import *


class ViolenceDatasetVideos(Dataset):
    def __init__(
        self,
        dataset,
        labels,
        spatial_transform,
        seqLen=0,
        interval_duration=0.0,
        difference=3,
    ):
        """
    Args:
        dataset (list): Paths to the videos.
        labels (list): labels froma data
        seqLen (int): Number of frames in each segment
        
        spatial_transform (callable, optional): Optional transform to be applied
            on a sample.
    """
        self.spatial_transform = spatial_transform
        self.images = dataset
        self.labels = labels
        self.seqLen = seqLen
        self.interval_duration = interval_duration
        self.diference_max = difference
        self.nDynamicImages = 0

    def getNoDynamicImagesEstimated(self, idx):
        vid_name = self.images[idx]
        cap = cv2.VideoCapture(vid_name)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = video_length / fps
        self.nDynamicImages = int(
                duration / self.interval_duration
            )  # Number of Dynamic images estimated
        return self.nDynamicImages

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        vid_name = self.images[idx]
        print(vid_name)
        label = self.labels[idx]
        inpSeq = []
        ################################ From videos ################################
        if self.seqLen == 0 and self.interval_duration != 0.0:
            # video_path = '/content/drive/My Drive/VIOLENCE DATASETS/HockeyFightsVideos/Fights/fi2_xvid.avi'
            cap = cv2.VideoCapture(vid_name)
            # start_time_ms = time.time()

            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # fps = cap.get(cv2.CAP_PROP_FPS)
            # duration = video_length / fps
            # print('video durationnnnnnnnnnnnnnnnnnn: ')

            count = 0
            success = True
            frames = []
            # self.nDynamicImages = int(
            #     duration / self.interval_duration
            # )  # Number of Dynamic images estimated

            numberFramesInterval = 0

            capture_duration = self.interval_duration
            while count < video_length - 1:
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                # print('current_time ',current_time, 'capture_duration ',capture_duration)
                if current_time <= capture_duration:
                    success, image = cap.read()
                    if success:
                        frames.append(image)
                else:
                    # print('dimimimimimimimimim: ')
                    numberFramesInterval = len(frames)  ##number of frames to sumarize
                    img = getDynamicImage(frames)
                    inpSeq.append(self.spatial_transform(img.convert("RGB")))
                    frames = []
                    frames.append(image)
                    capture_duration = current_time + self.interval_duration
                count += 1
            ## the last chunk
            if numberFramesInterval - len(frames) < self.diference_max:
                img = getDynamicImage(frames)
                inpSeq.append(self.spatial_transform(img.convert("RGB")))  ##add dynamic image

        ################################ From frames ################################
        elif self.seqLen != 0 and self.interval_duration == 0.0:
            print('frameeeeeeeeeeeeeeees: ')
            frames_list = os.listdir(vid_name)
            frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
            total_frames = len(frames_list)

            sequences = [
                frames_list[x : x + self.seqLen]
                for x in range(0, total_frames, self.seqLen)
            ]

            # inpSeq = []

            for index, seq in enumerate(sequences):
                if len(seq) == self.seqLen:
                    frames = []
                    for frame in seq:
                        img_dir = str(vid_name) + "/" + frame

                        img = Image.open(img_dir).convert("RGB")
                        img = np.array(img)
                        frames.append(img)
                    img = getDynamicImage(frames)
                    inpSeq.append(self.spatial_transform(img.convert("RGB")))

        inpSeq = torch.stack(inpSeq, 0)

        return inpSeq, label

        #####################################################################################################################################


def createDataset(path_violence, path_noviolence):
    imagesF = []

    list_violence = os.listdir(path_violence)
    list_violence.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    for target in list_violence:
        d = os.path.join(path_violence, target)
        imagesF.append(d)
    imagesNoF = []
    list_no_violence = os.listdir(path_noviolence)
    list_no_violence.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    for target in list_no_violence:
        d = os.path.join(path_noviolence, target)
        imagesNoF.append(d)

    Dataset = imagesF + imagesNoF
    Labels = list([1] * len(imagesF)) + list([0] * len(imagesNoF))
    NumFrames = [len(glob.glob1(Dataset[i], "*.jpg")) for i in range(len(Dataset))]
    return Dataset, Labels, NumFrames

