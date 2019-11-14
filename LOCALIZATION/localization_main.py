
import sys
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/MyCode')
import argparse
import ANOMALYCRIME.transforms_anomaly as transforms_anomaly
import ANOMALYCRIME.anomaly_initializeDataset as anomaly_initializeDataset
import SALIENCY.saliencyTester as saliencyTester
# import SALIENCY
import constants
import torch
import os
import tkinter
from PIL import Image, ImageFont, ImageDraw, ImageTk
import numpy as np
import cv2
import glob
from localization_utils import tensor2numpy
import localization_utils
from point import Point
from bounding_box import BoundingBox
 
def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saliencyModelFile", type=str)
    parser.add_argument("--batchSize", type=int, default=3)
    # parser.add_argument("--numEpochs", type=int, default=10)
    parser.add_argument("--numWorkers", type=int, default=1)
    parser.add_argument("--numDiPerVideos", type=int, default=5)
    parser.add_argument("--shuffle", type=lambda x: (str(x).lower() == 'true'), default=False)

    args = parser.parse_args()
    maxNumFramesOnVideo = 0
    videoSegmentLength = 16
    numDiPerVideos = args.numDiPerVideos
    positionSegment = 'random'
    num_classes = 2 #anomalus or not
    input_size = 224
    transforms = transforms_anomaly.createTransforms(input_size)
    dataset_source = 'frames'
    batch_size = args.batchSize
    num_workers = args.numWorkers
    saliency_model_file = args.saliencyModelFile
    shuffle = args.shuffle
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    threshold = 0.5

    saliency_model_file = args.saliencyModelFile
    saliency_model_config = saliency_model_file

    path_dataset = constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED
    train_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_split_AD.txt')
    test_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_split_AD.txt')
    dataloaders_dict, test_names = anomaly_initializeDataset.initialize_final_only_test_anomaly_dataset(path_dataset, train_videos_path, test_videos_path, dataset_source, batch_size,
                                                        num_workers, numDiPerVideos, transforms, maxNumFramesOnVideo, videoSegmentLength, positionSegment, shuffle)
    tester = saliencyTester.SaliencyTester(saliency_model_file, num_classes, dataloaders_dict['test'], test_names,
                                        input_size, saliency_model_config, numDiPerVideos, threshold)


    for i, data in enumerate(dataloaders_dict['test'], 0):
        print("-" * 150)
        di_images, labels, video_name, bbox_segments = data
        print(video_name)
        masks = tester.compute_mask(di_images, labels)
        
        masks = torch.squeeze(masks,0)
        masks = masks.detach().cpu()
        masks = tester.normalize_tensor(masks)
        # masks = masks.repeat(3, 1, 1)
        masks = tensor2numpy(masks)
        print('masks numpy', masks.shape)
        bboxes = localization_utils.computeBoundingBoxFromMask(masks)
        
        print('bboxes 0: ', bboxes[0])
        
        # print(bboxes)
        img = localization_utils.plotBBoxesOnImage(masks,bboxes)
        while (1):
            cv2.imshow('eefef', img)
            k = cv2.waitKey(33)
            if k == -1:
                continue
            if k == ord('a'):
                break
            if k == ord('q'):
                localization_utils.tuple2BoundingBox(bboxes[0])
                sys.exit('finish!!!')

        
        

        
    # saliencyTester.test(saliency_model_file, num_classes, dataloaders_dict['test'], test_names, input_size, saliency_model_config, numDiPerVideos)

__main__()