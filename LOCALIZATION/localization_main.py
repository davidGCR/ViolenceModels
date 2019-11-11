
import sys
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/MyCode')
import argparse
import AnomalyCrime.transforms_anomaly as transforms_anomaly
import torch
import AnomalyCrime.anomaly_initializeDataset as anomaly_initializeDataset
import saliencyTester
import constants
import os
import cv2

 
def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saliencyModelFile", type=str)
    parser.add_argument("--batchSize", type=int, default=3)
    # parser.add_argument("--numEpochs", type=int, default=10)
    parser.add_argument("--numWorkers", type=int, default=1)
    parser.add_argument("--numDiPerVideos", type=int, default=5)
    args = parser.parse_args()
    maxNumFramesOnVideo = 0
    videoSegmentLength = 16
    numDiPerVideos = args.numDiPerVideos
    positionSegment = 'random'
    num_classes = 2
    input_size = 224
    transforms = transforms_anomaly.createTransforms(input_size)
    dataset_source = 'frames'
    batch_size = args.batchSize
    num_workers = args.numWorkers
    saliency_model_file = args.saliencyModelFile
    shuffle = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    saliency_model_file = args.saliencyModelFile
    saliency_model_config = saliency_model_file

    path_dataset = constants.PATH_UCFCRIME2LOCAL_FRAMES_REDUCED
    train_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Train_split_AD.txt')
    test_videos_path = os.path.join(constants.PATH_UCFCRIME2LOCAL_README, 'Test_split_AD.txt')
    dataloaders_dict, test_names = anomaly_initializeDataset.initialize_final_anomaly_dataset(path_dataset, train_videos_path, test_videos_path, dataset_source, batch_size,
                                                        num_workers, numDiPerVideos, transforms, maxNumFramesOnVideo, videoSegmentLength, positionSegment, shuffle)
    saliencyTester.test(saliency_model_file, num_classes, dataloaders_dict['test'], test_names, input_size, saliency_model_config, numDiPerVideos)

 __main__()