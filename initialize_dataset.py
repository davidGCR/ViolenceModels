
import torch
import os
import glob
from ViolenceDatasetV2 import ViolenceDatasetVideos
from MaskDataset import MaskDataset
from saliency_model import *


def createDatasetViolence(path): #only violence videos
  imagesF = []
  list_violence = os.listdir(path)
  list_violence.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  for target in list_violence:
      d = os.path.join(path, target)
      imagesF.append(d)
  labels = list([1] * len(imagesF))
  numFrames = [len(glob.glob1(imagesF[i], "*.jpg")) for i in range(len(imagesF))]
  return imagesF, labels, numFrames

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
    print('Dataset, Labels, NumFrames: ', len(Dataset), len(Labels), len(NumFrames))
    return Dataset, Labels, NumFrames

def getDataLoader(x, y, data_transform, numDiPerVideos, dataset_source, avgmaxDuration, interval_duration, batch_size, num_workers, debugg_mode):
    dataset = ViolenceDatasetVideos( dataset=x, labels=y, spatial_transform=data_transform, source=dataset_source,
            interval_duration=interval_duration,difference=3, maxDuration=avgmaxDuration, nDynamicImages=numDiPerVideos, debugg_mode=debugg_mode, )
    dataloader = torch.utils.data.DataLoader( dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

def getDataLoaders(datasetType, train_x, train_y, test_x, test_y, data_transforms, numDiPerVideos, dataset_source, avgmaxDuration, interval_duration, batch_size, num_workers, debugg_mode, salModelFile):
    image_datasets = None
    if datasetType == 'hockey':
        image_datasets = {
            "train": ViolenceDatasetVideos( dataset=train_x, labels=train_y, spatial_transform=data_transforms["train"], source=dataset_source,
                interval_duration=interval_duration,difference=3, maxDuration=avgmaxDuration, nDynamicImages=numDiPerVideos, debugg_mode=debugg_mode, ),
            "test": ViolenceDatasetVideos( dataset=test_x, labels=test_y, spatial_transform=data_transforms["test"], source=dataset_source,
                interval_duration=interval_duration, difference=3, maxDuration=avgmaxDuration, nDynamicImages=numDiPerVideos, debugg_mode=debugg_mode, )
        }
        
    elif datasetType == 'masked':
        net = saliency_model(num_classes=2)
        # net = net.cuda()
        net = torch.load(salModelFile, map_location=lambda storage, loc: storage)

        image_datasets = {
            "train": MaskDataset( dataset=train_x, labels=train_y, spatial_transform=data_transforms["train"], source=dataset_source,
                difference=3, maxDuration=avgmaxDuration, nDynamicImages=numDiPerVideos, saliency_model=net ),
            "test": MaskDataset( dataset=test_x, labels=test_y, spatial_transform=data_transforms["test"], source=dataset_source,
                difference=3, maxDuration=avgmaxDuration, nDynamicImages=numDiPerVideos, saliency_model=net )
        }

    dataloaders_dict = {
        "train": torch.utils.data.DataLoader( image_datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "test": torch.utils.data.DataLoader( image_datasets["test"], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
    }
    return dataloaders_dict