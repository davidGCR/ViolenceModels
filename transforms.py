import torchvision.transforms as transforms
import numpy as np
from initialize_dataset import createDataset, getDataLoader
from tqdm import tqdm

def createTransforms(input_size):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4770381, 0.4767955, 0.4773611], [0.11147115, 0.11427314, 0.11617025])
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.4770381, 0.4767955, 0.4773611], [0.11147115, 0.11427314, 0.11617025])
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    return data_transforms

def compute_mean_std(dataloader):
    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    for i, data in tqdm(enumerate(dataloader, 0)):
        # shape (batch_size, 3, height, width)
        inputs, labels = data
        numpy_image = inputs.numpy()
        
        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std0 = np.std(numpy_image, axis=(0,2,3))
        batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
        
        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    pop_std1 = np.array(pop_std1).mean(axis=0)
    return pop_mean, pop_std0, pop_std1

def __main__():
    hockey_path_violence = "/media/david/datos/Violence DATA/HockeyFights/frames/violence"
    hockey_path_noviolence = "/media/david/datos/Violence DATA/HockeyFights/frames/nonviolence"
    x, y, numFramesAll = createDataset(hockey_path_violence, hockey_path_noviolence)  #ordered
    debugg_mode = False
    avgmaxDuration = 1.66
    interval_duration = 0.3
    num_workers = 4
    # input_size = 224
    numDiPerVideos = 1
    batch_size = 32
    dataset_source = "frames"
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    data_loader = getDataLoader(x, y, data_transform, numDiPerVideos, dataset_source, avgmaxDuration, interval_duration, batch_size, num_workers, debugg_mode)
    pop_mean, pop_std0, pop_std1 = compute_mean_std(data_loader)
    print('pop_mean, pop_std0, pop_std1', pop_mean, pop_std0, pop_std1)

# __main__()