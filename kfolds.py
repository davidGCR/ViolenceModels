import numpy as np
def partitions(number, k):
    '''
    Distribution of the folds
    Args:
        number: number of patients
        k: folds number
    '''
    n_partitions = np.ones(k) * int(number/k)
    n_partitions[0:(number % k)] += 1
    return n_partitions

def get_indices(n_splits, subjects):
    '''
    Indices of the set test
    Args:
        n_splits: folds number
        subjects: number of patients
        frames: length of the sequence of each patient
    '''
    l = partitions(subjects, n_splits)
    
    # fold_sizes = l * frames
    # indices = np.arange(subjects * frames).astype(int)
    indices = np.arange(subjects).astype(int)
    current = 0
    for fold_size in l:
        start = current
        stop =  current + fold_size
        current = stop
        yield(indices[int(start):int(stop)])

def k_folds(n_splits, subjects):
    '''
    Generates folds for cross validation
    Args:
        n_splits: folds number
        subjects: number of patients
        frames: length of the sequence of each patient
    '''
    # indices = np.arange(subjects * frames).astype(int)
    indices = np.arange(subjects ).astype(int)
    for test_idx in get_indices(n_splits, subjects):
        train_idx = np.setdiff1d(indices, test_idx)
        yield train_idx, test_idx

# def __main__():
#     print('test23 kfolfs...')
#     num_folds = 3
#     for train_idx, test_idx in k_folds(n_splits = 3, subjects = 20):
#         print('train_idx',len(train_idx))
#         print(train_idx)
#         print('test_idx',len(test_idx))
#         print(test_idx)
        # dataset_train = NNDataset(indices = train_idx)
        # dataset_test = NNDataset(indices = test_idx)
        # train_loader = torch.utils.data.DataLoader(dataset = dataset_train, batch_size = batch_size_train, **kwargs)
        # test_loader = torch.utils.data.DataLoader(dataset = dataset_test, batch_size = batch_size_test, **kwargs)
        # for epoch in range(1, num_epochs + 1):
        #     train(model, optimizer, epoch, device, train_loader, log_interval)
        #     test(model, device, test_loader)

    
    


# __main__()
# get_indices()