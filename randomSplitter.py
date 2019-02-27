import numpy as np


def randomSplitter(dataset_size, random_seed=42, margin=0.2):
    '''
        randomly shuffles the range(0, dataset_size)
        and return 2 slices, separated according to margin from the dataset_size
    '''
    indices = list(range(dataset_size))
    split_index1, split_index2 = int(np.floor(margin * dataset_size)), int(np.floor(margin * 2 * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[split_index2:], indices[:split_index1], indices[split_index1:split_index2]
    print('Datasets train {}, val {}, test {}'.format(len(train_indices), len(val_indices), len(test_indices)))
    return train_indices, val_indices, test_indices