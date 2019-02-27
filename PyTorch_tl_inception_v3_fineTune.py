# Using Pytorch's SubsetRandomSampler to split data into train, validation

import pandas as pd
import os
import time
import torch
import numpy as np
import torchvision
from copy import deepcopy
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
import cv2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class WhaleDataset:
    IMAGESPATH = '../data/train/'
    LABELSPATH = '../data/'

    @staticmethod
    def preprocessData(data):
        '''
        encodes class labels as 0, 1.. inplace
        returns 
            - a pandas DataFrame without 'new_whale' attribute
            - a dictionary containing original class labels and corresponding numeric class labels
        '''
        data.loc[:,('Whale_ID')].str.strip()
        # data = data[data.Whale_ID!='new_whale']
        class_labels = data.loc[:,('Whale_ID')].unique()
        class_labels_numeric = np.arange(len(class_labels))
        # print('Classes {}'.format(len(class_labels_numeric))) #i.e. 5005
        dict_of_unique_values = dict()
        for (label, label_num) in zip(class_labels, class_labels_numeric):
            dict_of_unique_values[label] = label_num
        data.Whale_ID = data.Whale_ID.apply(lambda x: dict_of_unique_values[x])
        # CrossEntropyLoss expects class labels 0..C-1 where C is the maximum number of classes
        # so, class labels have to ve encoded as numbers
        return data, dict_of_unique_values

    def __init__(self, transform=None):
        d = pd.read_csv(os.path.join(WhaleDataset.LABELSPATH, 'train.csv'))
        d.columns = ['Image', 'Whale_ID']
        (self.data, self.labels_encodings) = WhaleDataset.preprocessData(d)
        self.length = self.data.shape[0]
        print('Dataset size {}'.format(self.length))
        self.transform = transform

    def __getitem__(self, index):
        # This method should return only 1 sample and label 
        # (according to "index"), not the whole dataset
        # So probably something like this for you:
        img_name=self.data.iloc[index].Image
        img = cv2.imread(os.path.join(WhaleDataset.IMAGESPATH, img_name))
        #check if image is grayscale and replicate channels
        size = [x for x in img.shape]
        if (len(size) == 2):
            img = np.stack([img]*3, axis=0)
        if self.transform is not None:
            img = self.transform(img)
        # print('Image retrieved: {}, {} channels'.format(img_name, img.shape))
        return img.to(device), self.data.iloc[index].Whale_ID

    def __len__(self):
        return self.length

def randomSplitter(dataset_size, random_seed=42, margin=0.2):
    '''
        randomly shuffles the range(0, dataset_size)
        and return 2 slices, separated according to margin from the dataset_size
    '''
    indices = list(range(dataset_size))
    split_index = int(np.floor(margin * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split_index:], indices[:split_index]
    return train_indices, val_indices

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((299,299)),
    torchvision.transforms.ColorJitter(brightness=.04, hue=.05, saturation=.05),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(20, resample=Image.BILINEAR),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.443,0.453,0.461], [0.51,0.48,0.5])
    ])
dataset = WhaleDataset(transforms)
model = torchvision.models.inception_v3(num_classes=5005,aux_logits=False).to(device)



def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=25, batch_size=16):
    since = time.time()
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # create dataloader of randomly shuffled data
        train_indices, val_indices = randomSplitter(len(dataset))
        dataset_sizes={
                    'train':len(train_indices),
                    'val': len(val_indices)}
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        dataloader = {
                    'train':torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, 
                                                sampler=train_sampler),
                    'val': torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                                sampler=valid_sampler)
                    }

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                lr_scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for (inputs, labels) in dataloader[phase]:
                # for (Img, label) in zip(inputs, labels):
                # print(labels)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = deepcopy(model.state_dict())
                torch.save(best_model_wts, 'inception_v3_5004_categories.pth')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

optimizer_v3 = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model = train_model(model=model,
                    criterion=torch.nn.CrossEntropyLoss(reduction='mean'),
                    optimizer=optimizer_v3
                    # lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer_v3, step_size=7, gamma=0.1)
                    )
