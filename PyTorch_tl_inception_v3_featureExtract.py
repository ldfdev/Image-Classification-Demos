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
import PIL
from datetime import datetime

from AugmentImage import TransformedImages

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class WhaleDataset(torch.utils.data.Dataset):
    IMAGESPATH = '../data/train/'
    LABELSPATH = '../data/'

    @staticmethod
    def preprocessData():
        '''
        encodes class labels as 0, 1.. inplace
        returns 
            - a pandas DataFrame without 'new_whale' attribute
            - a dictionary containing original class labels and corresponding numeric class labels
        '''
        data = pd.read_csv(os.path.join(WhaleDataset.LABELSPATH, 'train.csv'))
        data.columns = ['Image', 'Whale_ID']
        data.loc[:,('Whale_ID')].str.strip()
        data = data[data.Whale_ID!='new_whale']
        
        # print('Classes {}'.format(len(class_labels_numeric))) #i.e. 5004

        # CrossEntropyLoss expects class labels 0..C-1 where C is the maximum number of classes
        # so, class labels have to be encoded as numbers
        only_classes_frequency_ge = 50
        if only_classes_frequency_ge is not None:
            stats_inp = pd.DataFrame(data['Whale_ID'].value_counts()).reset_index()
            stats_inp.columns=['Whale_ID','Whale_freq']
            stats_inp = stats_inp[stats_inp.Whale_freq >= only_classes_frequency_ge]
            print('Class labels frequencies >= {freq} are present in training and validation'.format(freq=only_classes_frequency_ge))
            print(stats_inp)
            stats_inp = stats_inp.Whale_ID
            data = data[data.Whale_ID.isin(stats_inp)]
        # all classes must be in tange 0.. numclasses. otherwise Assertion `t >= 0 && t < n_classes` failed.
        class_labels = data.loc[:,('Whale_ID')].unique()
        class_labels_numeric = np.arange(len(class_labels))
        dict_of_unique_values = dict()
        for (label, label_num) in zip(class_labels, class_labels_numeric):
            dict_of_unique_values[label] = label_num
        data.Whale_ID = data.Whale_ID.apply(lambda x: dict_of_unique_values[x])
        return data, dict_of_unique_values

    def __init__(self):
        (self.data, self.labels_encodings) = WhaleDataset.preprocessData()
        self.length = self.data.shape[0]
        print('Dataset size {}'.format(self.length))
        print('Different classes {}'.format(self.get_class_identities()))

    def get_class_identities(self):
        '''
            returns number of unique IDs in self.data dataframe object
        '''
        return self.data.Whale_ID.unique().shape[0]

    def __getitem__(self, index):
        # This method should return only 1 sample and label 
        # (according to "index"), not the whole dataset
        img_name=self.data.iloc[index].Image
        img = PIL.Image.open(os.path.join(WhaleDataset.IMAGESPATH, img_name))
        img = img.convert('RGB')
        #check if image is grayscale and replicate channels
        # size = [x for x in img.shape]
        # if (len(size) == 2):
        #     img = np.stack([img]*3, axis=0)
        # select randomly a transformed image from the list
        transformed_images, _ = TransformedImages()(img)
        # print('__getitem__: image{}. Available transformed images {}.'.format(img_name, len(transformed_images)))
        # for image in transformed_images:
        #     print(type(image))
        position = np.random.randint(0,len(transformed_images))
        # print('Image retrieved: {}, {} channels'.format(img_name, img.shape))
        return transformed_images[position].to(device), self.data.iloc[index].Whale_ID

    def __len__(self):
        return self.length

from randomSplitter import randomSplitter




dataset = WhaleDataset()

model = torchvision.models.inception_v3(pretrained=True)
model.aux_logits = False
delattr(model, 'AuxLogits')
for param in model.parameters():
    param.requires_grad = False
ftrs = model.fc.in_features
model.fc = torch.nn.Linear(ftrs, dataset.get_class_identities())
model = model.to(device)

def train_model(model, criterion, optimizer, num_epochs=25, batch_size=10,model_name='inception_v3_5004_categories_featExtr.pth'):
    since = time.time()
    best_model_wts = None # deepcopy(model.state_dict())
    best_acc = 0.0
    train_indices, val_indices, _ = randomSplitter(len(dataset))
    dataset_sizes={
                'train':len(train_indices),
                'val': len(val_indices)}
    epoch_best_acc = 0
    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # create dataloader of randomly shuffled data
        dataloader = {
        'train':torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size, 
                                            sampler=SubsetRandomSampler(train_indices)),
        'val': torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           sampler=SubsetRandomSampler(val_indices))}
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            epoch_acc = 0
            if phase == 'train':
                model.train()  # Set model to training mode;mandatory because of batchNorm, dropout
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch_idx, (inputs, labels) in enumerate(dataloader[phase]):
                # for (Img, label) in zip(inputs, labels):
                # print(labels)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss = loss.item() * inputs.size(0)
                running_corrects = torch.sum(preds == labels.data)

                batch_loss = running_loss / batch_size # dataset_sizes[phase]
                batch_acc = running_corrects.double() / batch_size # dataset_sizes[phase]
                epoch_acc += running_corrects.double()
                print('\tBatch {} {} Loss: {:.4f} Acc: {:.4f}'.format(
                    batch_idx, phase, batch_loss, batch_acc))

            # deep copy the model
            epoch_acc = epoch_acc / dataset_sizes[phase]
            print('Epoch accuracy for {} is {}'.format(phase, epoch_acc))
            if phase == 'val' and epoch_acc > epoch_best_acc:
                epoch_best_acc = epoch_acc
                best_model_wts = deepcopy(model.state_dict())
                modelname = str(datetime.now().time()) + model_name
                torch.save(best_model_wts, modelname)
                print('New best model saved as ', modelname)
                

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

optimizer_v3 = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
modelName = 'inception_v3_' + str(model.fc.out_features) + '_categories_featExtr.pth'
model = train_model(model=model,
                    criterion=torch.nn.CrossEntropyLoss(reduction='mean'),
                    optimizer=optimizer_v3,
                    model_name=modelName)
