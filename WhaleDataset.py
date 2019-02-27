
import pandas as pd
import os
import torch
import numpy as np
import PIL

import AugmentImage


class WhaleDataset(torch.utils.data.Dataset):
    '''
        class implementing 
            __len__(self, )
            __getitem__(self, index)
        to be used by PyTorch to retrieve data
    '''

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
        class_labels = data.loc[:,('Whale_ID')].unique()
        class_labels_numeric = np.arange(len(class_labels))
        # print('Classes {}'.format(len(class_labels_numeric))) #i.e. 5004
        dict_of_unique_values = dict()
        for (label, label_num) in zip(class_labels, class_labels_numeric):
            dict_of_unique_values[label] = label_num
        data.Whale_ID = data.Whale_ID.apply(lambda x: dict_of_unique_values[x])
        # CrossEntropyLoss expects class labels 0..C-1 where C is the maximum number of classes
        # so, class labels have to be encoded as numbers
        return data, dict_of_unique_values

    def __init__(self):
        (self.data, self.labels_encodings) = WhaleDataset.preprocessData()
        self.length = self.data.shape[0]
        print('Dataset size {}'.format(self.length))
    
    def get_data(self):
        return self.data
    def get_image_RGB(self, index):
        img_name=self.data.iloc[index].Image
        img = PIL.Image.open(os.path.join(WhaleDataset.IMAGESPATH, img_name))
        return img.convert('RGB')
    def compute_transformation(self, index):
        image = self.get_image_RGB(index)
        #check if image is grayscale and replicate channels
        # size = [x for x in img.shape]
        # if (len(size) == 2):
        #     img = np.stack([img]*3, axis=0)
        # select randomly a transformed image from the list
        return AugmentImage.TransformedImages()(image)
    
    def __getitem__(self, index):
        transformed_images, _ = self.compute_transformation(index)
        position = np.random.randint(0,len(transformed_images))
        return transformed_images[position], self.data.iloc[index].Whale_ID

    def __len__(self):
        return self.length

class BatchWhaleDataset(WhaleDataset):
    def __init__(self):
        super(BatchWhaleDataset, self).__init__()
    def __getitem__(self, index):
        transformed_images, _ = super(BatchWhaleDataset,self).compute_transformation(index)
        return transformed_images, super(BatchWhaleDataset,self).get_data().iloc[index].Whale_ID

class PatchesWhaleDataset(WhaleDataset):
    def __init__(self):
        super(PatchesWhaleDataset, self).__init__()
    
    def compute_transformation(self, index):
        image = super(PatchesWhaleDataset,self).get_image_RGB(index)
        return AugmentImage.Patches_of_Images()(image)
    
    def __getitem__(self, index):
        transformed_images, _ = self.compute_transformation(index)
        return transformed_images, super(PatchesWhaleDataset,self).get_data().iloc[index].Whale_ID