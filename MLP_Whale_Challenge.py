import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import os

DATA_PATH = '/home/loxor/Documents/Humpback_Whale_Identification/data/'
data_raw = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
# remove new_whale
data_raw = data_raw[data_raw.Id != 'new_whale']
print(data_raw.head(10))

