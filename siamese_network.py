# code modified from
# https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/blob/master/Siamese-networks-medium.ipynb

import torch, torchvision, os
import numpy as np
import pandas as pd 
import PIL.Image as Image
import PIL
from sklearn.neighbors import KNeighborsClassifier
import smaller_vgg



inputSize = (28,28)
imageConversion = 'RGB'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_test_split(random_seed = 42, margin = 0.3, only_seen_classes=True, only_classes_frequency_ge=None):
    '''
        @param random_seed - for reproducibility
        @param margin - proportion of validation set is margin * total data items in csv file
        @param only_seen_classes - if True, validation set contains a subset of class labels from training set
        @param only_classes_frequency_ge - if not None, only classes with frequency above or equal to only_classes_frequency_ge are used to create training set, validation set
    '''
    # acquire data from csv
    data = pd.read_csv(os.path.join('../data/', 'train.csv'))
    data.columns = ['Image', 'Whale_ID']
    data.loc[:,('Whale_ID')].str.strip()
    data = data[data.Whale_ID!='new_whale']
    if only_classes_frequency_ge is not None:
        stats_inp = pd.DataFrame(data['Whale_ID'].value_counts()).reset_index()
        stats_inp.columns=['Whale_ID','Whale_freq']
        stats_inp = stats_inp[stats_inp.Whale_freq >= only_classes_frequency_ge]
        print('Class labels frequencies >= {freq} are present in training and validation'.format(freq=only_classes_frequency_ge))
        print(stats_inp)
        stats_inp = stats_inp.Whale_ID
        data = data[data.Whale_ID.isin(stats_inp)]
    # convert string labels to numeric values in range 0,1,2 etc
    labels_as_string = data.Whale_ID.unique()
    labels_as_num = np.arange(0,len(labels_as_string))
    dictionary_string_num_labels = dict()
    for i in range(0,len(labels_as_string)):
        dictionary_string_num_labels[labels_as_string[i]] = labels_as_num[i]
    
    data.Whale_ID = data.Whale_ID.apply(lambda x: dictionary_string_num_labels[x])

    # create new dataframe storing frequency of classes
    # x = pd.concat([data.Whale_ID.value_counts()],axis=1).reset_index()
    # x.columns=['Whale_ID', 'Frequency_Whale']
    # # singleton classes are removed so that cross validation is performed
    # x = x[x.Frequency_Whale > 1]
    # data = data[~data.Whale_ID.isin(x)].dropna()

    # split dataframe according to margin, randomly
    dataset_size = data.shape[0]
    indices = list(range(dataset_size))
    split_index = int(np.floor(margin * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split_index:], indices[:split_index]
    train_dataset = data.iloc[train_indices].reset_index(drop=True)
    val_dataset = data.iloc[val_indices].reset_index(drop=True)
    print('train dataset size {}. val dataset size {}'.format(train_dataset.shape, val_dataset.shape))
    if only_seen_classes == True:
        non_duplicate_train_classes = data.iloc[train_indices].Whale_ID.unique()
        val_dataset = val_dataset[val_dataset.Whale_ID.isin(non_duplicate_train_classes)]
        print('val dataset size - containing only class labels from train dataset ', val_dataset.shape)
    return train_dataset, val_dataset


class SiameseNetworkTrainDataset(torch.utils.data.Dataset):
    def __init__(self,data,transform=None):
        self.data = data   
        self.transform = transform

    def get_random_image(self):
        '''
        returns a pandas series from a dataframe
        '''
        return self.data.iloc[np.random.randint(0,self.__len__())]
    
    def __getitem__(self,index):
        img0 = self.get_random_image()
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = np.random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1 = self.get_random_image()
                if img0.Whale_ID == img1.Whale_ID:
                    break
        else:
            while True:
                #keep looping till a different class image is found
                img1 = self.get_random_image() 
                if img0.Whale_ID != img1.Whale_ID:
                    break

        image0 = Image.open(os.path.join('../data/train/', img0.Image))
        image1 = Image.open(os.path.join('../data/train/', img1.Image))
        image0 = image0.convert(imageConversion) #grayscale
        image1 = image1.convert(imageConversion)
        
        # if self.should_invert: # invert colors
        #     image0 = PIL.ImageOps.invert(image0)
        #     image1 = PIL.ImageOps.invert(image1)

        image0 = torchvision.transforms.functional.resize(image0, inputSize)
        image1 = torchvision.transforms.functional.resize(image1, inputSize)
        
        if self.transform is not None:
            image0 = self.transform(image0)
            image1 = self.transform(image1)
        
        return image0, image1 , torch.from_numpy(np.array([int(img1.Whale_ID!=img0.Whale_ID)],dtype=np.float32))
    
    def __len__(self):
        return self.data.shape[0] # rows



class SiameseNetworkValDataset(torch.utils.data.Dataset):
    def __init__(self,data,transform=None):
        self.data = data   
        self.transform = transform
    
    def __getitem__(self,index):
        image, label = self.data.iloc[index]
        # print(image, label)
        image = Image.open(os.path.join('../data/train/', image)).convert(imageConversion) # grayscale

        # if self.should_invert: # invert colors
        #     image0 = PIL.ImageOps.invert(image0)
        #     image1 = PIL.ImageOps.invert(image1)

        image = torchvision.transforms.functional.resize(image, inputSize)
        
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.LongTensor([label])
    
    def __len__(self):
        return self.data.shape[0] # rows

class SiameseNetwork(torch.nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # self.cnn1 = torch.nn.Sequential(
        #     torch.nn.ReflectionPad2d(padding=1),
        #     torch.nn.Conv2d(in_channels=3, out_channels=15, kernel_size=3),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.MaxPool2d(kernel_size=4),
        #     # torch.nn.BatchNorm2d(num_features=15),
            
        #     torch.nn.ReflectionPad2d(padding=1),
        #     torch.nn.Conv2d(in_channels=15, out_channels=30, kernel_size=3),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.MaxPool2d(kernel_size=4),
        #     # torch.nn.BatchNorm2d(num_features=30),


        #     torch.nn.ReflectionPad2d(padding=1),
        #     torch.nn.Conv2d(in_channels=30, out_channels=30, kernel_size=3),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.BatchNorm2d(num_features=30)
        # )

        # self.fc1 = torch.nn.Sequential(
        #     torch.nn.Linear(5880, 50),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(50, 10),
        #     torch.nn.Sigmoid() # adjustement proposed to contrastive loss, as mentioed here
        #     # https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24
            
            
        #     # torch.nn.ReLU(inplace=True),
        #     # torch.nn.Linear(500, 5)
        # )

        self.cnn1 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(padding=1),
            torch.nn.Conv2d(in_channels=3,out_channels=8, kernel_size=3),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(num_features=8),
            
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(num_features=32),


            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_channels=32,out_channels=16, kernel_size=3),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(num_features=16)
        )

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(12544, 500), #12544
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(500, 100),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(100, 100),
            torch.nn.Sigmoid())

        # self.model = smaller_vgg.vgg11_modif().to(device)
    def forward_once(self, x):
        # torchvision.models.vgg11(num_classes=100).to(device)
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        # print('output shape of cnn block ',output.shape)
        output = self.fc1(output)
        return output
        # return  self.model(x)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2



class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (0.5 * label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        # print('dist between feat vectors, ', loss_contrastive)

        return loss_contrastive #  * pow(10,-1)

# this same function was used efore in feat extraction and finetuning
def batch_train_model(model, criterion, optimizer, num_epochs=25, batch_size=10):
    # with open('code_statistics/' + get_time_now() + model.__class__.__name__ + '_train_statistics.csv', 'a', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',')
    #     header=['phase', 'batch loss', 'batch accuracy']
    #     writer.writerow(header)

    # best_model_wts = deepcopy(model.state_dict())
    # best_epoch_acc = 0.0


    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.443,0.453,0.461], [0.51,0.48,0.5])
        ])
    
    train_data, val_data = train_test_split(only_classes_frequency_ge=50)
    train_data = SiameseNetworkTrainDataset(data=train_data, transform=transform)
    val_data = SiameseNetworkValDataset(data=val_data, transform=transform)

    dataset_sizes={
                'train':len(train_data),
                'val': len(val_data)}
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # create dataloader of randomly shuffled data
        dataloader = {
        'train':torch.utils.data.DataLoader(dataset=train_data,
                                            # batch_size=batch_size,
                                            batch_sampler = torch.utils.data.BatchSampler(sampler=torch.utils.data.RandomSampler(data_source=train_data),batch_size=batch_size,drop_last=True),
                                            num_workers=8
                                            ),
        'val': torch.utils.data.DataLoader(dataset=val_data,
                                           batch_size=batch_size,
                                           num_workers=8,
                                           shuffle=True
                                        )}

        epoch_loss = 0.0
        epoch_corrects = 0

        for phase in [ 'train', 'val']:
            if phase == 'train':
                for batch_idx, (inputs) in enumerate(dataloader[phase]):
                    # inputs = inputs.to(device)
                    # inputs is a list of tensors (3 x batch_size x image_W x image_H)
                    # inputs = torch.stack([x.squeeze() for x in inputs]).to(device)
                    # labels = torch.LongTensor([labels.data] * inputs.size(0)).to(device)
                
                    model.train()  # Set model to training mode;mandatory because of batchNorm, dropout
                    optimizer.zero_grad()
                    torch.set_grad_enabled(True)
                    img0, img1 , label = inputs
                    img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
                    output1,output2 = model.forward(img0,img1)
                    loss = criterion(output1, output2, label)
                    loss_contrastive = loss / batch_size
                    loss_contrastive.backward()
                    optimizer.step()
                    print("Train: Epoch number {:{width}} batch {:{width}} Current loss {:{width}.4f}\n".\
                        format(epoch, batch_idx, loss_contrastive.item(), width=4))
                    param_count = 0
                    # for parameters in model.parameters():
                    #     param_count  = param_count + 1
                    #     if param_count > 1:
                    #         break
                    #     print('Parameter')
                    #     print(parameters.data[0])
            else:
                model.eval()   # Set model to evaluate mode
                torch.set_grad_enabled(False)

                feat_vectors, labels_vector = [], []
                allClassesTraining = train_data.data #.drop_duplicates(subset=['Whale_ID']).reset_index(drop=True)
                print('Non duplicate entries(i.e. class labels) as feature vectors: ', allClassesTraining.shape[0])
                for j in range(0,allClassesTraining.shape[0]):
                    image = Image.open(os.path.join('../data/train/', allClassesTraining.loc[j,('Image')])).convert(imageConversion)
                    image = torchvision.transforms.functional.resize(image, inputSize)
                    feat_vectors.append(np.asarray(model.forward_once(train_data.transform(image).unsqueeze(0).cuda()).cpu().data))
                    labels_vector.append(allClassesTraining.loc[j,('Whale_ID')])
                
                feat_vectors = np.asarray(feat_vectors)
                labels_vector = np.asarray(labels_vector)
                num, x1, x2 = feat_vectors.shape
                feat_vectors = feat_vectors.reshape(num, x1 * x2)
                classifier = KNeighborsClassifier(n_neighbors=allClassesTraining.shape[0])
                # print('Shapes:', feat_vectors.shape, labels_vector.shape)
                classifier.fit(feat_vectors, labels_vector)
                
                for batch_idx, (inputs, labels) in enumerate(dataloader[phase]):
                    labels_np = np.asarray(labels).reshape(labels.shape[0])
                    # print('correct labels in batch ', labels_np)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model.forward_once(inputs).cpu()
                    # print('output shape ', outputs.shape)
                    prediction = classifier.predict(outputs)
                    print('prediction', prediction)
                    corrects = np.sum(prediction == labels_np)
                    print("Eval: Epoch number {:{width}} batch {:{width}} accuracy {:{width}.4f}\n".\
                        format(epoch, batch_idx, corrects/batch_size, width=4))
                    epoch_corrects += corrects
                print('Per epoch: {phase} aacuracy = {acc}'.format(phase = phase, acc = epoch_corrects / dataset_sizes[phase]))

    #         # deep copy the model
    #         epoch_loss = epoch_loss / dataset_sizes[phase]
    #         epoch_acc = epoch_corrects.double() / dataset_sizes[phase]
    #         print('Epoch loss {0:.4f} and accuracy {0:.4f}'.format(epoch_loss, epoch_acc))
    #         if phase == 'val' and epoch_acc > best_acc:
    #             best_acc = epoch_acc
    #             best_model_wts = deepcopy(model.state_dict())
    #             time_now = get_time_now()
    #             best_model_name = time_now + model.__class__.__name__ + '_featExtr.pth'
    #             torch.save(best_model_wts, best_model_name)
    #             print('Model saved as ',best_model_name)
    # print('Best val Acc: {:4f}'.format(best_acc))

    # # load best model weights
    # model.load_state_dict(best_model_wts)
    # return model

if __name__=='__main__':
    
    # train_dataloader = torch.utils.data.DataLoader(dataset=dataset,
    #     shuffle=True,
    #     num_workers=8,
    #     batch_size=25)

    # model = SiameseNetwork().cuda()
    # criterion = ContrastiveLoss()
    # optimizer = torch.optim.Adam(net.parameters(),lr = 0.001)

    # counter = []
    # loss_history = [] 
    # iteration_number= 0

    # for epoch in range(0, 25):
    #     for i, data in enumerate(train_dataloader):
    #         img0, img1 , label = data
    #         img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
    #         optimizer.zero_grad()
    #         output1,output2 = net(img0,img1)
    #         loss_contrastive = criterion(output1, output2, label)
    #         loss_contrastive.backward()
    #         optimizer.step()
    #         if i % 10 == 0 :
    #             print("Epoch number {:{width}} iter {:{width}} Current loss {:{width}.4f}\n".format(epoch,i,loss_contrastive.item(),width=4))
    #             iteration_number +=10
    #             counter.append(iteration_number)
    #             loss_history.append(loss_contrastive.item())
        
    # show_plot(counter,loss_history)
    
    model = SiameseNetwork().cuda()
    # if using modified vgg uncomment below lines
    # print('model description')
    # for param in model.model.features.parameters():
    #     param.requires_grad=False
    # parameter_count = 0
    # for param in model.parameters():
    #     if param.requires_grad == True:
    #         print(param.shape)
    #         parameter_count = parameter_count + np.prod(param.shape)
    # print('{} trainable parameters'.format(parameter_count))
    print('Press any keyboard to start training the model...')
    _=input()
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.005)
    batch_train_model(model=model,
                    criterion=ContrastiveLoss(),
                    optimizer=optimizer,
                    batch_size=15)

                    # lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1))
    # problems in choosing val indices:
    # split into train/test dataloader