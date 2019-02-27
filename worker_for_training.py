# Using Pytorch's SubsetRandomSampler to split data into train, validation

import os
import time, csv
from datetime import datetime
import torch
import numpy as np
import torchvision
from copy import deepcopy
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('Execution time for {}:  {:.0f}m {:.0f}s'.format(method.__name__,\
            time_elapsed // 60, time_elapsed % 60))
        return result
    return timed

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

@timeit
def train_model(model, dataset, criterion, optimizer, lr_scheduler, num_epochs=25, batch_size=10):
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0
    train_indices, val_indices = randomSplitter(len(dataset))
    dataset_sizes={
                'train':len(train_indices),
                'val': len(val_indices)}
    
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
            if phase == 'train':
                lr_scheduler.step()
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
                running_loss = loss.item() * inputs.size(0)
                running_corrects = torch.sum(preds == labels.data)

                batch_loss = running_loss / batch_size # dataset_sizes[phase]
                batch_acc = running_corrects.double() / batch_size # dataset_sizes[phase]

                print('\tBatch {} {} Loss: {:.4f} Acc: {:.4f}'.format(
                    batch_idx, phase, batch_loss, batch_acc))

                # deep copy the model
                if phase == 'val' and batch_acc > best_acc:
                    best_acc = batch_acc
                    best_model_wts = deepcopy(model.state_dict())
                    torch.save(best_model_wts, 'inception_v3_5004_categories_featExtr.pth')
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def get_time_now():
    return datetime.now().strftime('%a_%b_%H_%M_%S_')

@timeit
def batch_train_model(model, dataset, criterion, optimizer, lr_scheduler, num_epochs=25, batch_size=10):
    with open('code_statistics/' + get_time_now() + model.__class__.__name__ + '_train_statistics.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        header=['phase', 'batch loss', 'batch accuracy']
        writer.writerow(header)

        best_model_wts = deepcopy(model.state_dict())
        best_epoch_acc = 0.0
        train_indices, val_indices = randomSplitter(len(dataset))
        dataset_sizes={
                    'train':len(train_indices),
                    'val': len(val_indices)}
        
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
                epoch_loss = 0.0
                epoch_corrects = 0

                for batch_idx, (inputs, labels) in enumerate(dataloader[phase]):
                    # inputs = inputs.to(device)
                    # labels = labels.to(device)
                    # inputs is a list of tensors
                    inputs = torch.stack([x.squeeze() for x in inputs]).to(device)
                    labels = torch.LongTensor([labels.data] * inputs.size(0)).to(device)
                    if phase == 'train':
                        lr_scheduler.step()
                        model.train()  # Set model to training mode;mandatory because of batchNorm, dropout
                        torch.set_grad_enabled(True)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    else:
                        model.eval()   # Set model to evaluate mode
                        torch.set_grad_enabled(False)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs.data, 1)
                    print('Predictions size ', preds.size())
                    print('Correct class', labels.data[0])
                    for prediction in preds.data:
                        print(prediction, end=' ')
                    # statistics
                    running_loss = loss.item() * inputs.size(0)
                    running_corrects = torch.sum(preds == labels.data)

                    batch_loss = running_loss / inputs.size(0) # batch_size # dataset_sizes[phase]
                    batch_acc = running_corrects.double() / inputs.size(0) # batch_size # dataset_sizes[phase]

                    epoch_loss += running_loss
                    epoch_corrects += running_corrects

                    print('\tBatch {:{width}} {} Loss: {:{width}.{precision}f}  Acc: {:{width}.{precision}f}'.format(
                        batch_idx, phase, batch_loss, batch_acc, width=13, precision=6))
                    writer.writerow([phase, batch_loss, batch_acc])

                # deep copy the model
                epoch_loss = epoch_loss / dataset_sizes[phase]
                epoch_acc = epoch_corrects.double() / dataset_sizes[phase]
                print('Epoch loss {0:.4f} and accuracy {0:.4f}'.format(epoch_loss, epoch_acc))
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = deepcopy(model.state_dict())
                    time_now = get_time_now()
                    best_model_name = time_now + model.__class__.__name__ + '_featExtr.pth'
                    torch.save(best_model_wts, best_model_name)
                    print('Model saved as ',best_model_name)
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

# # example usage
# if __name__=='__main__':
#     dataset = WhaleDataset()
#     model = torchvision.models.inception_v3(num_classes=5004,aux_logits=False)
#     for param in model.parameters():
#         param.requires_grad = False
#     ftrs = model.fc.in_features
#     model.fc = torch.nn.Linear(ftrs, 5004)
#     model = model.to(device)
#     optimizer_v3 = torch.optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)
#     model = train_model(model=model,
#                         criterion=torch.nn.CrossEntropyLoss(reduction='mean'),
#                         optimizer=optimizer_v3,
#                         lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer_v3, step_size=5, gamma=0.1))
