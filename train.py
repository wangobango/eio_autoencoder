import cv2
import numpy as np 
import torchvision.datasets as dataset_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import pickle
from model import ConvAutoencoder, ConvNet


cifar_dataset = dataset_loader.CIFAR10('data/', download=True)
cifar_dataset = cifar_dataset.data

# Creating input and output data
batch_size = 100
op_data = np.array([cifar_dataset[_ix:_ix+batch_size] for _ix in range(0, cifar_dataset.shape[0], batch_size)])
ip_data = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in cifar_dataset])
ip_data = np.array([ip_data[_ix:_ix+batch_size] for _ix in range(0, ip_data.shape[0], batch_size)])

# Splitting the data
train_ratio, test_ratio = 0.8, 0.2
rand_ix = np.random.permutation(ip_data.shape[0])
train_ix, test_ix = rand_ix[:int(train_ratio*rand_ix.shape[0])], rand_ix[int(test_ratio*rand_ix.shape[0]):]
x_train, y_train, x_test, y_test = ip_data[train_ix,:,:], op_data[train_ix,:,:], ip_data[test_ix,:,:], op_data[test_ix,:,:]


def transform_and_create_torch_tensors(data):
    # reshaping the input and accelerate the tensor computations using the gpu.
    data = torch.from_numpy(data).float().cpu()
    return data.contiguous().view(data.size(0),batch_size,-1,32,32)

# transforming all the data.
x_train, y_train, x_test, y_test = map(transform_and_create_torch_tensors, [x_train, y_train, x_test, y_test])

mean = torch.mean(x_train[:,:,0,:,:])
std = torch.std(x_train[:,:,0,:,:])
print('mean: {}, std: {}'.format(mean, std))
x_train[:,:,0,:,:] = (x_train[:,:,0,:,:]-mean)/std
x_test[:,:,0,:,:] = (x_test[:,:,0,:,:]-mean)/std # using the mean and std from the training dataset
y_train = y_train/255
y_test = y_test/255

learning_rate = 0.0001
epochs = 1
criterion = torch.nn.MSELoss()

net = ConvNet(batch_size)
net.cpu()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

train_loss_container, test_loss_container = [], []

for e in range(epochs):  # loop over the dataset multiple times

    train_loss = 0.0
    test_loss = 0.0
    
    print("Epoch: {}".format(e))

    for batch, train_data in enumerate(x_train):
        
        # get the inputs
        ip, op = train_data, y_train[batch]
        # zero the parameter gradients
        optimizer.zero_grad()

        model_op = net(ip)
        loss = criterion(model_op, op)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        for batch_test, test_data in enumerate(x_test, 0):
            # get the inputs
            ip_test, op_test = test_data, y_test[batch_test]
            # forward + backward + optimize
            model_op = net(ip_test)
            loss_test = criterion(model_op, op_test)
            test_loss += loss_test.item()
            print('\rEPOCH: {} | Train_loss: {} | Test_loss: {}'.format(e, train_loss, test_loss), end='')
    
    train_loss_container.append(train_loss)
    test_loss_container.append(test_loss)
    print('\rEPOCH: {} | Train_loss: {} | Test_loss: {}'.format(e, train_loss, test_loss), end='')
print('\nFinished Training')

with open('model.pickl', 'wb') as f:
    pickle.dump(net, f)