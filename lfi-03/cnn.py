"""
Created on Thu Dez 01 12:38:39 2019

This file trains three difefrent types of Models:
    * My Own cnn 
    * LeNet
    * AlexNet 

Befor Trainig it preprocesses the Data. It is recomended to define the modelName befor 
running this file, so one can decide which model one wants to train. 
The Output is a prediction along with the plotted Accuracy and the Crossentropy Loss

Information: 
    * https://pramodmurthy.com/blog/2019/03/25/tutorial_001_mlp_mnist_pytorch.html
    * https://discuss.pytorch.org/t/inferring-shape-via-flatten-operator/138/4
    * https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    * source code inspireed by
        * https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#model-training-and-validation-code
    
@author: Konstantin Schuckmann
"""
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Define the Modelname so the Program knows what model to take 
#modelName = "MyNeuralNetwork"
modelName = "AlexNet"
#modelName = "LeNet"

root = './data'

# AlexNet takes images that are at least 224 x 224 pixel 
if modelName == "AlexNet":
    transform = transforms.Compose([    
        transforms.Resize((224,224), interpolation = 2),
        transforms.ToTensor()# Biliear interpolation
    ])
else:
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()    
    ])

# Load Data from torchvision.datasets    
train_set = datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)
test_set = datasets.FashionMNIST(root=root, train=False, transform=transform, download=True)

# overall hyperparameter to be changed for different models at the bottom of this file 
batch_size = 30
num_epochs = 15
learning_rate = 0.001
momentum = 0.9 # to avoid local minima

# Number of Output classes
num_classes = 10

# Load train and test data
data_loaders = {}
data_loaders['train'] = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
data_loaders['test'] = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

# Own NN with following Architecture
# Conv 3 -> ReLU -> Conv 3 -> ReLU -> Maxpool 2 -> Dropout -> Conv 3 -> ReLU ->
# Conv 3 -> ReLU -> Maxpool 2 -> Dropout -> flatten -> Linear 500 -> Relu -> Linear 10
class MyNeuralNetwork(nn.Module):
    """Inherit from nn.Module, calculates own architecture of NN. Only forwarding, 
    without any backpropagation. 
    
    Return: forwarded result
    """
    def __init__(self):
        #super(MyNeuralNetwork, self).__init__() # older way of calling super
        super().__init__() # inherit constructor of Module
        self.features = nn.Sequential(
                # Input of MINST FAshion dataset 28 x 28 x 1 because of grayscale 
                nn.Conv2d(in_channels = 1,out_channels = 32, kernel_size = 3),
                nn.ReLU(),
                # output = input  26 x 26 x 32
                nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3),
                nn.ReLU(),
                # output = input  24 x 24 x 64                
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                # output = input  12 x 12 x 64                                
                nn.Dropout2d(p = 0.25),
                # output = input  12 x 12 x 64                            
                nn.Conv2d(in_channels = 64,out_channels = 128, kernel_size = 3),
                nn.ReLU(),
                # output = input  10 x 10 x 128                
                nn.Conv2d(in_channels = 128, out_channels = 192, kernel_size = 3),
                nn.ReLU(),
                # output = input  8 x 8 x 192              
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                # output = input  4 x 4 x 192     
                nn.Dropout2d(p = 0.25),
                )
        self.classifier = nn.Sequential(
                nn.Linear(in_features = 4*4*192, out_features = 500),
                nn.ReLU(),
                nn.Linear(in_features = 500, out_features = num_classes),
                # no need of using softmax because Crossentropyloss takes care of that 
                )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)

    def name(self):
        return "MyNeuralNetwork"
            
    
class LeNet(MyNeuralNetwork):
    """Inherit of Own NN. Architecture is the LeNet 1998 architecture
    
    Output: forwarded result.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(in_channels = 1,out_channels = 28, kernel_size = 3, padding = 0),
                # BathcNorm to normalize the output and fasten the calculation 
                nn.BatchNorm2d(28),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size = 2, stride = 2),
                nn.Conv2d(in_channels = 28,out_channels = 60, kernel_size = 3, padding = 0),
                nn.BatchNorm2d(60),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size = 2, stride = 2),
                )
        self.classifier = nn.Sequential(
                nn.Linear(in_features = 5*5*60, out_features = 120),
                nn.Tanh(),
                nn.Linear(in_features = 120, out_features = 84),
                nn.Tanh(),
                nn.Linear(in_features = 84, out_features = num_classes),
                )
    def name(self):
        return "LeNet"

class AlexNet(MyNeuralNetwork):
    """Inherit of Own NN. Architecture is the AlexNet 2011 architecture. Takes special image size to 
    forward all calculations through the convolutional layers. 
    
    Output: forwarded result.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(in_channels = 1,out_channels = 64, kernel_size = 11, stride = 4, padding = 2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size = 3, stride = 2),
                nn.Conv2d(in_channels = 64,out_channels = 192, kernel_size = 5, padding = 2),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size = 3, stride = 2),
                nn.Conv2d(in_channels = 192,out_channels = 384, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels = 384,out_channels = 256, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels = 256,out_channels = 256 , kernel_size = 3, padding = 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),                    
                nn.MaxPool2d(kernel_size = 3, stride = 2),
                )
        self.classifier = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
                )
    
    def name(self):
        return "AlexNet"


## training
# Decide which model to take and initialize hyperparameters for different models        
if modelName == "MyNeuralNetwork":
    model = MyNeuralNetwork().to(device)
elif modelName == "AlexNet": # because of teh picture size the Algorithm is more time intensive
    batch_size = 200
    num_epochs = 3
    learning_rate = 0.001
    model = AlexNet().to(device)
else:
    model = LeNet().to(device)
    
# Create Optimizer for numerical minima search     
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

criterion = nn.CrossEntropyLoss()

train_acc_history = []
test_acc_history = []

train_loss_history = []
test_loss_history = []


best_acc = 0.0
# Starting time 
since = time.time()
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'test']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        for batch_idx, (inputs, labels) in enumerate(data_loaders[phase]):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if batch_idx % 10 == 0:
                print('{} Batch: {} of {}'.format(phase, batch_idx, len(data_loaders[phase])))

        epoch_loss = running_loss / len(data_loaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase == 'test' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        if phase == 'test':
            test_acc_history.append(epoch_acc)
            test_loss_history.append(epoch_loss)
        if phase == 'train':
            train_acc_history.append(epoch_acc)
            train_loss_history.append(epoch_loss)

    print()
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

acc_train_hist = []
acc_test_hist = []

acc_train_hist = [h.cpu().numpy() for h in train_acc_history]
acc_test_hist = [h.cpu().numpy() for h in test_acc_history]

plt.title("Validation/Test Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation/Test Accuracy")
plt.plot(range(1,num_epochs+1),acc_train_hist,label="Train")
plt.plot(range(1,num_epochs+1),acc_test_hist,label="Test")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()

plt.title("Validation/Test Loss vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation/Test Loss")
plt.plot(range(1,num_epochs+1),train_loss_history,label="Train")
plt.plot(range(1,num_epochs+1),test_loss_history,label="Test")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()


examples = enumerate(data_loaders['test'])
batch_idx, (example_data, example_targets) = next(examples)
with torch.no_grad():
  output = model(example_data)

categories = {
    0:	'T-shirt/top',
    1:	'Trouser',
    2:	'Pullover',
    3:	'Dress',
    4:	'Coat',
    5:	'Sandal',
    6:	'Shirt',
    7:	'Sneaker',
    8:	'Bag',
    9:	'Ankle boot'
}

for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Pred: {}".format(
      categories[output.data.max(1, keepdim=True)[1][i].item()]))
  plt.xticks([])
  plt.yticks([])
plt.show()