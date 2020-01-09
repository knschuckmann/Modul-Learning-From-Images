# -*- coding: utf-8 -*-
"""
Created on 08.01.20 
Learning from Images fourth Assignment 
implementing an autoencoder
@author: Konstantin Schuckmann
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from torch.autograd import Variable

# define global variable for number of dimensions
# 10/ 75 / 150
K = 10


def load_images(path: str, file_ending: str=".png") -> (list, int, int):
    """
    Load all images in path with matplotlib that have given file_ending

    Arguments:
    path: path of directory containing image files that can be assumed to have all the same dimensions
    file_ending: string that image files have to end with, if not->ignore file

    Return:
    images: list of images (each image as numpy.ndarray and dtype=float64)
    dimension_x: size of images in x direction
    dimension_y: size of images in y direction
    """

    images = []

    # TODO read each image in path as numpy.ndarray and append to images
    # Useful functions: lib.list_directory(), matplotlib.image.imread(), numpy.asarray()

    files = os.listdir(path)
    files.sort()
    for cur in files:
        if not cur.endswith(file_ending):
            continue

        try:
            image = mpl.image.imread(path + cur)
            img_mtx = np.asarray(image, dtype="float64")
            images.append(img_mtx)
        except:
            continue

    dimension_y = images[0].shape[0]
    dimension_x = images[0].shape[1]

    return images, dimension_x, dimension_y


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # TODO YOUR CODE HERE
        self.encoder = nn.Sequential(
            nn.Linear(116 * 98, K),
            #nn.ReLU(True),
            #nn.Linear(K*3, K*2),
            #nn.Tanh(),
            nn.Linear(K, K)
            )
        self.decoder = nn.Sequential(
            #nn.Linear(K,K*2),
            #nn.Tanh(),
            #nn.Linear(K*2,K*3),
            #nn.ReLU(True),
            nn.Linear(K, 116 * 98),
            #nn.Tanh()
            )
        

    def forward(self, x):
        # TODO YOUR CODE HERE
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':

    images, x, y = load_images('./data/train/')

    # setup data matrix
    D = np.zeros((len(images), images[0].size), dtype=np.float32)
    for i in range(len(images)):
        D[i, :] = images[i].flatten()

    # 1. calculate and subtract mean to center the data in D
        
    mean_data = D.mean(axis = 0)
    for i in range(len(D)):
        D[i, :] -= mean_data
        # D[i, :] -= mean_data[i]
            
    num_epochs = 1000
    batch_size = 50
    learning_rate = 0.01

    data = torch.from_numpy(D)

    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-05)

    for epoch in range(num_epochs):
        data = Variable(data)
        # ===================forward=====================
        output = model(data)
        loss = criterion(output, data)
        MSE_loss = nn.MSELoss()(output, data)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================log========================
        if (epoch %100) == 0:
            print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
                  .format(epoch + 1, num_epochs, loss.data, MSE_loss.data))

    # now we use the nn model to reconstruct test images
    # and measure their reconstruction error

    images_test, x, y = load_images('./data/test/')
    D_test = np.zeros((len(images_test), images_test[0].size), dtype=np.float32)
    for i in range(len(images_test)):
        D_test[i, :] = images_test[i].flatten()

    # mean_data = D_test.mean(axis = 1)
    for i in range(D_test.shape[0]):
        D_test[i, :] -= mean_data
        # D_test[i, :] -= mean_data[i]
        

    data_test = torch.from_numpy(D_test)


    errors = []
    for i, test_image in enumerate(images_test):

        # evaluate the model using data_test samples i
        img_reconst = model(data_test[i])
        img_reconst = img_reconst.data.numpy()
        # img_reconst += mean_data[i]
        img_reconst += mean_data
        img_reconst.resize(images_test[0].shape)
        # add the mean to the predicted/reconstructed image
        # and reshape to size (116,98)
        # TODO YOUR CODE HERE
        #pass
        # uncomment
        error = np.linalg.norm(images_test[i] - img_reconst)
        errors.append(error)
        print("reconstruction error: ", error)

    grid = plt.GridSpec(2, 9)

    plt.subplot(grid[0, 0:3])
    plt.imshow(images_test[14], cmap='Greys_r')
    plt.xlabel('Original person')

    pred = model(data_test[14])
    pred_np = pred.data.numpy()
    pred_np += mean_data
    img_reconst = pred_np.reshape(images_test[0].shape)
    plt.subplot(grid[0, 3:6])
    # img_reconst = plt.imshow(pred_np, cmap='Greys_r')
    plt.imshow(img_reconst, cmap='Greys_r')
    plt.xlabel('Reconstructed image')

    plt.subplot(grid[0, 6:])
    plt.plot(np.arange(len(images_test)), errors)
    plt.xlabel('Errors all images')

    plt.savefig("./results_ae/pca_ae_solution_" + str(K) +".png")
    plt.show()

    print("Mean error", np.asarray(errors).mean())
