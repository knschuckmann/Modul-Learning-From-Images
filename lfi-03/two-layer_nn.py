"""
Created on Thu Dez 01 09:01:21 2019

Two Layer NN 
is a self created two layer NN. With respect to forward and backward propagation 
the Network gets Trained on batches. 

The derivations with respect to weights and biases are very hard to understand 
so a video is provided :
    * youtube.com/watch?v=GlcnxUlrtek

The output is a trained Network and a plot of the loss in Training and Test rounds

@author: Konstantin Schuckmann
"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import torch

# use GPU if available else CPU 
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Hyperparameters
nn_img_size = 32
num_classes = 3
learning_rate = 0.0001
num_epochs = 500
batch_size = 4

# Which loss function to take
#loss_mode = 'mse' 
loss_mode = 'crossentropy' 

loss_train_hist = []

##################################################
## Please implement a two layer neural network  ##
##################################################

def relu(x):
    """ReLU activation function"""
    return torch.clamp(x, min=0.0)

def relu_derivative(output):
    """derivative of the ReLU activation function"""
    output[output <= 0] = 0
    output[output>0] = 1
    return output

def softmax(z):
    """softmax function to transform values to probabilities"""
    z -= z.max()
    z = torch.exp(z)
    sum_z = z.sum(1, keepdim=True)
    return z / sum_z 

def loss_mse(activation, y_batch):
    """mean squared loss function"""
    # use MSE error as loss function 
    # Hint: the computed error needs to get normalized over the number of samples
    loss = (activation - y_batch).pow(2).sum() 
    mse = 1.0 / activation.shape[0] * loss
    return mse

def loss_crossentropy(activation, y_batch):
    """cross entropy loss function"""
    batch_size = y_batch.shape[0]
    loss = ( - y_batch * activation.log()).sum() / batch_size
    return loss

def loss_deriv_mse(activation, y_batch):
    """derivative of the mean squared loss function"""
    dCda2 = (1 / activation.shape[0]) * (activation - y_batch)
    return dCda2

def loss_deriv_crossentropy(activation, y_batch):
    """derivative of the mean cross entropy loss function"""
    batch_size = y_batch.shape[0]
    dCda2 = activation
    dCda2[range(batch_size), np.argmax(y_batch, axis=1)] -= 1
    dCda2 /= batch_size
    return dCda2

def setup_train():
    """train function"""
    # load and resize train images in three categories
    # cars = 0, flowers = 1, faces = 2 ( true_ids )
    train_images_cars = glob.glob('./images/db/train/cars/*.jpg')
    train_images_flowers = glob.glob('./images/db/train/flowers/*.jpg')
    train_images_faces = glob.glob('./images/db/train/faces/*.jpg')
    train_images = [train_images_cars, train_images_flowers, train_images_faces]
    num_rows = len(train_images_cars)+len(train_images_flowers) +len(train_images_faces) 
    X_train = torch.zeros((num_rows, nn_img_size*nn_img_size))
    y_train = torch.zeros((num_rows, num_classes))
    
    counter = 0
    for (label, fnames) in enumerate(train_images):
        for fname in fnames:
            print(label, fname)
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (nn_img_size, nn_img_size) , interpolation=cv2.INTER_AREA)

            # print( label, " -- ", fname, img.shape)

            # fill matrices X_train - each row is an image vector
            # y_train - one-hot encoded, put only a 1 where the label is correct for the row in X_train
            y_train[counter, label] = 1
            X_train[counter] = torch.from_numpy(img.flatten().astype(np.float32))
            
            counter += 1
    return X_train, y_train

def forward(X_batch, y_batch, W1, W2, b1, b2):
    """forward pass in the neural network """
    a1 = relu(torch.mm(X_batch, W1) + b1)
    a2 = torch.mm(a1, W2) + b2
    
    if loss_mode == 'crossentropy':
        a2 = softmax(a2)
        loss = loss_crossentropy(a2, y_batch)
    else:
        loss = loss_mse(a2,y_batch)
    # loss and both intermediate activations
    return loss, a1, a2

def backward(a2, a1, X_batch, y_batch, W2):
    """backward pass in the neural network """
    # using the chain rule as discussed in the lecture

    # derivatives for W2
    dm2dw2 = a1
    
    if loss_mode == 'crossentropy':
        dcdm2 = loss_deriv_crossentropy(a2, y_batch)
    else:
        dcdm2 = loss_deriv_mse(a2,y_batch)

    
    # derivatives for W1 only the non calculated terms
    dm2da1 = W2
    da1dm1 = relu_derivative(a1)
    dm1dw1 = X_batch
    
    # calculate recurently terms
    basic = torch.mm(dcdm2,dm2da1.T)
    final = (basic * da1dm1) 
    
    # function should return 4 derivatives with respect to
    # W1, W2, b1, b2
    dCdW1 = torch.mm(dm1dw1.T,final)
    dCdW2 = torch.mm(dm2dw2.T,dcdm2)
    dCdb1 = final.sum(axis=0) 
    dCdb2 = dcdm2.sum(axis=0)

    return dCdW1, dCdW2, dCdb1, dCdb2

def train(X_train, y_train):
    """ train procedure """
    # for simplicity of this execise you don't need to find useful hyperparameter
    # I've done this for you already and every test image should work for the
    # given very small trainings database and the following parameters.
    # h is the dimenion of hidden Layer
    h = 1500
    std = 0.001
    cols, rows = X_train.shape
    # initialize W1, W2, b1, b2 randomly
    # Note: W1, W2 should be scaled by variable std
    b1, b2 = torch.randn(1,h), torch.randn(1,num_classes)
    W1 = torch.randn(rows, h) * std
    W2 = torch.randn(h, num_classes) * std    
    # run for num_epochs
    for i in range(num_epochs):
        # use only a batch of batch_size of the training images in each run
        # sample the batch images randomly from the training set
        sample_batch = np.random.choice(cols ,batch_size) # important for no replacement 
        X_batch = X_train[sample_batch]
        y_batch = y_train[sample_batch]
        
        # forward pass for two-layer neural network using ReLU as activation function
        loss, a1 , a2 = forward(X_batch,y_batch, W1,W2,b1,b2)
        # add loss to loss_train_hist for plotting
        loss_train_hist.append(loss)
        
        if i % 10 == 0:
            print("iteration %d: loss %f" % (i, loss))
        # backward pass 
        dCdW1, dCdW2, dCdb1, dCdb2 = backward(a2,a1,X_batch,y_batch,W2)
        
        #print("dCdb2.shape:", dCdb2.shape, dCdb1.shape)

        # depending on the derivatives of W1, and W2 regaring the cost/loss
        # we need to adapt the values in the negative direction of the 
        # gradient decreasing towards the minimum
        # we weight the gradient by a learning rate
        W1 -= dCdW1 * learning_rate
        W2 -= dCdW2 * learning_rate
        b1 -= dCdb1 * learning_rate
        b2 -= dCdb2 * learning_rate
        
    return W1, W2, b1, b2

X_train, y_train = setup_train()
W1, W2, b1, b2 = train(X_train, y_train)

# predict the test images, load all test images and 
# run prediction by computing the forward pass
test_images = []
test_images.append( (cv2.imread('./images/db/test/flower.jpg', cv2.IMREAD_GRAYSCALE), 1) )
test_images.append( (cv2.imread('./images/db/test/car.jpg', cv2.IMREAD_GRAYSCALE), 0) )
test_images.append( (cv2.imread('./images/db/test/face.jpg', cv2.IMREAD_GRAYSCALE), 2) )

for ti in test_images:
    resized_ti = cv2.resize(ti[0], (nn_img_size, nn_img_size) , interpolation=cv2.INTER_AREA)
    x_test = resized_ti.reshape(1,-1)
    x_test = torch.from_numpy(x_test).double()
    if loss_mode == 'crossentropy':
        y_test = np.array([ti[1]])  
        y_test = torch.from_numpy(y_test)
    else:
        y_test = np.array(ti[1])  
        y_test = torch.from_numpy(y_test).double()
    
  
    loss_test, a1_test, a2_test = forward(x_test,y_test,W1.double(), W2.double(), b1.double(), b2.double() )
    # convert test images to pytorch
    # do forward pass depending mse or softmax
    print("Test output (values / pred_id / true_id):", a2_test, np.argmax(a2_test), ti[1])

print("------------------------------------")
print("Test model output Weights:", W1, W2)
print("Test model output bias:", b1, b2)


plt.title("Training Loss vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Training Loss")
plt.plot(range(1,num_epochs +1),loss_train_hist,label="Train")
plt.ylim((0,3.))
plt.xticks(np.arange(1, num_epochs+1, 50.0))
plt.legend()
plt.show()
plt.savefig("simple_nn_train.png")