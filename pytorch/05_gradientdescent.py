import numpy as np
import torch

####FIRST MANUALLY
import numpy

# f = w * x linear regression

# f = 2 * x our function
#some training examples
# X = np.array([1,2,3,4], dtype=np.float32)
# Y = np.array([2,4,6,8], dtype=np.float32) # x values multiple by 2, bacause is our formula
# w = 0.0   # we inizialize our weigths

X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#1. model predictions
def forward(x):
    return w*x

#2. calculate the loss
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

#3. calculate the gradient
#MSE = 1/N(w*x - y)**2
#dJ/dW = 1/N * 2(w*x - y)
# def gradient(x,y, y_predicted):
#     return np.dot(2*x, y_predicted - y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Let;s start the training
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    #predictions>forward
    y_predictions = forward(X)
    #calculate loss
    l = loss(Y, y_predictions)
    #gradients
    #dW = gradient(X, Y, y_predictions)

    l.backward() #dl/dW

    #update weigths
    #we go in the negative direction of the gradient -=
    #w -= learning_rate *dW

    #we do not want the update to be part of our computational graph
    with torch.no_grad():
        w -= learning_rate * w.grad
    # zero gradients, w.grad() atrribute is acumulating our gradients
    # so before the next iteration you want them to become 0
    w.grad.zero_()

    if epoch % 1 == 0:
        print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')




