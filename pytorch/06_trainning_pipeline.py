
# The training pipeline has different steps
# 1.Design the model(n of input, n of outputs,forward pass with all the different layers
# 2.Contruct the loss and optimizer
# 3.Trainning loop
#       - forward = compute predictions and loos
#       - backward = compute gradients
#       - update weight
#       - iterate until it is done

import torch
import torch.nn as nn

# Linear model
# f = w * x
# f = 2 * x

# 0) Training samples
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

# 1) Design the model, initialize weigth and define forward
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return w * x

# 2) Contruct loss and optimizer
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()

optimizer = torch.optim.SGD([w], lr=learning_rate)

# 3) Loop
for epoch in range(n_iters):

    # predictions
    y_pred = forward(X)

    # loss
    l = loss(Y , y_pred)

    # calculate gradients = backward pass
    l.backward()

    # update weigths
    optimizer.step()

    # put in zero the gradient after updating
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print("epoch ", epoch+1, "w: ", w, "loss: ", l)







