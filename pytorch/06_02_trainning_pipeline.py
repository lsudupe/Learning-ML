

import torch
import torch.nn as nn

# 0) Design the model. Trainning samples
# we need a 2D array now, n of rows is number of samples
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

n_samples, n_features = X.shape
print(f'n_samples: {n_samples}, n_features: {n_features}')

# create a test sample
X_test = torch.tensor([5], dtype=torch.float32)

# 1) Design the model. input size and output size
input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

# 2) Define the loss and the optimizer
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

# 3) Training loop

for epoch in range(n_iters):
    # predictions with forward pass
    y_pred = model(X)
    # loss function
    l = loss(Y, y_pred)
    # calculate gradients, backward
    l.backward()
    # update weigths
    optimizer.step()
    # put the gradient in zero
    optimizer.zero_grad()

    if epoch % 10 == 0:
    [w, b] = model.parameters() #extract parameters
    print(f'epoch', epoch+1, ':w = ', w[0][0].item(), 'loss= ', l)

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')









