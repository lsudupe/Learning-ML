import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# 0) Prepare the data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(y_numpy.astype(np.float32))
y = Y.view(Y.shape[0], 1) #only one column

n_samples, n_features = X.shape
# 1) Model f = wx * b
output_size = 1 # we only want to have one value for each input we add
model = nn.Linear(n_features, output_size)


# 2) Loss and optimize
learning_rate = 0.01
iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),learning_rate)

# 3) Trainning loop

for epoch in range(iters):
    #forward and loss
    y_predicted = model(X)
    l = loss(y_predicted, y)
    #backward
    l.backward()
    #update
    optimizer.step()
    #derivates to 0
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss = {l.item()}')

# Plot
predicted = model(X).detach().numpy()

plt.plot(X_numpy, y_numpy, "ro")
plt.plot(X_numpy, predicted, "b")
plt.show()
