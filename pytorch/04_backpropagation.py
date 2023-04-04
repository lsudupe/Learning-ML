import torch

x = torch.tensor(1.0)
print(x)
y = torch.tensor(2.0)

# w is the parameter we want to optimize > requires_grad=true
w = torch.tensor(1.0, requires_grad=True)
print(w)

# forward pass to compute loss
y_predicted = w * x
loss = (y_predicted - y)**2
print(loss)

# backward pass, pytorch will compute the local gradients automatically
loss.backward() #gradient computation
print(w.grad)

# update our weigths
## next forward and backward pass for a couple of times

# optimization
# update weigths, this operation should not be part of the computational graph
with torch.no_grad():
    w -= 0.01 * w.grad
# do not forget to zero the gradients
w.grad.zero_()

print(w)
# next forward and backward pass







