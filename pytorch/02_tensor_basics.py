import torch
import numpy as np


# create a empty tensor
x = torch.empty(1) #scalar
print(x)
x = torch.empty(3) #vector
print(x)
x = torch.empty(2,3) #matrix
print(x)
x torch.empty(2,2,3) #tensor 3 dimensions
print(x)

# create a only ones object
x = torch.ones(2,2)
print(x)
# chech datatype
print(x.dtype)
# select when creation datatype
x = torch.ones(2,3, dtype=torch.float16)
print(x)

# create a tensor with a list
x = torch.tensor([2.2,3.4])
print(x)

# create tensors with random variables and perform operations
x = torch.rand(2,2)
y = torch.rand(2,2)
print(x)
print(y)
# addition
z = x + y
z = torch.add(x,y) # or
y.add_(x) # or, modify y
print(y)

# print only row number one but all columns
print(x[1, :])

# reshape the tensor
x = torch.rand(4,4)
print(x)
y = x.view(16) # the number of elements must be the same
print(y)
y = x.view(-1,8)
print(y.size())

##transform from numpy to tensor
a = torch.ones(5)
print(type(a))
b = a.numpy()
print(type(b))

## check where your variables are stored, GPU or CPU, you can change
# both by mistake
## the _ function will modify our variable in the place
a.add_(1)
print(a)
print(b)

## other way around
c = np.ones(5)
print(c)
d= torch.from_numpy(c) # you can specify dtype
print(d)

c += 1
print(c)
print(d) #OUR TENSOR HAD BEEN MODIFY TO!! YOUR TENSOR IS IN THE CPU


#set the device: create a device variable to automatically use
#GPU if avail
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#movel your model and/or tensor to the device
# For a model
model = YourModel().to(device)

# For a tensor
tensor = torch.Tensor([1, 2, 3]).to(device)

#check the device of a tensor or model. Verify the device on which a tensor
#or model is allocated
print("Model device:", model.device)
print("Tensor device:", tensor.device)
