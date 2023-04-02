import torch


# specify that you want to calculate the gradient
x = torch.randn(3, requires_grad=True)
print(x)
# now if we perform any operation, python will create a so call computational graph
y = x+2
print(y)
# and with the backward() function it will calculate the gradients
z = y*y*2
print(z)
z = z.mean()
print(z)
#calculate the gradiant
z.backward() #gradient of z respect to x dz/dx
print(x.grad) #we have the gradiants in this tensor
#our z is a scalar, so we do not have to specify anything in backward(). It multiply the partial
#derivative matrix with the vector to obtain the vector-Jacobian product.
# SO NOW LETS DO THE SAME WITH NO SCALAR

x = torch.randn(3, requires_grad=True)
print(x)
y = x+2
print(y)
z = y*y*2
print(z)
#z = z.mean()

z.backward()
#given the gradient argument, we need to create a vector from the same size
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float32)
# and now pass this vector to our backward function
z.backward(v)
print(x.grad)

##to stop pytorch for tracking the history. Maybe we want to recalculate the weigths but not the gradients
# x.requires_grad_(False)
# x.detach() this will create a new tensor that doesn't require the gradient
# with torch.no_grad()
x = torch.randn(3, requires_grad=True)
print(x)

#first
x.requires_grad_(False) #everything that has _ will modify our variable
print(x)
#second
y = x.detach() #will create a new vector with same values but with out the graph
print(y)
#third
with torch.no_grad():
    y = x + 2
    print(y)

##BIG PROBLEM. backward() accumulates the gradients. Take this into account during optimization

weigths = torch.ones(4, requires_grad=True)
print(weigths)

for epoch in range(3):
    model_output = (weigths*3).sum()

    model_output.backward()

    print(weigths.grad)

    ##empty the gradient
    weigths.grad.zero_()


