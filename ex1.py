# Pytorch examples following 
# https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html

from __future__ import print_function
import torch

# Uninitialized matrix, can contain any binary number, even NaN
print("Uninitialized matrix")
x = torch.empty(5 ,3)
print(x)

print("\n Random matrix")
x = torch.rand(5, 3)
print(x)

print("\n Long type zero matrix")
x = torch.zeros(5, 3, dtype = torch.long)
print(x)

print("\n This is a tensor")
x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype = torch.double)
print(x)

x = torch.randn_like(x, dtype = torch.float)
print(x)

print("\n Size of tensor")
print(x.size())

y = torch.randn_like(x, dtype = torch.float)
print(" Tensor x + y")
print(x+y)
print(" or using torch.add()")
print(torch.add(x,y ))

result = torch.empty_like(x)
print(" Empty result tensor")
print(result)
torch.add(x, y, out = result)
print(" result tensor after addition storage")
print(result)

print("Indexing")
print(x[:3,:3])

x = torch.rand(4,4)
print("Resizing")
print(x.view(x.size()[0]*x.size()[1]))
print(x.view(-1,2))

print(" \n Moving Tensors to GPU")
if torch.cuda.is_available():
	device = torch.device("cuda")
	y = torch.ones_like(x, device=device) # directly create on GPU
	
	print("\n Z = x + y , Y is in GPU and X in CPU... this will be an ERROR")
	#z = x + y
	print("\n like this:  RuntimeError: expected type torch.FloatTensor but got torch.cuda.FloatTensor")
	#z = y + x
	print("\n or like this: RuntimeError: expected type torch.cuda.FloatTensor but got torch.FloatTensor")
	#print(z)
	x = x.to(device)  					# move x from CPU to GPU
	print("\n...moved X to GPU: {}".format(x))
	print("\n Z = x + y , both X and Y in GPU")
	z = x + y
	print(z)
