import torch
import torch.nn as nn
import torch.nn.functional as F 

# create a NN
class Net(nn.Module):

    def __init__(self):

        # initilize parent
        super(Net, self).__init__()

        # 1 input image channel, 6 output channels, 5x5 conv kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 6 input channels, 16 output channels, 5x5 conv kernel
        self.conv2 = nn.Conv2d(6, 16, 5)

        # affine operation: y = w*x + b
        # 16 output channels each 5x5, 120 output channels
        self.fc1 = nn.Linear(16*5*5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Apply convolution layer 1 + Max pooling with 2x2 window
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2,2))
        # Apply Convolution layer 2 + Max pooling with 2x2 window
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2,2))
        # Flatten features
        x = x.view(-1, self.num_flat_features(x))
        # Apply dense layer 1
        x = F.relu(self.fc1(x))
        # Apply dense layer 2
        x = F.relu(self.fc2(x))
        # Apply dense layer 3
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
# Create a network... and print it
print("\n\n# *********************************** #\n"+
      "#         NETWORK SHAPE               #\n"+
      "# *********************************** #\n")
net = Net()
print(net)

# See params
print("\n\n# *********************************** #\n"+
      "#         PARAMETERS                  #\n"+
      "# *********************************** #\n")
params = list(net.parameters())
print("Learnable Parameters: {}".format(len(params)))
print("Parameters per Layer: {}".format([layer.size() for layer in params]))

# Test a random  32x32 input
print("\n\n# *********************************** #\n"+
      "#         RANDOM INPUT TEST           #\n"+
      "# *********************************** #\n")
inp = torch.randn(1, 1, 32, 32)
out = net(inp)
print(out)

# Zero the gradients and fill brackpropagation with random gradients
net.zero_grad()
out.backward(torch.randn(1,10))

# Define a loss function
print("\n\n# *********************************** #\n"+
      "#             LOSS FUNCTION               #\n"+
      "# *********************************** #\n")
output = net(inp)
print("Output prediction values \n{}".format(output))
target = torch.randn(10)  # a dummy target, it is a column vector, we need row bector
target = target.view(1,-1)  # we need a row vector, to match the output shape
print(" Dummy target ground truth \n{}".format(target))

loss_function = nn.MSELoss()     # Loss is Mean Squared Root 
loss = loss_function(output, target) 
print("Loss is {}".format(loss))


print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# Backpropagate the gradient
print("\n\n# *********************************** #\n"+
      "#         BACK PROPAGATION           #\n"+
      "# *********************************** #\n")

net.zero_grad()  # zero the gradient buffers for all parameters, if not they will accumulate
print("conv1.bias.grad before backward \n {}".format(net.conv1.bias.grad))
loss.backward()  # do backward propagation
print("conv1.bias.grad after backward \n {}".format(net.conv1.bias.grad))

# Update weights
print("\n\n# *********************************** #\n"+
      "#         UPDATE WEIGHTS              #\n"+
      "# *********************************** #\n")


# There are 2 ways. One is this:

learn_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learn_rate)

# The second one
import torch.optim as optim

# create optimizer 
optimizer = optim.SGD(net.parameters(), lr = 0.01)

# this should go in the training loop
optimizer.zero_grad()      # zero the gradient buffers
output = net(inp)        # forward pass
loss = loss_function(output, target)         # calculate loss
loss.backward()
optimizer.step()           # Does the actual update of the weights

