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
