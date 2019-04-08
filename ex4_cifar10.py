import matplotlib.pyplot as plt 
import numpy as np 
import time

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def imshow(img):
	img = img/2.0 + 0.5
	npimg = img.numpy()  # convert to np.array
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	


class CNN(nn.Module):

	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)		# 3 input channels (RGB image), 6 output channels, 5x5 kernel 
		self.pool = nn.MaxPool2d(2, 2)		# 2x2 max pooling windown
		self.conv2 = nn.Conv2d(6, 16, 5)	# 6 input channels, 16 output channels, 5x5 kernel
		# Flatten the output
		self.fc1 = nn.Linear(16*5*5, 120) 	# 400 input neurons, 120 output neurons
		self.fc2 = nn.Linear(120, 84)		# 120 input neurons, 84 output neurons
		self.fc3 = nn.Linear(84, 10)		# 84 input neuron, 10 output neuros (10 classes)

	def forward(self, x):
		# First conv + relu + maxpool layer
		x = self.conv1(x)
		x = F.relu(x)
		x = self.pool(x)
		# Second conv + relu + maxpool layer
		x = self.conv2(x)
		x = F.relu(x)
		x = self.pool(x)
		# Flatten the output
		x = x.view(-1, 16*5*5)  # view() = np.reshape()
		# First dense + relu layer
		x = self.fc1(x)
		x = F.relu(x)
		# Second dense + relu layer
		x = self.fc2(x)
		x = F.relu(x)
		# Output layer
		x = self.fc3(x)

		return x



# ****************************************** #
#          LOAD TRAINING DATA                #
# ****************************************** #

# define a series of operations
# this will be applied over images
preprocessing = transforms.Compose([
								# First convert to a pytorch Tensor
								transforms.ToTensor(),
								# Then normalize in range -1, 1
								transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# download and import train, testdata
trainset = torchvision.datasets.CIFAR10(root = './data', train=True, download=True, transform = preprocessing)
testset = torchvision.datasets.CIFAR10(root = './data', train=False, download=True, transform = preprocessing)

# define a loader, or feeder, to the network
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle = False, num_workers=2)

# Define class names
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' )


# ****************************************** #
#     DISPLAY SOME TRAINING DATA             #
# ****************************************** #

train_data_iter = iter(trainloader)
train_images, train_labels = train_data_iter.next()

#imshow(torchvision.utils.make_grid(train_images))
# print labels
print(' '.join('%5s' % classes[train_labels[j]] for j in range(4)))

# ****************************************** #
#     DEFINE CONVOLUTIONAL NN                #
# ****************************************** #

net = CNN()

# first move to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print("CUDA device: {}".format(device))
net.to(device)


print("\n# *************************************** #\n"+
	  "#   NETWORK LAYER STRUCTURE               #\n"+
	  "# *************************************** #")
print(net)


# ****************************************** #
#    DEFINE LOSS FUNCTION and OPTIMIZER      #
# ****************************************** #

# There are >2 classes, so use cross entropy instead of binary cross entropy
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

# ****************************************** #
#           TRAIN NETWORK                    #
# ****************************************** #


print("\n# *************************************** #\n"+
	  "#    TRAINING STARTED...                  #\n"+
	  "# *************************************** #")

t1 = time.time()
epochs = 2
for epoch in range(epochs):

	running_loss = 0.0 # Total loss per minibatch

	for i, data in enumerate(trainloader, 0):

		# ******************************* #
		#   Load data in minibatches and  #
		#   move to GPU                   #
		# ******************************* #
		minibatch, labels = data
		minibatch, labels = minibatch.to(device), labels.to(device)


		# make zero the parameter gradients
		optimizer.zero_grad()

		# forward + backpropagate + optimizer
		outputs = net(minibatch)
		loss = loss_function(outputs, labels)
		loss.backward()
		optimizer.step()

		# print stats
		running_loss += loss.item()  # get single item from loss and add
		if i%2000 == 1999:
			print("Epoch {}  Minibatch {}  Running Loss {}".format(epoch, i, running_loss/2000))
			running_loss = 0.0
t2 = time.time()
train_time = t2 - t1


print("***    TRAINING FINISHED    ****")
print(" Training time:  {:.4f}".format(train_time))

# ****************************************** #
#       PREDICT SOME IMAGES                  #
# ****************************************** #

test_data_iter = iter(testloader)
test_images, test_labels = test_data_iter.next()
# print images
imshow(torchvision.utils.make_grid(test_images))
print('\nGroundTruth: ', ' '.join('%5s' % classes[test_labels[j]] for j in range(4)))


# REMEMBER! If network is in GPU, move data to GPU!
test_images, test_labels = test_images.to(device), test_labels.to(device)
output = net(test_images)
_, predictions = torch.max(output, 1)
print('\nPredicted: ', ' '.join('%5s' % classes[predictions[j]] for j in range(4)))

plt.show()







