'''
	Kishore Reddy and Akhil Aji Kumar
	
    Extension : Custom Network
'''

import torch
import torchvision
import sys
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, kernel_size=7)
        self.conv2 = nn.Conv2d(15, 30, kernel_size=7)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(750, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 750)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def train(epoch, network, train_loader, optimizer, train_losses, train_counter, log_interval):
  network.train()

  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()

    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), 'model_custom.pth')
      torch.save(optimizer.state_dict(), 'optimizer_custom.pth')

def test(network, test_loader, test_losses):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  

# Train MNIST Data using custom network``
def main(argv):


    n_epochs = 3
    batch_size_train = 50
    batch_size_test = 160
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 40
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/home/kishore/Documents/mnist', train=True, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize([38,38]),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])),
    batch_size=batch_size_train, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/home/kishore/Documents/mnist', train=False, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize([38,38]),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])),
    batch_size=batch_size_test, shuffle=True)


    data = torchvision.datasets.ImageFolder('/home/kishore/PRCV/Project_5/data', 
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize([28,28]),
                                torchvision.transforms.RandomInvert(p=1),
                                torchvision.transforms.Grayscale(num_output_channels=1),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                                ]))

    train_loader2 = torch.utils.data.DataLoader(data,
    batch_size=batch_size_train, shuffle=True)
   
    examples = enumerate(train_loader)
    batch_idx,(example_data,example_targets) = next(examples)


    examples2 = enumerate(train_loader2)
    batch_idx,(example_data2,example_targets2) = next(examples2)
    example_targets2 = np.reshape(example_targets , (50,1))



    examples = enumerate(train_loader)
    batch_idx,(example_data,example_targets) = next(examples)
    
    example_data[0][0].shape

    fig = plt.figure(1)
    print()
    for i in range(6):
      plt.subplot(2,3,i+1)
      plt.tight_layout()
      plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
      plt.title("Ground Truth: {}".format(example_targets[i]))
      plt.xticks([])
      plt.yticks([])
    fig

    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    test(network, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1):
      train(epoch, network, train_loader, optimizer, train_losses, train_counter, log_interval)
      test(network, test_loader, test_losses)

    fig = plt.figure(2)
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    fig

    with torch.no_grad():
      output = network(example_data)
    
    m = nn.Dropout(p=0.2)
    input = torch.randn(20,16)
    output = m(input)
    print(output)

    fig = plt.figure(3)
    for i in range(6):
      plt.subplot(2,3,i+1)
      plt.tight_layout()
      plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
      plt.title("Prediction: {}".format(
        output.data.max(1, keepdim=True)[1][i].item()))
      plt.xticks([])
      plt.yticks([])
    fig



    sub_network = Net()
    sub_optimizer = optim.SGD(sub_network.parameters(), lr=learning_rate,
                                    momentum=momentum)
    sub_network_state_dict = torch.load('model_custom.pth')
    sub_network.load_state_dict(sub_network_state_dict)

    sub_optimizer_state_dict = torch.load('optimizer_custom.pth')
    sub_optimizer.load_state_dict(sub_optimizer_state_dict)
    sub_network.eval()

    with torch.no_grad():
      sub_output = sub_network(example_data2)

    fig = plt.figure()
    for i in range(6):
      plt.subplot(2,3,i+1)
      plt.tight_layout()
      plt.imshow(example_data2[i][0], cmap='gray', interpolation='none')
      plt.title("Prediction: {}".format(
        sub_output.data.max(1, keepdim=True)[1][i].item()))
      plt.xticks([])
      plt.yticks([])
    fig

 

    kernels = network.conv2.weight.cpu().detach().clone()
    kernels_1 = network.conv1.weight.cpu().detach().numpy()
    print(kernels_1[9])



    kernels = kernels - kernels.min()
    # print(kernels)
    kernels = kernels / kernels.max()
    print(kernels[9])
    # custom_viz(kernels, 'conv1_weights.png', 4)


    return

if __name__ == "__main__":
    main(sys.argv)