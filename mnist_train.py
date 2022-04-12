"""
Kishore Reddy and Akhil Ajikumar
CS 5330 Computer Vision
Spring 2022

This Python file includes

- Task 1 : Build and train a network to recognize digits

"""


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
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
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
      torch.save(network.state_dict(), 'model.pth')
      torch.save(optimizer.state_dict(), 'optimizer.pth')

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
  

def custom_viz(kernels, path=None, cols=None, size=None, verbose=False):
   
    def set_size(w,h, ax=None):
        """ w, h: width, height in inches """
        if not ax: ax=plt.gca()
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w)/(r-l)
        figh = float(h)/(t-b)
        ax.figure.set_size_inches(figw, figh)
    
    N = kernels.shape[0]
    C = kernels.shape[1]
    
    if verbose:
        print("Shape of input: ", kernels.shape)
    # If single channel kernel with HxW size,
    # plot them in a row.
    # Else, plot image with C number of columns.
    if cols==None:
        req_cols = C
    elif cols:
        req_cols = cols
    elif C>1:
        req_cols = C
    
    total_cols = N*C
    req_cols = cols
    num_rows = int(np.ceil(total_cols/req_cols))
    pos = range(1,total_cols + 1)

    fig = plt.figure(1)
    fig.tight_layout()
    k=0
    for i in range(kernels.shape[0]):
        for j in range(kernels.shape[1]):
            img = kernels[i][j]
            ax = fig.add_subplot(num_rows,req_cols,pos[k])
            ax.imshow(img, cmap='gray')
            plt.axis('off')
            k = k+1
    if size:
        size_h,size_w = size
        set_size(size_h,size_w,ax)
    if path:
        plt.savefig(path, dpi=100)
    plt.show()

# Train MNIST Data
def main(argv):

    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 40
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/home/kishore/Documents/mnist', train=True, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])),
    batch_size=batch_size_train, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/home/kishore/Documents/mnist', train=False, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])),
    batch_size=batch_size_test, shuffle=True)

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

    with torch.no_grad():
      output = network(example_data)
    
    m = nn.Dropout(p=0.2)
    input = torch.randn(20,16)
    output = m(input)
    print(output)

    fig = plt.figure()
    for i in range(6):
      plt.subplot(2,3,i+1)
      plt.tight_layout()
      plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
      plt.title("Prediction: {}".format(
        output.data.max(1, keepdim=True)[1][i].item()))
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