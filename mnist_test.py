"""
Kishore Reddy and Akhil Ajikumar
CS 5330 Computer Vision
Spring 2022

This Python file includes

- Task 1 : Build model to recognize mnist and handwritten

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
from sklearn.neighbors import KNeighborsClassifier

# Network Class
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

# Sub Network class
class Submodel(Net):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override the forward method
    def forward( self, x ):
        x = F.relu( F.max_pool2d( self.conv1(x), 2 ) ) # relu on max pooled results of conv1
        x = F.relu( F.max_pool2d( self.conv2_drop( self.conv2(x)), 2 ) ) # relu on max pooled results of dropout of conv2
        return x

# Test 
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
    example_targets2 = np.reshape(example_targets , (64,1))
    
    example_data[0][0].shape

    fig = plt.figure(1)
  
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


    sub_network = Submodel()
    sub_optimizer = optim.SGD(sub_network.parameters(), lr=learning_rate,
                          momentum=momentum)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    # test(network, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1):

      test(network, test_loader, test_losses)

  

    with torch.no_grad():
      output = network(example_data)

    m = nn.Dropout(p=0.2)
    input = torch.randn(20,16)
    output = m(input)
   

    kernels = network.conv2.weight.cpu().detach().clone()
    kernels_1 = network.conv1.weight.cpu().detach().numpy()
  


    kernels = kernels - kernels.min()
    # print(kernels)
    kernels = kernels / kernels.max()
    
    # custom_viz(kernels, 'conv1_weights.png', 4)


    sub_network = Net()
    sub_optimizer = optim.SGD(sub_network.parameters(), lr=learning_rate,
                                    momentum=momentum)
    sub_network_state_dict = torch.load('model.pth')
    sub_network.load_state_dict(sub_network_state_dict)

    sub_optimizer_state_dict = torch.load('optimizer.pth')
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



    import cv2
    import torchvision.transforms as T

    with torch.no_grad():
      value = example_data.clone()
      value.unsqueeze_(0)
      value = Variable(value,requires_grad = False)

      print(value[0][0].cpu().detach().clone().size())

      src_image = value[0][0].cpu().detach().numpy()
      imgplot = plt.imshow(src_image[0])
      plt.show()
      print(type(src_image[0]))

      for i in range(20):
        resulting_image = cv2.filter2D(src_image[0], -1, kernels_1[i][0])
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(kernels_1[i][0], cmap='gray', interpolation='none')
        axarr[1].imshow(resulting_image, cmap='gray', interpolation='none')
        plt.title("Kernel: {}".format(i))
        plt.xticks([])
        plt.yticks([])

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(sub_output, example_targets2)

    print("score = ",neigh.score(sub_output, example_targets2))

    
    return

if __name__ == "__main__":
    main(sys.argv)