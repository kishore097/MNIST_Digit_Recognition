"""

Kishore Reddy and Akhil Ajikumar
CS 5330 Computer Vision
Spring 2021

This Python file includes

- Task 3 and Extension - Greek letters and KNN
"""

from torchvision.transforms.transforms import Grayscale
import PIL.ImageOps
from torch.utils.data import DataLoader, Dataset
import numpy as np
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


class Submodel_greek(Net):
  
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override the forward method
    def forward( self, x ):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return x

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
  

# Test greek letters handwritten, extension and calculate KNN
def main(argv):

    n_epochs = 3
    batch_size_train = 1000
    batch_size_test = 9
    learning_rate = 0.01
    momentum = 0.5 
    log_interval = 10
    random_seed = 1 
    torch.backends.cudnn.enabled = True
    torch.manual_seed(random_seed)
 
    data = torchvision.datasets.ImageFolder('/home/kishore/PRCV/Project_5/greek_hand', 
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
    print(data.class_to_idx)
    # print(data.shape)


    examples2 = enumerate(train_loader2)
    batch_idx,(example_data2,example_targets) = next(examples2)
    print(example_targets.shape)
    example_targets = np.reshape(example_targets , (4,1))
    print(example_targets)


    greek_network = Submodel_greek()
    greek_optimizer = optim.SGD(greek_network.parameters(), lr=learning_rate,
                                    momentum=momentum)
    greek_network_state_dict = torch.load('model.pth')
    greek_network.load_state_dict(greek_network_state_dict)

    greek_optimizer_state_dict = torch.load('optimizer.pth')
    greek_optimizer.load_state_dict(greek_optimizer_state_dict)
    greek_network.eval()

    with torch.no_grad():
      greek_output = greek_network(example_data2)

    fig = plt.figure()
    for i in range(4):
      plt.subplot(2,3,i+1)
      plt.tight_layout()
      plt.imshow(example_data2[i][0], cmap='gray', interpolation='none')
      plt.title("Prediction: {}".format(
        greek_output.data.max(1, keepdim=True)[1][i].item()))
      plt.xticks([])
      plt.yticks([])
    fig


    plt.imshow(greek_output)
   
    print(greek_output.shape)

    # print(np.sum(greek_output[26][:]))
    ssd = 0

    ssd = torch.cat((np.square(greek_output[:,None] - greek_output[3]).sum(axis=2), example_targets) , 1)

    print(ssd[ssd[:, 0].sort()[1]])

    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(greek_output, example_targets)
    
    print("score = ",neigh.score(greek_output, example_targets))


if __name__ == "__main__":
    main(sys.argv)
