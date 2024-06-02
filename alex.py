import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def traval_dataloader(datafile, batchsize, augment, seed, valratio= 0.1, shuffle=True):

    normalize = transforms.Normalize(mean = [0.4914, 0.4822, 0.4465],
                                   std = [0.2023, 0.1994, 0.2010])

    if augment:
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              #transforms.CenterCrop(100)
                                           transforms.Resize(227),
                                           transforms.ToTensor(),
                                           normalize])
    else:
        train_transform = transforms.Compose([transforms.Resize(227),
                                               transforms.ToTensor(),
                                               normalize])
    val_transform = transforms.Compose([transforms.Resize(227),
                                        transforms.ToTensor(),
                                        normalize])
    train_dataset = datasets.CIFAR10(root= datafile,
                                     train= True,
                                     download= True,
                                     transform= train_transform
                                     )
    val_dataset = datasets.CIFAR10(root= datafile,
                                   train= True,
                                   download= True,
                                   transform= val_transform)
    num_train = len(train_dataset)
    num_val = np.floor(valratio*num_train)

    incides = list(range(num_train))

    if shuffle:
        np.random.seed(seed= seed)
        np.random.shuffle(incides)

    train_idx = incides[int(num_val):]
    val_idx = incides[:int(num_val)]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= batchsize, sampler=train_sampler)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, sampler= val_sampler)

    return (train_dataloader, val_dataloader)

def test_loader(datafile, batchsize, shuffle):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

    test_transform = transforms.Compose([transforms.Resize(227),
                                         transforms.ToTensor(),
                                         normalize])
    test_dataset = datasets.CIFAR10(root= datafile,
                                    train= False,
                                    transform= test_transform,
                                    download= True
                                    )
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batchsize,shuffle=shuffle)
    return test_dataloader

class Alexnet(nn.Module):
    def __init__(self, nc=10):
        super(Alexnet, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(3, 96, 11, 4, 0),
                        nn.ReLU(),
                        nn.MaxPool2d(3, 2),
                        nn.Conv2d(96, 256, 5, 1, 2),
                        nn.ReLU(),
                        nn.MaxPool2d(3, 2),
                        nn.Conv2d(256, 384, 3, 1, 1),
                        nn.ReLU(),
                        nn.Conv2d(384, 384, 3, 1, 1),
                        nn.ReLU(),
                        nn.Conv2d(384, 256, 3, 1, 1),
                        nn.ReLU(),
                        nn.MaxPool2d(3, 2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features= nc),
        )
    def forward(self, x):
        x = self.conv(x)
        x= x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x

datafile = './data'
train_loader, val_loader = traval_dataloader(datafile= datafile, batchsize= 16, augment= True, seed= 1)
testloader = test_loader(datafile=datafile, batchsize=1, shuffle=False)

print("dataloader created!")

nc = 10
lr = 0.005
momentum = 0.9
weight_decay = 0.005
epochs =10

net = Alexnet(nc= nc).to(device)
print(net)
print("net created!")

loss_f = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),
                            lr= lr,
                            momentum=momentum,
                            weight_decay= weight_decay)
train_step = len(train_loader)

print("training strated!")

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        out = net(images)
        loss = loss_f(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch {}/{}, Step {}/{}, Loss:{:.4f}".format(
                 epoch+1, epochs, i+1, train_step, loss.item())
         )

    with torch.no_grad():
        total = 0
        positive = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            predic = net(images)

            total += labels.size(0)
            _, prediction = torch.max(predic, 1)
            positive += (prediction == labels).sum().item()

        print("Accuracy of the network is {} %".format(100 * positive / total))




