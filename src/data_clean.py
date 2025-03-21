import torchvision
import torch
from  torchvision.transforms import v2
import matplotlib.pyplot as plt
import random as rd
import numpy as np
dataset = torchvision.datasets.GTSRB(root = '/home/pierres/Projet_S8/data/',
                           download=True,
                           transform = v2.Compose([v2.Resize(size = (200,200)),v2.ToTensor(),v2.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))

#split en train test
data_train, data_test = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)),int(0.2*len(dataset))])
trainloader = torch.utils.data.DataLoader(data_train, batch_size=16,shuffle = True)
trainloader_test = torch.utils.data.DataLoader(data_train, batch_size=16,shuffle = True)


dataiter = iter(trainloader)
images, labels = next(dataiter)
plt.imshow(images[0].permute(1,2,0))
print(labels)
plt.show()