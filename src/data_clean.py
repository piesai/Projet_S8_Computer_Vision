import torchvision
import torch
import torchvision.transforms as v1
import matplotlib.pyplot as plt
import random as rd
dataset = torchvision.datasets.GTSRB(root = '/home/pierres/Projet_S8/data/',
                           download=True,
                           transform = v1.Compose([v1.PILToTensor(),v1.Pad(243)]))

print(dataset.__getitem__(2))
for k in range(10):
    m = rd.randint(0,len(dataset))
    plt.imshow(dataset.__getitem__(m)[0].permute(1,2,0))
    plt.title(dataset.__getitem__(m)[1])
    plt.show()


