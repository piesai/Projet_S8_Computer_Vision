import torchvision
import torch
import torchvision.transforms as v1
import matplotlib.pyplot as plt
dataset = torchvision.datasets.GTSRB(root = '/home/pierres/Projet_S8/data/',
                           download=True,
                           transform = v1.Compose([v1.PILToTensor(),v1.Pad(300)]))

print(dataset.__getitem__(2))

plt.plot(dataset.__getitem__(2))

data_loader = torch.utils.data.DataLoader(dataset,batch_size =1,shuffle = True)

