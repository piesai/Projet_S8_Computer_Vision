import torchvision
import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import random as rd
import numpy as np

dataset = torchvision.datasets.GTSRB(root='/home/pierres/Projet_S8/data',
                                     download=False,
                                     transform=v2.Compose([v2.Resize(size=(200, 200)), v2.ToTensor(), v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
def adjust_labels(dataset):
    
    new_dataset = []
    for item in dataset:
        if item[-1] > 8:
            break
        
        # Assuming the label is the last element in each item
        features = item[:-1]  # All elements except the last one are features
        
        label = item[-1]      # The last element is the label
        

        # Adjust the label to be between 0 and 8
        adjusted_label = max(0, min(label, 8))

        # Create a new item with adjusted label
        new_item = (*features, adjusted_label)
        new_dataset.append(new_item)

    return new_dataset

# Example usage:
# dataset = [(tensor1, tensor2, ..., label), (tensor1, tensor2, ..., label), ...]
# new_dataset = adjust_labels(dataset)
dataset = adjust_labels(dataset)

data_train, data_test = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))])
trainloader = torch.utils.data.DataLoader(data_train, batch_size=16, shuffle=True)
trainloader_test = torch.utils.data.DataLoader(data_train, batch_size=16, shuffle=True)

#dataiter = iter(trainloader)
#images, labels = next(dataiter)
#images, labels = images.cuda(), labels.cuda()
#plt.imshow(images[0].permute(1, 2, 0).cpu())
#print(labels.cpu())
#plt.show()  
