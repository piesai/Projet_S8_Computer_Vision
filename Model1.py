import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader









            ### Définir les transformations à appliquer à l'ouverture du dataset ###

simple_transform = transforms.Compose([
    transforms.ToTensor()]) # Convertir les images PIL en tenseurs
"""    transforms.Normalize(),
    transforms.Pad(300)     # Exemple de transformation supplémentaire
])"""

            ### Charger le jeu de données GTSRB ###
dataset = torchvision.datasets.GTSRB(
    root=r'C:\Users\gasti\Documents\CS\2A\Projet_S8_Computer_Vision\Dataset\gtsrb',
    download=True,
    transform=simple_transform  # Appliquer les transformations définies: transformation en tenseur
)

data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)



sum_r, sum_g, sum_b = 0, 0, 0
sum_r2, sum_g2, sum_b2 = 0, 0, 0  # Pour les carrés des pixels
total_pixels = 0

# Parcours du DataLoader
for images, _ in data_loader:
    # Accumuler les valeurs des trois canaux et leurs carrés
    sum_r += images[:, 0, :, :].sum()  # Canal rouge
    sum_g += images[:, 1, :, :].sum()  # Canal vert
    sum_b += images[:, 2, :, :].sum()  # Canal bleu
    sum_r2 += (images[:, 0, :, :] ** 2).sum()  # Carré du canal rouge
    sum_g2 += (images[:, 1, :, :] ** 2).sum()  # Carré du canal vert
    sum_b2 += (images[:, 2, :, :] ** 2).sum()  # Carré du canal bleu
    total_pixels += images.numel() // 3  # Total de pixels (images.numel() donne le nombre d'éléments dans le tenseur)

# Calcul des moyennes pour chaque canal
mean_r = sum_r / total_pixels
mean_g = sum_g / total_pixels
mean_b = sum_b / total_pixels

# Calcul des variances (et des écarts-types) pour chaque canal
var_r = (sum_r2 / total_pixels) - (mean_r ** 2)
var_g = (sum_g2 / total_pixels) - (mean_g ** 2)
var_b = (sum_b2 / total_pixels) - (mean_b ** 2)

std_r = torch.sqrt(var_r)
std_g = torch.sqrt(var_g)
std_b = torch.sqrt(var_b)

# Afficher les résultats
print(f'Moyenne canal rouge: {mean_r:.4f}, Ecart-type: {std_r:.4f}')
print(f'Moyenne canal vert: {mean_g:.4f}, Ecart-type: {std_g:.4f}')
print(f'Moyenne canal bleu: {mean_b:.4f}, Ecart-type: {std_b:.4f}')



"""
simple_transform = transforms.Compose([
    transforms.Resize((300, 300)),  # Redimensionner à 300x300
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean_r, mean_g, mean_b], std=[1.0, 1.0, 1.0])  # Normalisation
])

# Recréer le dataset avec la transformation finale
dataset = torchvision.datasets.GTSRB("""
    #root=r'C:\Users\gasti\Documents\CS\2A\Projet_S8_Computer_Vision\Dataset\gtsrb',
"""train=True,
    download=True,
    transform=simple_transform
)

data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
"""




"""
# Exemple d'itération sur le DataLoader
if __name__ == '__main__':
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    print(images.size())  # Vérifiez la taille des images pour confirmer qu'elles sont des tenseurs


batch_size = 4



classes = tuple(str(i) for i in range(43))

print("ok1")







import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)
"""
"""# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))"""
print("ok2")


"""
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=8, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(8 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)






for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
"""






"""dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))"""



"""
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
"""