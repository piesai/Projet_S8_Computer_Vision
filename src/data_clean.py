import torchvision
import torch
from  torchvision.transforms import v2
import matplotlib.pyplot as plt
import random as rd
dataset = torchvision.datasets.GTSRB(root = '/home/pierres/Projet_S8/data/',
                           download=True,
                           transform = v2.Compose([v2.PILToTensor(),v2.Resize(size = (200,200))])) #quel padding mettre sachant que la plus grosse image va jusqu'à 243


