import os 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import random as rd
from data_clean import dataset
L = []

#affichage du nombre de fichier
for k in range(42):
    rep = '/home/pierres/Projet_S8/data/train/' + str(k)
    nombre_de_fichier_1 = len([f for f in os.listdir(rep) if os.path.isfile(os.path.join(rep, f))])
    rep2 = '/home/pierres/Projet_S8/data/validation/' + str(k)
    nombre_de_fichier_2 = len([f for f in os.listdir(rep2) if os.path.isfile(os.path.join(rep2, f))])
    print(nombre_de_fichier_1,nombre_de_fichier_2)


#essaye de la bibiliothéque cv2
im = cv2.imread("/home/pierres/Projet_S8/data/train/0/00000_00005_00003.png")
cv2.resize(im,(32,32))
im2 = cv2.imread("/home/pierres/Projet_S8/data/train/0/00000_00000_00003.png")
cv2.resize(im2,(32,32))


#affichage de dimensions d'images
for k in range(42):
    rep = '/home/pierres/Projet_S8/data/train/' + str(k)
    for m in range(10):
        l = rd.randint(0,len(os.listdir(rep))-1)
        print(cv2.imread(rep + '/' + os.listdir(rep)[l]).shape)
    


#affichage de 10 images aléatoires
print(dataset.__getitem__(2))
for k in range(10):
    m = rd.randint(0,len(dataset))
    plt.imshow(dataset.__getitem__(m)[0].permute(1,2,0))
    plt.title(dataset.__getitem__(m)[1])
    plt.show()



