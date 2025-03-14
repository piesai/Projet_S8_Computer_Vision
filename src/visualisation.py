import os 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import random as rd
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
    



print(im.shape,im2.shape)

cv2.imshow('image',im)
cv2.imshow('image2',im2)
cv2.waitKey(0)
L = []
rep = '/home/pierres/Projet_S8/data/train/1/'
