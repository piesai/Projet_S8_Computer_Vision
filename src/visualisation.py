import os 
import matplotlib.pyplot as plt
L = []
for k in range(42):
    rep = '/home/pierres/Projet_S8/data/train/' + str(k)
    nombre_de_fichier_1 = len([f for f in os.listdir(rep) if os.path.isfile(os.path.join(rep, f))])
    rep2 = '/home/pierres/Projet_S8/data/validation/' + str(k)
    nombre_de_fichier_2 = len([f for f in os.listdir(rep2) if os.path.isfile(os.path.join(rep2, f))])
    print(nombre_de_fichier_1,nombre_de_fichier_2)