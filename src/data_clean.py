
import random
import os 
import shutil 

#création d'un train/validation/test
for k in range(1,2):
    dossier = '/home/pierres/Projet_S8/data/validation/' + str(k)
    os.makedirs(dossier, exist_ok=True)
    rep = '/home/pierres/Projet_S8/data/train/' + str(k)
    nombre_de_fichier = len([f for f in os.listdir(rep) if os.path.isfile(os.path.join(rep, f))])
    nombre_de_fichier_valid = len([f for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))])
    if nombre_de_fichier_valid != 0:
        print("Attention, le dossier" + '' + str(k) + '' + "n'est pas vide")
        break
    for m in range(int(nombre_de_fichier/5)):
        nombre_de_fichier_bis = len([f for f in os.listdir(rep) if os.path.isfile(os.path.join(rep, f))])
        i = random.randint(0,nombre_de_fichier_bis-1)
        f = os.listdir(rep)[i]
        shutil.move(rep + '/' + f,dossier)
