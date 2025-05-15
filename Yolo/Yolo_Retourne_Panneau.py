import torch
from PIL import Image
import numpy as np

chemin_vers_image = r"C:\Users\gasti\Pictures\Screenshots\Capture d’écran (9).png"



def detect_possible_signs(image_path):
    # Charger le modèle pré-entraîné sur COCO
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # Exécuter l'inférence
    results = model(image_path)

    # Obtenir les noms de classes et les prédictions
    labels = results.names
    detections = results.xyxy[0]  # (x1, y1, x2, y2, conf, cls)

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    cropped_results = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        label = labels[int(cls)].lower()

        # Chercher les objets pertinents (panneaux, feux...)
        if "sign" in label or "light" in label:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cropped = image_np[y1:y2, x1:x2]
            cropped_image = Image.fromarray(cropped)
            cropped_results.append((label, conf, cropped_image))

    return cropped_results

# Exemple d'utilisation
results = detect_possible_signs(chemin_vers_image)

if results:
    for i, (label, conf, img) in enumerate(results):
        print(f"{i+1}. Détection : {label} (confiance : {conf:.2f})")
        img.show()  # Ou img.save(f"output_{i}.jpg")
else:
    print("Aucun objet ressemblant à un panneau détecté.")
