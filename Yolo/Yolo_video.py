import torch
import cv2


model = torch.hub.load("ultralytics/yolov5", "yolov5s", trust_repo=True)

# Ouvrir une vidéo (fichier local ou webcam : 0)
video_path = "chemin/vers/ta_video.mp4"
cap = cv2.VideoCapture(video_path)


save_output = True
output_path = "resultat_detection.avi"
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if save_output:
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inférence sur la frame (OpenCV: BGR → RGB)
    results = model(frame[..., ::-1])  # Convertir BGR en RGB

    # Récupérer l’image annotée (résultat) sous forme de tableau numpy
    annotated_frame = results.render()[0]  # render() modifie results.imgs

    # Afficher le résultat
    # cv2.imshow("Détection", annotated_frame)

    # Sauvegarder la vidéo si activé
    if save_output:
        out.write(annotated_frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libération des ressources
cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
