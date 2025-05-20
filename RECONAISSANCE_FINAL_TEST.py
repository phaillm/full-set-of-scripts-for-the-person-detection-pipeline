from ultralytics import YOLO
import torch
import os
import cv2
import matplotlib.pyplot as plt

os.chdir('./dataset_segmentation/segmentation_dataset/')
print(os.getcwd())

model = YOLO("best_trained_model.pt")

image_path = "Test2/Image_test21.jpg" #Récupération de l'image
results = model(image_path, conf=0.6) #Seuil de confiance bas pour maximiser la détection

print(results[0].boxes)


filtered_boxes = [] #Ne garder que la classe person
for box in results[0].boxes:
    class_id = int(box.cls[0])  #Récupérer l'ID de la classe
    confidence = float(box.conf[0]) #Seuil de confiance dans la détection

    if class_id == 0 and 0.3 <= confidence <= 1:
        filtered_boxes.append(box)

results[0].boxes = filtered_boxes #Image avec les personnes détectées
result_image = results[0].plot()

print(len(filtered_boxes))

plt.imshow(result_image)
plt.axis("off")
plt.show()


