import torch
import os
import glob


# Load a YOLOv5 model (options: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Default: yolov5s

# Define the input image source (URL, local file, PIL image, OpenCV frame, numpy array, or list)
#img = "https://ultralytics.com/images/zidane.jpg"  # Example image


    ### Attention avec les folders ###
img_folder = r"C:\Users\gasti\Pictures\Screenshots"  # Local image file

img_files = glob.glob(os.path.join(img_folder, "*.jpg")) + \
            glob.glob(os.path.join(img_folder, "*.png")) + \
            glob.glob(os.path.join(img_folder, "*.jpeg"))

#img =  os.path.join(img_folder)  # Local image file
 # List of images (can be a single image or multiple images)
# Perform inference (handles batching, resizing, normalization automatically)
results = model(img_files)

# Process the results (options: .print(), .show(), .save(), .crop(), .pandas())
results.print()  # Print results to console
results.show()  # Display results in a window   


