# Road Sign Detection and Classification

This project was developed in collaboration with **SysNav Engineers**, who provided valuable guidance and expertise. Their recommendations were instrumental in shaping the project, such as using **YOLO for segmentation in video frames**.

---

## Project Structure

### `Yolo/`
Contains all scripts for **YOLO-based image segmentation**, which is responsible for detecting and extracting road signs from images.

### `src/`
Includes the scripts that integrate **YOLO segmentation** with a **CNN classifier** to process videos **frame-by-frame**.

### `demo.mp4`
A demonstration video of the model, recorded outside **CentraleSupélec Metz Campus**. The video illustrates the detection and classification pipeline, but it is **not processed in real-time**.

---

## Overview
The project combines **YOLO for road sign detection** and a **CNN for classification**, processing videos frame-by-frame to identify and categorize road signs.
