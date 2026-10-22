# AI-Powered Conveyor Belt Damage Detection and Monitoring System

## Project Overview
Conveyor belts play a vital role in transporting materials in mining industries. Undetected belt damage can lead to equipment failure, production downtime, and increased maintenance costs.

This project uses **YOLOv8, Python, and Computer Vision** to detect visible conveyor belt defects from images and support early damage identification.

## Objectives
- Detect visible conveyor belt defects using deep-learning-based object detection.
- Combine and preprocess datasets with different annotation formats.
- Train and evaluate a YOLOv8 model for conveyor belt damage detection.
- Support future camera-based inspection and automated alerts.

## Technologies Used
- **Programming Language:** Python
- **AI Model:** YOLOv8
- **Domain:** Artificial Intelligence, Machine Learning, Computer Vision
- **Libraries:** Ultralytics, PyTorch, OpenCV
- **Training Platform:** Google Colab

## Defect Categories
The trained model was designed to detect six categories:
- Tear
- Hole
- Crack
- Puncture
- Patch Work
- Impact Damage

## Project Workflow
1. Collect conveyor belt defect datasets.
2. Inspect and preprocess dataset annotations.
3. Convert different annotation formats into a common YOLO format.
4. Merge relevant defect classes into a unified dataset.
5. Train the YOLOv8 model using a GPU.
6. Evaluate model performance and save the trained weights.
7. Test the model on conveyor belt images.

## Repository Structure
- `convert_datasets.py` — Converts dataset annotation formats.
- `inspect_datasets.py` — Inspects dataset structure and labels.
- `merge_datasets.py` — Merges datasets and maps defect classes.
- `train.py` — Trains the YOLOv8 model.
- `reports/` — Contains dataset audit, conversion, and merging reports.

## Current Status
Dataset preprocessing, merging, model training, and initial image-based testing have been completed. Webcam-based inspection and further model improvements remain ongoing work.

## Limitations and Future Scope
- Improve detection of small cracks and subtle defects.
- Validate the model on more diverse and real-world conveyor belt images.
- Integrate camera-based inspection and visual alerts.
- Explore industrial monitoring and predictive maintenance.

## Applications
- Mining and material handling industries
- Conveyor belt maintenance
- Industrial equipment inspection
- Early identification of visible belt damage

## Conclusion
This project explores the use of deep learning and computer vision to identify visible conveyor belt defects and support more efficient inspection and maintenance in mining environments.
