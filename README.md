# AI-Powered Conveyor Belt Damage Detection and Monitoring System

## Project Overview

Conveyor belts are widely used in mining and industrial environments to transport materials continuously. Damage such as tears, holes, cracks, and impact-related defects can interrupt operations, increase maintenance costs, and create safety risks.

This project explores the use of Artificial Intelligence (AI) and Computer Vision to detect visible conveyor belt damage using the YOLOv8 object detection model.

The system uses a trained deep learning model to identify selected types of conveyor belt damage from images. The project focuses on dataset preparation, model training, evaluation, and initial image-based testing.

## Objectives

- Develop an AI-based approach for detecting conveyor belt damage.
- Prepare and combine datasets for model training.
- Train a YOLOv8 object detection model.
- Evaluate the model using precision, recall, and mean Average Precision (mAP).
- Test the trained model on conveyor belt images.
- Explore future integration with camera-based inspection systems.

## Technologies Used

- **Programming Language:** Python
- **AI Framework:** Ultralytics YOLOv8
- **Deep Learning:** PyTorch
- **Computer Vision:** OpenCV
- **Dataset Preparation:** Python-based preprocessing and merging scripts
- **Training Environment:** Google Colab
- **Dataset Source:** Roboflow

## Defect Classes

The unified dataset was prepared to include the following six classes:

1. Tear
2. Hole
3. Crack
4. Puncture
5. Patch Work
6. Impact Damage

Detection performance varies between classes and depends on the availability and quality of training data.

## Project Workflow

1. **Dataset Inspection:** Inspect datasets and identify issues with images, annotations, and labels.
2. **Dataset Conversion:** Convert dataset annotations into the format required for YOLO training.
3. **Dataset Merging:** Combine compatible datasets to support multi-class damage detection.
4. **Model Training:** Train a YOLOv8 model using the prepared dataset.
5. **Model Evaluation:** Evaluate model performance using validation metrics.
6. **Image-Based Testing:** Test the trained model on sample conveyor belt images and examine its predictions.

## Model Performance

The trained model was evaluated using a validation dataset.

| Metric | Result |
|---|---:|
| Precision | 0.812 |
| Recall | 0.645 |
| mAP@0.50 | 0.646 |

These results represent the reported validation performance of the trained model. They do not guarantee the same performance in real industrial environments.

Some defect classes, particularly small cracks, require further training and testing because the available validation examples are limited.

## Current Project Status

- Dataset inspection and preprocessing scripts developed.
- Dataset conversion and merging workflows prepared.
- YOLOv8 model training completed.
- Initial image-based model testing completed.
- Validation metrics recorded.
- Webcam-based detection and real-time camera integration remain future work.
- Further improvements are needed to increase detection reliability, especially for small defects.

## Repository Structure

```text
ai-conveyor-belt-damage-detection/
│
├── README.md
├── requirements.txt
├── convert_datasets.py
├── inspect_datasets.py
├── merge_datasets.py
├── train.py
│
└── reports/
    ├── audit_report.txt
    ├── conversion_report.json
    └── merge_report.json
```

## Installation

Ensure Python is installed on your system. Clone the repository and install the required dependencies.

```bash
git clone https://github.com/harishmakamali/ai-conveyor-belt-damage-detection.git
cd ai-conveyor-belt-damage-detection
python -m pip install -r requirements.txt
```

The scripts may require appropriate dataset paths and configuration before they can be run. The trained model weights and datasets are not included in this repository.

## Future Improvements

- Improve detection of small and low-visibility defects.
- Expand the dataset with more diverse conveyor belt images.
- Improve class balance and validation coverage.
- Integrate a webcam or industrial camera for live inspection.
- Explore edge deployment for industrial monitoring.
- Investigate alert generation and maintenance reporting.

## Applications

- Conveyor belt condition monitoring
- Mining and material-handling industries
- Industrial maintenance support
- Automated visual inspection
- Predictive maintenance research

## Conclusion

This project demonstrates an initial application of YOLOv8-based object detection for conveyor belt damage identification. Dataset preparation, model training, validation, and initial image-based testing have been completed. Further work is required to improve detection reliability and implement camera-based inspection before the system can be considered ready for real-world deployment.

---

**Developed as an AI and Computer Vision project for industrial conveyor belt damage detection.**
