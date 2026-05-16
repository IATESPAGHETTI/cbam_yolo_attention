# CBAM YOLOv8 Attention for Agricultural Object Detection

This repository contains the implementation and evaluation of a YOLOv8-based object detection model enhanced with the **Convolutional Block Attention Module (CBAM)**. The models are trained and evaluated on agricultural datasets for precision detection.

## 🚀 Project Overview

The goal of this project is to improve the baseline YOLOv8 performance by integrating CBAM, which applies both spatial and channel-wise attention. This allows the model to better focus on critical features in complex background environments.

### Key Components
* **Baseline YOLOv8:** Standard PyTorch implementation of YOLOv8n.
* **CBAM-YOLOv8:** Custom architecture modifying YOLOv8 (cbam_yolov8n.yaml) to include CBAM attention layers for enhanced feature extraction.
* **Comparative Analysis:** Thorough evaluation and comparison between the baseline and the attention-enhanced models.

## 📊 Findings & Evaluation

Our detailed evaluations (visualized in the notebooks and cadi_ai directories) include:
* **Enhanced Feature Extraction:** The attention mechanism helps suppress background noise and focus on foreground target objects.
* **Precision-Recall (PR) Curves:** Measured the trade-off between precision and recall across different classes, showing reliable detection thresholds (ig5_pr_curves.png, ig6_pr_curves.png).
* **F1-Score Comparison:** Analyzed the harmonic mean of precision and recall to confirm the robustness of the attention-based models vs. the baseline.
* **IoU Distributions:** Showcased the Intersection over Union distributions to assess the bounding box regression accuracy and object localization improvements (ig7_iou_distribution.png).
* **Per-Class Performance:** Detailed breakdown of detection capabilities across different object classes using radar charts and per-class PR curves.

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ``bash
   git clone https://github.com/IATESPAGHETTI/cbam_yolo_attention.git
   cd cbam_yolo_attention
   ``

2. **Install dependencies:**  
   Ensure you have Python 3.8+ installed. You will need standard deep learning and vision libraries:
   ``bash
   pip install torch torchvision ultralytics matplotlib pandas jupyter
   ``

3. **Run the Notebooks:**  
   Open and execute the main assignment notebooks to view the training pipeline, attention module implementation, and results analysis:
   ``bash
   jupyter notebook "cadi_ai/CADI_AI_Final.ipynb"
   ``

## 📁 Repository Structure
* cadi_ai/ - Contains dataset configurations (data.yaml), custom YAML definitions (cbam_yolov8n.yaml), architecture diagrams, and generated evaluation plots.
* *.ipynb - Jupyter notebooks containing the training code, model definitions, dataset preprocessing, and result generation.
* scripts/ - Python utilities like save_images.py, update_report.py, and update_notebook_images.py.

*(Note: Heavy datasets, output runs, and PyTorch model weights .pt are ignored via .gitignore to maintain a clean and lightweight repository).*
