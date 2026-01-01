# Multi-Class Retinal Disease Detection

A deep learning-based system to automatically detect and classify multiple retinal diseases from fundus imaging.  
This project leverages Convolutional Neural Networks (CNNs) and transfer learning to classify retinal images into several disease categories, assisting early diagnosis and decision support for ophthalmic care.

## Overview

Retinal diseases like diabetic retinopathy, macular degeneration, glaucoma, and others are leading causes of vision impairment worldwide. Automated detection using deep learning can significantly improve screening efficiency and diagnostic accuracy, especially in resource-constrained settings. :contentReference[oaicite:0]{index=0}

This repository implements a **multi-class deep learning model** that:

- Ingests retinal fundus images  
- Applies preprocessing and augmentation  
- Trains a CNN classifier for multiple disease categories  
- Evaluates performance with standard metrics  
- Provides inference code for new images

---

## Features

- **Multi-class classification** of retinal diseases  
- **Transfer learning** with pre-trained CNN backbones  
- **Data augmentation & preprocessing** to improve generalization  
- **Model evaluation** using accuracy, precision, recall, and confusion matrices  
- Easily extendable to additional diseases or datasets

## Project Structure
├── data/ # Dataset folders (train, validation, test)
├── notebooks/ # Jupyter notebooks for training & visualization
├── models/ # Saved model weights/checkpoints
├── src/ # Core training and evaluation scripts
│ ├── train.py
│ ├── evaluate.py
│ ├── preprocess.py
│ └── utils.py
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore


## Dataset

Use a labeled retinal fundus image dataset containing multiple disease categories.  
Typical datasets include images annotated with labels such as:

- Normal
- Diabetic Retinopathy
- Glaucoma
- Age-Related Macular Degeneration
- Other retinal conditions

> **Important**: Ensure the dataset is organized into `train/validation/test` folders with sub-folders for each class. This helps with automatic PyTorch/Keras data loading.


## Installation

1. **Clone the repository**

```bash
git clone https://github.com/siddhijaiswal08/Multi-class-retinal-disease-detection.git
cd Multi-class-retinal-disease-detection
```
2. **Create & activate a virtual environment**
``` bash
python3 -m venv venv
source venv/bin/activate           # macOS/Linux
venv\Scripts\activate              # Windows
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Download dataset and place it in /data**
data/

├── train/

│   ├── Normal/

│   ├── Disease1/

│   └── DiseaseN/

├── val/

└── test/

## Training
To train the model 
```bash
python src/train.py \
  --data_dir data/ \
  --model_name resnet50 \
  --epochs 30 \
  --batch_size 16 \
  --learning_rate 1e-4
```
Training highlights:

Uses transfer learning with a pre-trained backbone (e.g., ResNet50)

Applies augmentation (flip, rotation, contrast) during training

Validates on a separate validation set

## Evaluation & Inference
To evaluate on the test set:
```bash
python src/evaluate.py \
  --model_path models/best_model.pth \
  --data_dir data/test/
```
Example usage for a single image:
```bash
python src/infer.py --image sample_fundus.jpg --model models/best_model.pth
```
Outputs class predictions and confidence scores.

## Results
Model performance is evaluated with:

Accuracy

Precision & Recall

F1-score

Confusion Matrix

A well-trained model should achieve high accuracy (>80–90%) depending on dataset size and quality. 

Visualizations and plots from training and evaluation are provided in the notebooks/ directory.

## Future Improvements

- Add more retinal disease categories

- Fine-tune with larger, clinically validated datasets

- Integrate explainability (Grad-CAM / saliency maps)
- Deploy as REST API or mobile app for real-time diagnosis

## Requirements
```bash
pip install -r requirements.txt
```
Key packages include:

- PyTorch / TensorFlow

- torchvision

- numpy, pandas

- opencv-python

- scikit-learn

- matplotlib

If want any help please contact : jaiswalsiddhi084@gmail.com
