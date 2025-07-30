# flower-classification
Classifies flower images using CNN AI

This project aims to classify flower images into five distinct categories using Convolutional Neural Networks (CNN) and Transfer Learning. We experimented with both custom-built CNNs and pre-trained models including VGG16, VGG19, ResNet50V2, and MobileNetV2 to compare performance.

 Dataset 
Source: Kaggle

Dataset name: 5 Flower Types Classification Dataset

Total images: 5,000 (1,000 images for each category)

Categories: Orchid, Sunflower, Tulip, Lotus, Lilly

Image format: jpg, png, webp

Image size: 224 x 224

Structure: Each class has its own subfolder inside the main "flower_images" folder

Models Used
The following models were trained and evaluated:

CNN Model 1: Accuracy 95.44%, Loss 0.1727

CNN Model 2: Accuracy 97.13%, Loss 0.1794

VGG16: Accuracy 97.98%, Loss 0.0921

VGG19: Accuracy 97.98%, Loss 0.0795

ResNet50V2: Accuracy 98.19%, Loss 0.0436

MobileNetV2: Accuracy 96.71%, Loss 0.0611

ResNet50V2 achieved the highest accuracy among all models.

Key Features
Dataset analysis with visualization of sample images and class distribution

Image preprocessing including normalization and resizing

Data splitting into training, testing, and prediction sets

Training of custom CNNs and pre-trained models

Evaluation of model performance using accuracy and loss metrics

Real-time prediction pipeline for unseen flower images

Project Structure
Notebooks: Data analysis, CNN models, Transfer Learning models

Dataset: flower_images directory with subfolders for each class

Saved models: Pre-trained model weights in .h5 format

Utilities: Scripts for preprocessing and data splitting

How to Run
Clone the repository:

bash
Copy
Edit
git clone https:neural-finall (1) (3).ipynb
cd flower-classification
Install dependencies:

nginx
Copy
Edit
pip install -r requirements.txt
Run the Jupyter Notebooks or Python scripts for:

Data loading and preprocessing

Model training and evaluation

Making predictions on new images

Requirements
TensorFlow / Keras

OpenCV

NumPy

Pandas

Matplotlib

Seaborn

scikit-learn

tqdm

os / shutil / random

Install all using the requirements.txt or manually via pip.

Conclusion
This project demonstrates the effectiveness of CNNs and Transfer Learning in classifying flower images. By evaluating several models, we found that ResNet50V2 provides the best performance with an accuracy of 98.19%. The project follows a full AI workflow including data preparation, training, evaluation, and deployment-ready prediction pipeline.

 
