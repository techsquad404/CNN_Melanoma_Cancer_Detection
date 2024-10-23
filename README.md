# Melanoma Cancer Detection using Convolutional Neural Network (CNN)
## Project Overview

This project focuses on building a Convolutional Neural Network (CNN) model to accurately detect melanoma, a dangerous form of skin cancer. Melanoma detection is critical as it accounts for 75% of skin cancer-related deaths. An automated solution that evaluates skin images and alerts dermatologists can greatly reduce the manual effort in diagnosis and help with early detection.

## Motivation
The primary goal is to create a multi-class classification model using a custom CNN architecture implemented in TensorFlow. The model classifies images into various skin disease categories, including melanoma.

## Dataset
The dataset used consists of 2,357 images of malignant and benign skin diseases from the International Skin Imaging Collaboration (ISIC). These images represent various skin conditions, with the following nine classes:

Actinic Keratosis
Basal Cell Carcinoma
Dermatofibroma
Melanoma
Nevus
Pigmented Benign Keratosis
Seborrheic Keratosis
Squamous Cell Carcinoma
Vascular Lesion
Each class has a varying number of samples, with melanoma and nevus being more dominant.

## Steps Performed
### 1. Data Loading and Preprocessing
Image Dataset Loading: We used TensorFlow’s image_dataset_from_directory utility to load images from disk, ensuring the dataset was split into 80% training and 20% validation.
Data Augmentation:
To address class imbalance, we used the Augmentor library to create additional samples for under-represented classes like Actinic Keratosis and Seborrheic Keratosis. We added 500 images to each class to balance the dataset.
### 2. Model Development
Model Architecture: We designed a custom CNN model with normalization, dropout layers, and batch normalization to avoid overfitting and enhance training.
Optimization: 
The model was compiled using an appropriate optimizer and loss function suitable for multi-class classification.
### 3. Training and Evaluation
Training Process: The model was trained on the augmented dataset, and the effects of various techniques like dropout, batch normalization, and data augmentation were evaluated.
Performance Tuning: Rebalancing the classes helped reduce overfitting. However, balancing the dataset and applying augmentation led to lower accuracy initially, which was later improved by fine-tuning the model further.
### 4. Results
Underfitting/Overfitting: Initial models showed signs of overfitting. Introducing data augmentation and rebalancing the dataset helped mitigate this issue.
Model Accuracy: By using techniques like dropout, augmentation, and batch normalization, we improved the model's ability to generalize and reduce loss over time.

## Future Work
Further improvements can be made by experimenting with different architectures and advanced augmentation techniques.
Tuning the model to enhance accuracy for minority classes while maintaining overall performance.
## Conclusion
The CNN-based melanoma detection model built in this project demonstrates the potential for using deep learning in medical image analysis. Although the class imbalance posed challenges, the application of data augmentation, dropout, and batch normalization proved effective in improving the model's performance.
