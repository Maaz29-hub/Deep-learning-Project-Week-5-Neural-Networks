# Week 5: Neural Networks, Convolutions, and Transfer Learning

## Project Overview

This project is part of **Project 2: Deep Learning Applications** for the course Introduction to Applied AI. The purpose of this project is to understand the fundamentals of neural networks, convolutional neural networks (CNNs), and transfer learning through practical image classification.

The project uses the Intel Image Classification dataset and a pretrained ResNet-18 model to classify images into six different scene categories. Throughout this lab, I explored image processing, convolution operations, parameter counting, data augmentation, model training, fine-tuning, and error analysis.

## Objectives

- Understand how digital images are represented as numerical tensors.
- Learn how convolution operations detect visual features.
- Calculate convolution output dimensions and neural network parameters.
- Prepare training, validation, and testing datasets.
- Apply image transformations and data augmentation.
- Implement a reusable training loop using PyTorch.
- Understand transfer learning and feature extraction.
- Fine-tune a pretrained ResNet-18 model.
- Evaluate image classification models using appropriate metrics.
- Analyze classification errors using a confusion matrix and error gallery.
- Save the trained model for future use.

## Dataset

**Intel Image Classification Dataset**

Dataset link: https://www.kaggle.com/datasets/puneet6060/intel-image-classification

The dataset contains images belonging to six scene categories:

1. Buildings
2. Forest
3. Glacier
4. Mountain
5. Sea
6. Street

The dataset provides separate training and testing folders. A validation set is created from the training data to monitor the model during training.

## Technologies and Libraries

- **Programming Language:** Python
- **Deep Learning Framework:** PyTorch
- **Computer Vision:** Torchvision and Pillow (PIL)
- **Numerical Computing:** NumPy
- **Data Visualization:** Matplotlib
- **Model Evaluation:** Scikit-learn
- **Development Environment:** Kaggle Notebook

## Project Workflow

### Part 1: The Data, and an Image as Numbers

Imported the required libraries, checked GPU availability, explored the dataset structure, counted images in each class, visualized sample images, and examined image dimensions and pixel values.

### Part 2: Convolution and Parameters by Hand

Implemented a convolution operation using PyTorch, explored edge detection on a real image, and studied convolution output dimensions, stride, max pooling, and parameter counting.

### Part 3: Transforms and Data Loaders

Prepared images for the ResNet-18 model by resizing and normalizing them. Applied data augmentation to training images and created separate DataLoaders for training, validation, and testing.

### Part 4: Training Loop and Feature Extraction

Implemented reusable training and evaluation functions. Loaded a pretrained ResNet-18 model, froze its existing parameters, and replaced the final fully connected layer to support the six scene categories.

### Part 5: Fine-Tuning and Learning Curves

Unfroze the final network block, `layer4`, and used different learning rates for the pretrained block and the new classification head. Plotted training and validation loss and accuracy to study the learning process.

### Part 6: Test Evaluation and Error Analysis

Evaluated the trained model on the test dataset. Used a classification report, confusion matrix, and error gallery to examine predictions and understand classification mistakes. Also explored logits and softmax probabilities for an individual image.

### Part 7: Save and Share

Saved the model's learned parameters and organized the project for documentation and sharing through GitHub. This part also introduced planning for a domain-specific deep learning project in Week 6.

## Model Architecture

The project uses **ResNet-18**, a convolutional neural network pretrained on ImageNet.

The final fully connected layer is replaced with a new classification head containing six output classes. The model is trained in two phases:

- **Feature Extraction:** The pretrained network is frozen while the new classification head is trained.
- **Fine-Tuning:** The final network block is unfrozen so its features can adapt to the scene classification task.

This approach demonstrates how transfer learning can reuse previously learned visual features for a new image classification problem.

## Key Concepts Learned

- Image tensors and pixel representation
- Convolution filters and edge detection
- Parameter sharing in convolutional layers
- Image normalization and data augmentation
- Training, validation, and testing pipelines
- Forward propagation and backpropagation
- Transfer learning and feature extraction
- Fine-tuning pretrained neural networks
- Learning curves and overfitting
- Classification reports and confusion matrices
- Softmax probabilities and error analysis
- Saving and reusing trained model parameters

## Saved Model

The model is saved using the filename:

`resnet18_intel.pt`

The saved file contains the model's learned parameters. To reuse it, the compatible ResNet-18 architecture must be recreated with a six-class output layer before loading the saved state dictionary.

## Conclusion

This project provided practical experience with deep learning and image classification using PyTorch. It helped me understand how convolutional neural networks process images, how transfer learning reuses pretrained features, and how fine-tuning adapts a model to a specific task.

I also learned how to prepare image datasets, implement a training pipeline, visualize learning curves, investigate classification errors, and save a trained model. These concepts provide a foundation for developing more specialized deep learning applications in future projects.

## Week 6: Future Work

For the next phase, I plan to explore a domain-specific image classification problem, such as crop disease detection, manufacturing defect detection, document classification, or satellite image classification.

The next step will be to select a suitable labeled dataset, review its license, understand the classification problem, and investigate how deep learning and transfer learning can be applied to the selected domain.

## Author

- **Student:** Maaz Imtiaz
- **Department:** Electronics
- **Course:** Introduction to Applied AI
- **Project:** Project 2 — Deep Learning Applications
- **Lab:** Week 5 — Neural Networks, Convolutions, and Transfer Learning

## Acknowledgment

This project follows the Week 5 laboratory instructions for Introduction to Applied AI. The Intel Image Classification dataset is available through Kaggle.
