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

---

# Week 6: Project Proposal — Steel Surface Defect Detection for Manufacturing Quality Control

## Proposed Project Title

**Hot-Rolled Steel Surface Defect Classification Using Transfer Learning with ResNet-18**

## Problem Statement

In steel manufacturing, surface defects such as cracks, scratches, and inclusions reduce product quality and can cause failures in later use. Manual inspection is slow, tiring, and inconsistent between inspectors. This project proposes an image classification system that automatically identifies the type of defect on a steel strip surface from a photograph, supporting automated quality control on a production line.

## Domain and Motivation

- **Domain:** Manufacturing and industrial quality inspection
- **Why this domain:** Automated visual inspection is a practical application in industrial electronics and automation, where fast and consistent defect detection reduces waste and cost.
- **Why deep learning:** Different defects have distinct textures and patterns that CNNs can learn automatically, without hand-designed features.
- **Link to Week 5:** The same workflow from this lab (data loaders, augmentation, feature extraction, fine-tuning, and error analysis) will be applied to a new industrial dataset.

## Proposed Dataset

**NEU Surface Defect Database (NEU-CLS)**

- **Source:** Surface Inspection Laboratory, Northeastern University (NEU), China
- **Official page:** http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
- **Kaggle copy:** https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
- **Example GitHub project using this dataset:** https://github.com/rccohn/NEU-Cluster
- **Size:** 1,800 images in total.
- **Classes (6):** Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, and Scratches.
- **Balance:** 300 images per class, so the dataset is perfectly balanced.
- **Image format:** Grayscale, 200 × 200 pixels.
- **Origin:** Images were collected from real hot-rolled steel production.
- **Dataset challenges:** Defects within the same class can look very different, while defects from different classes can look similar. Lighting and material variations also affect the images.
- **Related paper:** Song, K. & Yan, Y. (2013), *A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects*, Applied Surface Science, 285.
- **License:** To be confirmed before use. The dataset is publicly available for research, and it will be credited to the original authors.

> Note: The dataset is also available as NEU-DET, which adds bounding-box annotations for defect localization. This project uses the image-only classification version (NEU-CLS).

## Proposed Approach

1. **Data exploration:** Count images per class and visualize sample defects from each class.
2. **Data splitting:** Create stratified training, validation, and test sets (for example 70% / 15% / 15%).
3. **Preprocessing:** Convert grayscale images to 3 channels, resize to 224×224, and normalize with ImageNet mean and standard deviation.
4. **Augmentation:** Apply random flips and small rotations to the training set only. Because the dataset is small, augmentation is important to reduce overfitting.
5. **Baseline model:** Use a pretrained ResNet-18 as a frozen feature extractor with a new six-class head.
6. **Fine-tuning:** Unfreeze `layer4` and train with different learning rates for the pretrained block and the new head, as done in Week 5.
7. **Evaluation:** Report accuracy, per-class precision, recall, and F1-score, plus a confusion matrix.
8. **Error analysis:** Build an error gallery and study which defect types are confused with each other, and why.
9. **Saving:** Save the trained weights and document how to reload them.

## Tools and Environment

- Python, PyTorch, Torchvision, Pillow, NumPy
- Matplotlib and Scikit-learn for visualization and evaluation
- Kaggle Notebook (GPU) for training
- GitHub for version control and sharing

## Evaluation Metrics

- Overall accuracy on the held-out test set
- Per-class precision, recall, and F1-score
- Confusion matrix to find commonly confused defect types
- Training and validation learning curves to monitor overfitting

## Expected Challenges and Limitations

- **Small dataset:** With only 1,800 images, overfitting is a risk, so augmentation and transfer learning are important.
- **Similar-looking defects:** Some defect classes (for example crazing and rolled-in scale) may be confused.
- **Grayscale and low resolution:** The images are small, so fine details may be limited.
- **Real-world gap:** Results on this benchmark may not directly match performance on a live production line with different lighting and cameras.

## Expected Outcomes

- A working steel defect classifier built with transfer learning.
- A comparison between feature extraction and fine-tuning on a small industrial dataset.
- A written analysis of the model's strengths, mistakes, and limitations.
- A well-documented GitHub repository.

## Timeline (Week 6 Plan)

| Step | Task |
|------|------|
| 1 | Finalize dataset choice and review its license |
| 2 | Explore the data and check the six classes |
| 3 | Build the data pipeline (splits, transforms, loaders) |
| 4 | Train the baseline and fine-tuned models |
| 5 | Evaluate, analyze errors, and document results |

## Author

- **Student:** Maaz Imtiaz
- **Department:** Electronics
- **Course:** Introduction to Applied AI
- **Project:** Project 2 — Deep Learning Applications
- **Lab:** Week 5 — Neural Networks, Convolutions, and Transfer Learning
- **Proposal:** Week 6 — Domain-Specific Image Classification

## Acknowledgment

This project follows the Week 5 laboratory instructions for Introduction to Applied AI. The Intel Image Classification dataset is available through Kaggle. The Week 6 proposal uses the NEU Surface Defect Database created by the Surface Inspection Laboratory of Northeastern University, China.
