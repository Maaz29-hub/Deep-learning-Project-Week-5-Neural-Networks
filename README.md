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

# Week 6: Project Proposal — Land Use and Land Cover Classification from Satellite Images

## Proposed Project Title

**Satellite Image Land Use Classification Using Transfer Learning with ResNet-18**

## Problem Statement

Governments, city planners, and environmental agencies need up-to-date information about how land is used, such as residential areas, industrial zones, highways, forests, and water bodies. Checking this manually across large regions is slow and expensive. This project proposes an image classification system that automatically identifies the land use or land cover type from a satellite image patch.

## Domain and Motivation

- **Domain:** Remote sensing and satellite imagery
- **Why this domain:** Automated land classification supports urban planning, environmental monitoring, disaster response, and infrastructure mapping.
- **Why deep learning:** Different land types have distinct textures, shapes, and layouts (for example road lines, building grids, water surfaces), which CNNs can learn automatically.
- **Link to Week 5:** The same workflow from this lab (data loaders, augmentation, feature extraction, fine-tuning, and error analysis) will be applied to a new satellite dataset.

## Proposed Dataset

**EuroSAT: Land Use and Land Cover Classification Dataset**

- **GitHub repository (by the dataset creators):** https://github.com/phelber/EuroSAT
- **Official download (Zenodo):** https://zenodo.org/records/7711810
- **Paper:** Helber, P., Bischke, B., Dengel, A. & Borth, D., *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification*. https://arxiv.org/abs/1709.00029
- **Source imagery:** Sentinel-2 satellite images from the European Space Agency's Copernicus program.
- **Size:** 27,000 labeled and geo-referenced images.
- **Classes (10):** Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial, Pasture, Permanent Crop, Residential, River, and Sea/Lake.
- **Versions:** An RGB version (`EuroSAT_RGB.zip`, JPEG images) and a multispectral version (`EuroSAT_MS.zip`, all 13 Sentinel-2 bands). This project uses the **RGB version**.
- **Image size:** 64 × 64 pixel patches.
- **License and terms:** Sentinel data is free and open to the public under EU law. The dataset is used under the Copernicus Sentinel Data Terms and Conditions, and it is credited to its original authors.

> Note: The exact number of images per class will be recorded after exploring the downloaded copy of the dataset.

## Proposed Approach

1. **Data exploration:** Count images per class, check class balance, and visualize samples from each of the 10 classes.
2. **Data splitting:** Create stratified training, validation, and test sets (for example 70% / 15% / 15%).
3. **Preprocessing:** Resize the 64×64 patches to 224×224 and normalize with ImageNet mean and standard deviation.
4. **Augmentation:** Apply random horizontal and vertical flips and 90-degree rotations to the training set only. Satellite images have no fixed "up" direction, so these transformations are safe.
5. **Baseline model:** Use a pretrained ResNet-18 as a frozen feature extractor with a new ten-class head.
6. **Fine-tuning:** Unfreeze `layer4` and train with different learning rates for the pretrained block and the new head, as done in Week 5.
7. **Evaluation:** Report accuracy, per-class precision, recall, and F1-score, plus a confusion matrix.
8. **Error analysis:** Build an error gallery and study which land types are confused with each other, and why.
9. **Saving:** Save the trained weights and document how to reload them.

## Tools and Environment

- Python, PyTorch, Torchvision, Pillow, NumPy
- Matplotlib and Scikit-learn for visualization and evaluation
- Kaggle Notebook (GPU) for training
- GitHub for version control and sharing

## Evaluation Metrics

- Overall accuracy on the held-out test set
- Per-class precision, recall, and F1-score
- Confusion matrix to find commonly confused land types
- Training and validation learning curves to monitor overfitting

## Expected Challenges and Limitations

- **Similar-looking classes:** Some classes look alike from above, such as Annual Crop, Permanent Crop, and Pasture, or Highway and River.
- **Low resolution:** The 64×64 patches limit fine detail, and upscaling to 224×224 does not add new information.
- **Geographic bias:** The images come from European cities, so the model may not generalize to other regions with different landscapes and building styles.
- **Seasonal and lighting variation:** Appearance changes with season and weather.

## Expected Outcomes

- A working land use classifier built with transfer learning.
- A comparison between feature extraction and fine-tuning on satellite imagery.
- A written analysis of the model's strengths, mistakes, and limitations.
- A well-documented GitHub repository.

## Timeline (Week 6 Plan)

| Step | Task |
|------|------|
| 1 | Finalize dataset choice and review its terms of use |
| 2 | Explore the data and check the ten classes |
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

This project follows the Week 5 laboratory instructions for Introduction to Applied AI. The Intel Image Classification dataset is available through Kaggle. The Week 6 proposal uses the EuroSAT dataset created by Helber et al., based on Sentinel-2 satellite imagery from the European Space Agency (Copernicus program).
