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

**EuroSAT Dataset**

- **GitHub repository:** https://github.com/phelber/EuroSAT
- **Source:** Sentinel-2 satellite images from the European Space Agency (Copernicus program), covering cities across Europe.
- **Size:** About 27,000 labeled and geo-referenced image patches.
- **Classes (10):** Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial, Pasture, Permanent Crop, Residential, River, and Sea/Lake.
- **Class size:** Roughly 2,000 to 3,000 images per class.
- **Image format:** 64 × 64 pixel patches. An RGB version is available, plus a multispectral version with 13 spectral bands. This project will use the **RGB version**.
- **Related paper:** Helber, P., Bischke, B., Dengel, A. & Borth, D. (2019), *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification*, IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.
- **License:** The repository is reported to use the MIT license. This will be confirmed on the repository page before use, and the dataset will be credited to its original authors.

> Note: The exact image counts per class will be recorded after exploring the downloaded copy of the dataset.

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
| 1 | Finalize dataset choice and review its license |
| 2 | Explore the data and check the ten classes |
| 3 | Build the data pipeline (splits, transforms, loaders) |
| 4 | Train the baseline and fine-tuned models |
| 5 | Evaluate, analyze errors, and document results |
