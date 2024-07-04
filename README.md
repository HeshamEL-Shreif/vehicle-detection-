# vehicle-detection
# README

## YOLOv9 Training for Vehicle and Pedestrian Detection

This repository contains a Jupyter Notebook for training a YOLOv9 (You Only Look Once, Version 9) model for detecting cars, pedestrians, trucks, and motorcycles. The notebook provides a step-by-step guide to preparing the dataset, configuring the YOLOv9 model, training the model, and evaluating its performance.

### Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset Preparation](#dataset-preparation)
3. [Environment Setup](#environment-setup)
4. [Model Configuration](#model-configuration)
5. [Training the Model](#training-the-model)
6. [Evaluation and Inference](#evaluation-and-inference)
7. [Acknowledgments](#acknowledgments)
8. [References](#references)

### Prerequisites

Before starting, ensure you have the following software and hardware requirements:

- Python 3.8 or later
- Jupyter Notebook
- CUDA-enabled GPU (NVIDIA recommended)
- PyTorch
- YOLOv9 repository and dependencies

### Dataset Preparation

1. **Data Collection**: Gather images containing cars, pedestrians, trucks, and motorcycles. Ensure the dataset is diverse and contains various scenes and lighting conditions.
   
2. **Annotation**: Annotate the images using a tool like LabelImg or CVAT. The annotations should be in YOLO format (i.e., `class x_center y_center width height`).

3. **Directory Structure**: Organize the dataset into the following directory structure:
   ```
   /path/to/dataset
   ├── images
   │   ├── train
   │   ├── val
   │   └── test
   └── labels
       ├── train
       ├── val
       └── test
   ```
   dataset used to train the model is annotated using roboflow and it contains 5 classes (car, pedestrain, truck, motorcycle and bus)

### Environment Setup

1. Clone the YOLOv9 repository:
   ```bash
   git clone https://github.com/ultralytics/yolov9.git
   cd yolov9
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify the installation:
   ```bash
   python detect.py --source path/to/sample/image.jpg
   ```

### Model Configuration

1. **Configuration File**: Create a custom configuration file for the dataset. This includes setting the number of classes and adjusting anchor boxes if necessary.
   
2. **Hyperparameters**: Adjust the hyperparameters such as learning rate, batch size, and epochs in the training script.

3. **Data Configuration**: Create a `.yaml` file specifying the paths to the training, validation, and test sets.

### Training the Model

1. **Start Training**: Execute the training script with the specified configuration and dataset paths:
   ```bash
   python train.py --img 640 --batch 16 --epochs 100 --data custom_dataset.yaml --cfg yolov9.yaml --weights yolov9.pt
   ```

2. **Monitor Training**: Use tools like TensorBoard to monitor the training process, visualize losses, and track the model's performance.

### Evaluation and Inference

1. **Evaluation**: After training, evaluate the model's performance on the validation and test sets:
   ```bash
   python val.py --data custom_dataset.yaml --weights path/to/trained_weights.pt
   ```

2. **Inference**: Run inference on new images or video files to test the trained model:
   ```bash
   python detect.py --source path/to/image_or_video --weights path/to/trained_weights.pt
   ```
   the link of the best.pt https://drive.google.com/file/d/153UJCTQEPKjp_sfvN3ygqQahHTXX60jc/view?usp=share_link

### Acknowledgments

- [Ultralytics YOLOv9](https://github.com/ultralytics/yolov9) repository for providing the base implementation and pre-trained weights.
- [LabelImg](https://github.com/tzutalin/labelImg) for the annotation tool.
- The open-source community for contributing valuable resources and support.

### References

- YOLOv9 Paper: [YOLOv9: An Improved YOLO Algorithm](https://arxiv.org/abs/2104.00761)
- PyTorch Documentation: [PyTorch](https://pytorch.org/docs/stable/index.html)
- YOLOv9 GitHub: [Ultralytics YOLOv9](https://github.com/ultralytics/yolov9)

---

Feel free to open issues or contribute to this project. Happy training and detecting!

---
