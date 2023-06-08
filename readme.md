# Car Issue Classification

This project aims to develop a Convolutional Neural Network (CNN) model for classifying car issues. The model will be trained to identify different types of issues such as scratches, dents, and broken glass in car images.

## Dataset

The dataset used for this project consists of labeled images of cars with different issues. The dataset is divided into three classes: 'scratches', 'dents', and 'broken_glass'. The labeled images are stored in the `labeled` folder, organized by class names.

## Model Architecture

The model architecture used for this project is based on the ResNet-18 architecture. The ResNet-18 model is a popular deep CNN model that has shown excellent performance on various image classification tasks. The last fully connected layer of the ResNet-18 model is modified to match the number of classes in the car issue classification problem.

## Prerequisites

- Python 3.7 or higher
- PyTorch library
- torchvision library

## Installation

1. Clone the repository:


2. Set up a virtual environment (optional but recommended):


3. Install the required dependencies:


## Usage

1. Prepare the labeled dataset:

- Place the labeled images (in XML format) in the respective class folders ('scratches', 'dents', 'broken_glass') inside the `labeled` folder.

2. Train the model:


This script will train the CNN model on the labeled dataset and save the trained model weights to a file.

3. Evaluate the model:


This script will evaluate the trained model on a test set and display the accuracy and other evaluation metrics.

4. Predict on new images:


Replace `<path-to-image>` with the path to the image you want to classify. The script will load the trained model and predict the class of the input image.

## Results

The trained model achieves an accuracy of XX% on the test set. The confusion matrix and other evaluation metrics are shown below:

|           | scratches | dents | broken_glass |
|-----------|-----------|-------|--------------|
| scratches |     XX    |  XX   |     XX       |
|   dents   |     XX    |  XX   |     XX       |
|broken_glass|     XX    |  XX   |     XX       |

## License

This project is licensed under the [MIT License](LICENSE).
