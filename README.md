# HeadHunter - Brain Tumor Detector

Brain tumors can have a significant impact on an individual's quality of life and can lead to serious health problems if not detected and treated promptly. Early and accurate diagnosis of brain tumors is crucial for successful treatment. With advancements in deep learning and computer vision, it is now possible to automate the process of brain tumor detection and segmentation with high accuracy.

Introducing the Brain Tumor Segmentation Streamlit app, an innovative solution for detecting brain tumors using a trained Unet Model. This app allows you to upload a brain MRI scan and get an instant prediction of the presence of a brain tumor and its segmentation, all within a user-friendly interface. The app utilizes a state-of-the-art deep learning algorithm that has been trained on a large dataset of brain MRI scans to accurately detect brain tumors with high precision. Get started with our app now and take the first step towards a more efficient and effective way of diagnosing brain tumors.

<img width="1437" alt="Screenshot 2023-02-08 at 12 02 17" src="https://user-images.githubusercontent.com/60676144/217518605-88ccdd53-e46c-4bf4-b849-adeef117c610.png">

# Dataset

The dataset used to train the Unet Model can be found here: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

# UNet Model Architecture

The UNet Model Architecture was inspired from: https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5

![Unet-Architecture](https://user-images.githubusercontent.com/60676144/218311881-60baca16-c791-44a8-96eb-b4a6fa165866.png)

UNet architecture consists of two main parts: an encoder path and a decoder path. The encoder path is a traditional convolutional neural network (CNN) that downsamples the input image to reduce its spatial resolution and increase the depth of the feature maps. The decoder path then upsamples the feature maps to reconstruct the original image size while recovering the spatial details that were lost in the encoding process.

The UNet architecture has a unique U-shaped structure where the encoder and decoder paths are connected by "skip connections" which provide high-resolution information from the encoder to the decoder. This allows the decoder to make more accurate predictions because it has access to both the high-level features learned by the encoder and the low-level details from the input image.

Here is a step-by-step explanation of how the UNet architecture works:

1- Input: The input to the UNet is an image.

2- Encoder: The encoder consists of several blocks of convolutional and pooling layers. Each block increases the depth of the feature maps while reducing the spatial resolution by a factor of 2. The number of filters in each block is typically doubled, allowing the network to learn increasingly complex features.

3- Skip Connections: The outputs of each block in the encoder are connected to the corresponding block in the decoder by skip connections. These connections allow the high-resolution information from the encoder to be combined with the upsampled feature maps from the decoder.

4- Decoder: The decoder consists of several blocks of upsampling and convolutional layers that increase the spatial resolution of the feature maps while reducing their depth. The decoder uses the information from the skip connections to make more accurate predictions.

5- Output: The final layer of the decoder produces a segmentation map where each pixel is assigned a class label based on the learned features.

UNet has proven to be a highly effective architecture for image segmentation tasks and has been used in various medical imaging applications such as brain tumor segmentation and retinal vessel segmentation. Its U-shaped structure and use of skip connections allow it to effectively balance the tradeoff between spatial resolution and feature abstraction, making it a powerful tool for image segmentation tasks.

# Application Setup

## 1- Clone this repository
Use `git clone` command to clone this repository on your own device or directory.

## 2- Create a Conda Environment
Use the command `conda create --name {ENV_NAME}` to create an environment and `conda activate {ENV_NAME}` to activate it

## 3- Installing Dependencies
Use the command `conda install pip` to install and use pip in your environment

Run the command `pip install -r requirements.txt` to install the required packages to run the app

## 4- Run Application
Run the command `streamit run src/app.py` to run your streamlit application on your browser



