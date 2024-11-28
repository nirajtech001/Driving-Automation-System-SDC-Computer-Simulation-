---
# Behavioral Cloning for Driving Automation System

This project demonstrates the development of a self-driving car simulation using a Convolutional Neural Network (CNN) to mimic human driving behavior. The model utilizes Computer Vision techniques for real-time image processing and decision-making, enabling autonomous navigation in a simulated environment.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Future Scope](#future-scope)
- [Acknowledgements](#acknowledgements)

## Project Overview
The goal of this project is to create a driving automation system capable of autonomous navigation in a simulated environment. Using Behavioral Cloning, the system learns from human driving data to predict appropriate steering angles based on the input from a front-facing camera.

The project demonstrates:
1. Data preprocessing and augmentation for improved model generalization.
2. Real-time integration of the trained model with a simulation environment using Flask and Socket.io.
3. Testing and validation of the model in different tracks.

### **Key Contributions**
- **Niraj**: Led the efforts in data preprocessing, building the CNN network, and integrating the model with the simulator.

## Features
- Real-time image processing and steering angle prediction.
- Integration with a Unity-based driving simulator.
- Interactive interface for visualization and control.
- Data augmentation techniques for robust model performance.

## Technology Stack
### Frontend
- **Unity Simulator**: Simulates the driving environment.

### Backend
- **Python**: Core programming language.
- **Flask & Socket.io**: Facilitates real-time communication between the simulation and the model.
- **Keras/TensorFlow**: Frameworks for building and training the CNN model.
- **OpenCV**: Used for image preprocessing and augmentation.

### Tools
- Jupyter Notebook, PyCharm, or VSCode for development.
- NVIDIA model architecture for CNN-based steering prediction.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository-url.git
   cd your-repository-folder
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the simulator and Flask server:
   - Start the Unity-based simulator in training mode.
   - Run the Flask server:
     ```bash
     python app.py
     ```

## Usage
1. **Data Collection**: Use the simulator in training mode to collect images and steering angles. Save the data for preprocessing.
2. **Model Training**: Train the CNN model using the collected dataset.
3. **Autonomous Driving**: Switch to autonomous mode in the simulator and use the trained model to predict steering angles in real-time.

## Dataset
The dataset consists of images captured from three camera views (left, center, right) and corresponding steering angles. Images undergo augmentation, including:
- Brightness adjustments.
- Horizontal flips.
- Cropping irrelevant parts.

## Model Architecture
The CNN model is based on NVIDIA's architecture for autonomous driving:
- Input layer processes 66x200 RGB images converted to YUV color space.
- Convolutional and pooling layers extract features.
- Fully connected layers output the steering angle.


## Results

- Successfully navigated through multiple tracks in the simulator.
- Demonstrated generalization by driving smoothly on unseen tracks.

## Future Scope
- Real-world deployment using actual car sensors.
- Integration with additional features such as object detection and traffic sign recognition.
- Enhanced decision-making algorithms for complex driving scenarios.

## Acknowledgements
**Rayan Slim** for providing the foundational knowledge through their Applied Deep Learning courses.

---
