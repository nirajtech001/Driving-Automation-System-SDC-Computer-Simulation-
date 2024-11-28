
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

### **My Contributions**
- Led the efforts in data preprocessing, building the CNN network, and integrating the model with the simulator.

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

## Dataset Collection & Preprocessing
The dataset consists of images captured from three camera views (left, center, right) and corresponding steering angles. Images undergo augmentation, including:
![image](https://github.com/user-attachments/assets/90721280-69d9-4333-93ed-9dff05f07a69)
![image](https://github.com/user-attachments/assets/eca796cb-f051-4cb5-a8ba-87ad3da1aa61)
### Data Preprocessing and Augmentation
- Brightness adjustments.
- Horizontal flips.
- Cropping irrelevant parts.
![image](https://github.com/user-attachments/assets/3dd44b17-bdf6-4f5e-94ca-16fed647b463)


## Model Architecture
The CNN model is based on NVIDIA's architecture for autonomous driving:
![image](https://github.com/user-attachments/assets/a68262d7-410d-4bc1-9e04-9a06ea8a4f4c)

- Input layer processes 66x200 RGB images converted to YUV color space.
- Convolutional and pooling layers extract features.
- Fully connected layers output the steering angle.

## Results
- Successfully navigated through multiple tracks in the simulator.
- Demonstrated generalization by driving smoothly on unseen tracks.
  ![image](https://github.com/user-attachments/assets/ed94686d-777b-474c-b87d-28dd2af2b0e5)


## Future Scope
- Real-world deployment using actual car sensors.
- Integration with additional features such as object detection and traffic sign recognition.
- Enhanced decision-making algorithms for complex driving scenarios.

## Acknowledgements
-**Rayan Slim** for providing the foundational knowledge through their applied deep learning courses.


---
------------------------------------------------------------------
Prerequisites:- 
Manual Installation Language & Libreries - Python 3.8, Udacity Simulator Setup, Unity3D Engine, Tensorflow, Keras, OpenCV, Flask, NumPy, Socket.io, Eventlet

About Model Training File
Model_training_source_code file in Google Colab or Jupyter Notebook
Dataset Used in this model from github link https://github.com/nirajtech001/Driving-Automation-System-SDC-Computer-Simulation-/tree/main/IMG

### Detailed Steps to Run the SImulator

Step 1- in Command Prompt open project source file
Step 2- Setup the Virtual Environment 
Step 3- Open Virtual Environment using command 'activate myenviron' 
Step 4- In CMD Open drive file using command '[path] python drive.py'
Step 4- Click on the Simulator 'Default Windows desktop 64-bit' 
Step 5- Click on autonomous mode button in the simulator
Step 6- Wait for Connect the Model to the Simulator
Step 7- Simulator connected sucessfully, now watch and experience driving automation technology.


