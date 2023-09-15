# Action Recognition in Videos

This script identifies the start and end of a round in boxing video footage.

## Key Features:
- **Video Processing**: Load a video and process it frame by frame.
- **Action Recognition**: Utilizes the R(2+1)D-18 model, a 3D ResNet model pretrained on the Kinetics-400 dataset.
- **Frame Transformation**: Applies image transformations (resize, center crop, and normalization) using the `albumentations` library.
- **Output Video**: Generates an output video with frames labeled with the detected action.

- ![Action Recognition Demonstration](./assets/joyce_gif.gif)


## Requirements:
- Python 3.x
- torch
- torchvision
- cv2 (OpenCV)
- numpy
- tqdm
- albumentations

## Usage:

1. **Setup**:
    ```bash
    pip install torch torchvision opencv-python-headless numpy tqdm albumentations
    ```

## Directory Structure:
- **input/**: This directory should contain the video files you want to process.
- **outputs/**: The processed videos will be saved in this directory with the action label superimposed on the frames.

2. **Run the Script**:
    - By default, the script processes the video "input/video.mp4". 
    ```bash
    python action_recognition.py
    `

3. **Output**:
    - The processed video will be saved in the "outputs" directory with the action label superimposed on the frames.

## Customization:
- To process a different video, change the `input_video` variable in the main function to the desired video path.

## Future Work:
- Enhance the script to provide timestamps indicating intervals where no boxing activity is detected for a specified duration.
