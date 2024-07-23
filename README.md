# Pedestrian Auto-labeling with Feature Extraction and Behaviour Analysis

This project involves processing video files to automatically detect and analyze pedestrian behavior. It leverages a deep learning approach with computer vision, specifically using the YOLOv8 model for pose estimation. The project resulted in the creation of the **EGY-Drives dataset**, which features attributes unique worldwide and is not available in any other dataset.

## Key Attributes

- **Walking/Standing Status**: Automatically identifies if a pedestrian is walking or standing.
- **Looking/Not Looking Status**: Detects whether a pedestrian is looking or not looking.
- **Speed**: Calculates the speed at which the pedestrian is moving.
- **Direction of Motion**: Determines the direction in which the pedestrian is moving.

This deep learning-based approach ensures accurate and real-time analysis of pedestrian behavior through frame-by-frame video processing. The resulting EGY-Drives dataset provides unparalleled insights and is a significant contribution to the field of pedestrian detection and behavior analysis.

## Example Video

Here is a short video demonstrating the project in action:

[Watch the Example Video on YouTube](https://youtu.be/qBHLyzyITz4?si=keEcnC-M9YkFEZAZ).
## Table of Contents

- [Overview](#overview)
- [Key Components and Variables](#key-components-and-variables)
  - [Input and Output Folders](#input-and-output-folders)
  - [Counters and Thresholds](#counters-and-thresholds)
  - [Functions](#functions)
  - [Model Loading](#model-loading)
  - [Video Processing](#video-processing)
  - [Frame-by-Frame Processing](#frame-by-frame-processing)
  - [Annotations and Display](#annotations-and-display)
  - [Saving Results](#saving-results)
- [Main Arrays and Their Usage](#main-arrays-and-their-usage)
- [How to Run](#how-to-run)
- [Live Video Processing](#live-video-processing)

## Overview

The provided code processes videos from a specified folder, extracting frame-by-frame information about pedestrians. It detects pedestrians, tracks them across frames, and determines whether they are walking or standing, looking or not looking, their speed, and the direction of motion. The results are saved into output video files, and annotations are stored in text files.

## Key Components and Variables

### Input and Output Folders

- **input folder**: Directory containing the input video files.
- **output folder**: Directory where the processed video files will be saved.
- **output annotation dir**: Directory where annotation files will be saved.

### Counters and Thresholds

- **videos count**: Counts the total number of videos processed.
- **frame count**: Counts the total number of frames processed.
- **total pedestrians**: Total number of pedestrians detected in all frames.
- **transition counter**: Buffer counter to confirm status changes.
- **transition threshold**: Threshold for buffer time to confirm status change.
- **movement threshold**: Threshold for movement detection.
- **ped num per frame**: List to store the number of pedestrians detected per frame.

### Functions

- **calculate_distance(point1, point2)**: Calculates the Euclidean distance between two points.
- **get_direction(point1, point2)**: Determines the direction of movement based on the change in position.
- **estimate_speed(Location1, Location2)**: Estimates the speed of movement based on the distance covered between frames.
- **write_to_txt(file_name, data)**: Writes the given data to a text file, either as a dictionary or as plain text.

### Model Loading

The code imports necessary libraries and loads the YOLOv8 model for pose estimation.

### Video Processing

The code loops over each video file in the specified input folder. For each video:

- Initializes a video capture object and retrieves the video properties such as frame width, height, and FPS.
- A video writer object is created to save the output video.

### Frame-by-Frame Processing

For each frame in the video:

- The YOLO model is used to detect and track pedestrians, extracting bounding boxes and keypoints.
- The code calculates the walking/standing status, looking/not looking status, speed, and direction of each detected pedestrian.
- Maintains histories of object positions to track movement across frames.

### Annotations and Display

- Bounding boxes and annotations (walking/standing, looking/not looking, speed, direction) are drawn on the frames.
- The annotated frame is displayed in a window, and the original frame is written to the output video file.

### Saving Results

After processing all frames, the video capture and writer objects are released. Annotations are saved to text files.

## Main Arrays and Their Usage

- **boxes**: Array of bounding boxes for detected pedestrians in each frame. Used to draw rectangles around detected objects and for movement analysis.
- **keypoints_data**: Array of keypoints data for all detected persons. Contains keypoints for body parts, used to determine looking/not looking status and movement analysis.
- **object_histories**: Dictionary storing historical positions of each tracked object. Helps track the movement of objects across frames and analyze their movement.
- **prev_walking_status**: Dictionary storing the previous walking/standing status of each object. Used to apply hysteresis in status changes and smooth out the walking/standing determination.
- **directions**: Dictionary storing the direction of movement for each object. Stores the current direction of each object, used for annotations.
- **speeds**: Dictionary storing the speed of each object. Contains the estimated speed of each object, used for annotations.
- **ped_num_per_frame**: List storing the number of pedestrians detected per frame. Helps keep track of pedestrian count statistics across frames.
- **transition_counter**: Counter used for confirming status changes. Buffers the transition between statuses to avoid rapid fluctuations.
- **transition_threshold**: Threshold value for the transition counter. Determines how many frames are considered before confirming a status change.
- **movement_threshold**: Distance threshold for determining movement. Used to differentiate between walking and standing statuses based on the distance moved.
- **ids**: Array of unique IDs for each detected object. Used to track objects consistently across frames, ensuring the correct association of historical data.

## How to Run

### Prerequisites

- Python 3.8 or higher
- Required Python packages listed in `requirements.txt`

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/Zeyad-Saadawy/Pedestrian-Auto-labeling-with-feature-extraction-and-behaviour-analysis.git
    ```

2. Create a virtual environment:

    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Running the Code

1. Prepare your input videos and place them in the `input` folder.

2. Run the `Track-Merge-loop.py` script:

    ```sh
    python Track-Merge-loop.py --input_folder path/to/input --output_folder path/to/output --output_annotation_dir path/to/annotations
    ```

## Running the Live Video Processing

1. Connect your video source (e.g., webcam).

2. Run the `live.py` script:

    ```sh
    python live.py
    ```

    This will start processing the live video feed and perform pedestrian detection and behavior analysis in real-time.

This documentation provides an overview and detailed explanation of the code’s functionality, key components, and main variables. It should help in understanding how the code processes video files, tracks pedestrians, and saves the results.

**Make sure to adjust the repository URL, folder paths, and any other specific details to match your project setup.**
