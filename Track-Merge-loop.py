import math
import os
import cv2
import numpy as np
import cvzone

# Load YOLOv8 model for pose estimation
from ultralytics import YOLO
model = YOLO('Models/yolov8x-pose-p6.pt')
 

# Specify the folder containing the input videos
input_folder = r"C:\Users\zeyad\OneDrive\Desktop"

# Specify the folder where you want to save the output videos
output_folder = r"C:\Users\zeyad\OneDrive\Desktop\Yolov8\Output-vods"

# Define the output annotation directory
output_annotation_dir = r"C:\Users\zeyad\OneDrive\Desktop\Yolov8\Annotations"
videos_count = 0 # counting total videos processed
frame_count = 0 # counting total frames for naming the output files
total_pedestrians = 0 # Total number of pedestrians detected in all frames
transition_counter = 0
transition_threshold = 3  # Threshold for buffer time to confirm status change
movement_threshold = 10  # Threshold for movement detection



def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def get_direction(point1, point2):
    direction_str = ""
    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""

    return direction_str

def estimatespeed(Location1, Location2):
    #Euclidean Distance Formula
    d_pixel = calculate_distance(Location1, Location2)
    # defining thr pixels per meter
    ppm = 220
    d_meters = d_pixel/ppm
    time_constant = 15*3.6
    #distance = speed/time
    speed = d_meters * time_constant

    return int(speed)

# Function to write data to text file
def write_to_txt(file_name, data):
    with open(file_name, 'w') as f:
            # check if data is a dictionary
        if isinstance(data, dict):
            for i in range(len(data)):
                f.write(data[i] + '\n')  # Write each pedestrian's data to a new line
        else:
            f.write(data)

# Iterate over all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".mp4") or filename.endswith(".avi") or filename.endswith(".MOV"):
        videos_count += 1
        video_path = os.path.join(input_folder, filename)
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Get the original dimensions of the video
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create a VideoWriter object to save the output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(output_folder, filename)  # Generate output video file path
        out = cv2.VideoWriter(output_path, fourcc, 30, (original_width, original_height))
        ############################################################################################################
        prev_walking_status = {}  # Dictionary to store previous status for each object ID
        Directions = {}  # Dictionary to store the direction of each object ID
        Speeds = {}  # Dictionary to store the speed of each object ID
        # Object history dictionary
        object_histories = {}

       # Process each frame in the video
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            
            print(f"Processing frame {frame_count} in video {filename} number {videos_count}...")
            frame_count += 1
            results = model.track(frame, save=True, persist=True, tracker='bytetrack.yaml')

            # Get boxes and IDs
            boxes = results[0].boxes.xywh.cpu().numpy()
            boxesxyxy = results[0].boxes.xyxy.cpu().numpy()
            try:
                ids = results[0].boxes.id.cpu().numpy()
            except AttributeError:
                ids = np.arange(len(boxes))

            # Get the keypoints data for all detected persons
            keypoints_data = results[0].keypoints.data

            walking_statueses = []
            Looking_statuses = []
            direction="No Direction"
            speed=0
            # annotation Data for each frame
            data = {}

            # Process detected objects
            for i, ((x1, y1, x2, y2) , keypoints) in  enumerate(zip(boxesxyxy , keypoints_data)):
                object_id = ids[i]
                # part 1 = 'Walking vs Standing'
                # Extract center coordinates (using xywh format directly)
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Update object history
                if object_id not in object_histories:
                    object_histories[object_id] = []
                object_histories[object_id].append((center_x, center_y))

                # Manage object history length (CORRECT PLACEMENT)
                if len(object_histories[object_id]) > 10:  # Adjust as needed
                    object_histories[object_id].pop(0)  # Remove oldest entry

                # Movement analysis (alternative methods)
                if len(object_histories[object_id]) > 1:
                    prev_x, prev_y = object_histories[object_id][0]
                    curr_x, curr_y = object_histories[object_id][-1]
                    distance = calculate_distance((curr_x, curr_y), (prev_x, prev_y))
                    if distance > movement_threshold:
                        walking_statues = "Walking"
                    else:
                        walking_statues = "Standing"
                else:
                    walking_statues = "Not Identified"

                # Apply hysteresis (check previous status for same object ID)
                if object_id in prev_walking_status:
                    if prev_walking_status[object_id] != walking_statues:
                        transition_counter += 1
                        if transition_counter >= transition_threshold:
                            prev_walking_status[object_id] = walking_statues
                            transition_counter = 0
                else:
                    prev_walking_status[object_id] = walking_statues

                walking_statueses.append((object_id, prev_walking_status[object_id]))
                if walking_statues == "Walking":
                    direction = get_direction((prev_x, prev_y), (curr_x, curr_y))
                    Directions[object_id] = direction
                    speed = estimatespeed((prev_x, prev_y), (curr_x, curr_y))
                    Speeds[object_id] = speed

                else:
                    Directions[object_id] = ""
                    Speeds[object_id] = 0
                # part 2 = 'Looking vs Not Looking'
                # Ensure keypoints are detected
                if keypoints.shape[0] > 0:
                    # Get the left and right eye keypoints
                    left_eye = keypoints[1][:2]
                    right_eye = keypoints[2][:2]
                    # Check if the keypoints are not detected
                    if keypoints[1][2].cpu() <0.3 and keypoints[2][2].cpu() <0.3:
                        looking_status = 'Not Identified'
                    if (keypoints[1][0].cpu().numpy().astype(int) ==0 | keypoints[1][1].cpu().numpy().astype(int) ==0) | \
                        (keypoints[2][0].cpu().numpy().astype(int) ==0 | keypoints[2][1].cpu().numpy().astype(int) ==0):
                        looking_status = 'Not Looking'
                    else:
                        eye_distance = calculate_distance(left_eye.cpu(), right_eye.cpu())
                        if eye_distance < 5:
                            looking_status = 'Not Looking'
                        else:
                            looking_status = 'Looking'
                    Looking_statuses.append(looking_status)

                # Store data in dictionary
                data[i] = f"pedestrian {i}: {x1} , {y1} , {x2} , {y2} , {walking_statues} , {looking_status} , {direction} , {speed} km/h"
                total_pedestrians += 1

            #append to data the total data length at a new index
            data[len(data)] = f"Total Pedestrians: {len(data)}"

            # Write data to a text file
            annotationfilename = os.path.join(output_annotation_dir, f"{frame_count:06d}.txt") # Generate filename with 4-digit padded index
            print(f"Writing annotation data to {annotationfilename}...")
            os.makedirs(output_annotation_dir, exist_ok=True)
            write_to_txt(annotationfilename, data)

            # Draw bounding boxes and statuses
            for i, (box, (object_id, walking_statues)) in enumerate(zip(boxesxyxy, walking_statueses)):
                x1, y1, x2, y2  = box.astype(int)
                direction = Directions[object_id] if object_id in Directions else ""
                speed = Speeds[object_id] if object_id in Speeds else 0
                # distance = Distances[object_id] if object_id in Distances else 0
                if Looking_statuses[i] == 'Not Looking':
                    colorR = (0, 191, 255)  # Yellow color
                    colorT = (0, 0, 0)  # Black color
                else:
                    colorR = (255, 111, 111)  # Light blue color
                    colorT = (255, 255, 255)  # White color
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cvzone.putTextRect(
                    frame, f"{direction} {speed}Km/h", (x1, y2 + 35),
                    scale=1, thickness=1,
                    colorT=(255, 255, 255), colorR=(255, 111, 111),
                    offset=10, border=0, colorB=(255, 111, 111)
                )
                cvzone.putTextRect(
                    frame, f"{walking_statues}", (x1, y2 + 10),
                    scale=1, thickness=1,
                    colorT=(255, 255, 255), colorR=(255, 111, 111),
                    offset=10, border=0, colorB=(255, 111, 111)
                )

                cvzone.putTextRect(
                frame, f"{Looking_statuses[i]}", (x1, y1 - 10),  # Image and starting position of the rectangle
                scale=1, thickness=1,  # Font scale and thickness
                colorT=colorT, colorR=colorR,  # Text color and Rectangle color
                offset=10,  # Offset of text inside the rectangle
                border=0, colorB=colorR  # Border thickness and color
            )

            # Write the frame to the output video file
            out.write(frame)
            
        ############################################################################################################
        # Release the video capture device
        cap.release()

        # Release the output video writer
        out.release()

filename = os.path.join(output_annotation_dir, f"total#ped.txt") # Generate filename with 4-digit padded index
write_to_txt(filename, f" total#ped: {total_pedestrians} in {videos_count} videos")
