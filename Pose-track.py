import math
import cv2
import numpy as np
import cvzone

# Load YOLOv8 model for pose estimation
from ultralytics import YOLO

model = YOLO('Models\yolov8m-pose.pt')

# Open the video file
cap = cv2.VideoCapture(r'Media\2crossagainst.mp4')

# Get the original dimensions of the video
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(r"C:\Users\zeyad\OneDrive\Desktop\Demo Results\Week8/2crossagainst-speed-TrackV1.mp4", fourcc, 30, (original_width, original_height))

# Initialize variables for hysteresis
prev_status = {}  # Dictionary to store previous status for each object ID
Directions = {}  # Dictionary to store the direction of each object ID
Speeds = {}  # Dictionary to store the speed of each object ID
transition_counter = 0
transition_threshold = 3  # Adjust as needed

# Object history dictionary
object_histories = {}

# Define movement threshold
movement_threshold = 10  # Adjust as needed

def get_direction(point1, point2):
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""

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
    d_pixel = math.sqrt(math.pow(Location2[0] - Location1[0], 2) + math.pow(Location2[1] - Location1[1], 2))
    # defining thr pixels per meter
    ppm = 160
    d_meters = d_pixel/ppm
    time_constant = 15*3.6
    #distance = speed/time
    speed = d_meters * time_constant

    return int(speed)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(frame, save=True, persist=True, tracker='bytetrack.yaml')

    # Get boxes and IDs
    boxes = results[0].boxes.xywh.cpu().numpy()
    boxesxyxy = results[0].boxes.xyxy.cpu().numpy()
    try:
        ids = results[0].boxes.id.cpu().numpy()
    except AttributeError:
        ids = np.arange(len(boxes))


    statuses = []

    # Process detected objects
    for i, (cx, cy, w, h) in enumerate(boxes):
        object_id = ids[i]

        # Extract center coordinates (using xywh format directly)
        center_x = int(cx)
        center_y = int(cy)
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

            # Option 1: Euclidean distance formula
            distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)

            # Option 2: Manhattan distance (city block distance)
            # distance = abs(curr_x - prev_x) + abs(curr_y - prev_y)

            if distance > movement_threshold:
                status = "Walking"
            else:
                status = "Standing"
        else:
            status = "Not Identified"

        # Apply hysteresis (check previous status for same object ID)
        if object_id in prev_status:
            if prev_status[object_id] != status:
                transition_counter += 1
                if transition_counter >= transition_threshold:
                    prev_status[object_id] = status
                    transition_counter = 0
        else:
            prev_status[object_id] = status

        statuses.append((object_id, status))
        if status == "Walking":
            direction = get_direction((prev_x, prev_y), (curr_x, curr_y))
            Directions[object_id] = direction
            speed = estimatespeed((prev_x, prev_y), (curr_x, curr_y))
            Speeds[object_id] = speed

        else:
            Directions[object_id] = ""
            Speeds[object_id] = 0
            

    # Draw bounding boxes and statuses
    for i, (box, (object_id, status)) in enumerate(zip(boxesxyxy, statuses)):
        x1, y1, x2, y2  = box.astype(int)
        direction = Directions[object_id] if object_id in Directions else ""
        speed = Speeds[object_id] if object_id in Speeds else 0
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cvzone.putTextRect(
            frame, f"{int(object_id)}: {status} {direction} {speed}", (x1, y1 - 10),
            scale=2, thickness=2,
            colorT=(255, 255, 255), colorR=(255, 0, 255),
            font=cv2.FONT_HERSHEY_PLAIN,
            offset=10, border=0, colorB=(0, 255, 0)
        )

    # Write the frame to the output video file
    out.write(frame)

