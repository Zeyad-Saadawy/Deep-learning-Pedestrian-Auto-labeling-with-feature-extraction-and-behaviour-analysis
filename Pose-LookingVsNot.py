import cv2
import numpy as np
import cvzone

# Load YOLOv8 model for pose estimation
from ultralytics import YOLO
model = YOLO('Models\yolov8m-pose.pt') 

# Open the video file
cap = cv2.VideoCapture(r'Media\1cross1stand.mp4')

# Get the original dimensions of the video
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(r"C:\Users\zeyad\OneDrive\Desktop\Demo Results\Week8/1cross1stand-Lookingmodel.mp4", fourcc, 30, (original_width, original_height))

# Define a function to calculate the distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
frame_count = 0
while True:
    success, frame = cap.read()
    
    if not success:
        break
    
    results = model.predict(frame, save=True)
    frame_count += 1
    # Get the bounding box information in xyxy format
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    Looking_statuses = []

    # Get the keypoints data for all detected persons
    keypoints_data = results[0].keypoints.data

    # Iterate through the detected persons
    for i, keypoints in enumerate(keypoints_data):
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
                eye_distance = calculate_distance(left_eye, right_eye)
                if eye_distance < 5:
                    looking_status = 'Not Looking'
                else:
                    looking_status = 'Looking'
            Looking_statuses.append(looking_status)

    # Draw bounding boxes and statuses on the frame
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        if Looking_statuses[i] == 'Not Looking':
            colorR = (0, 191, 255)  # Yellow color
            colorT = (0, 0, 0)  # Black color
        else:
            colorR = (255, 111, 111)  # Light blue color
            colorT = (255, 255, 255)  # White color

        cvzone.putTextRect(
            frame, f"{Looking_statuses[i]}", (x1, y1 - 10),  # Image and starting position of the rectangle
            scale=3, thickness=3,  # Font scale and thickness
            colorT=colorT, colorR=colorR,  # Text color and Rectangle color
            font=cv2.FONT_HERSHEY_PLAIN,  # Font type
            offset=10,  # Offset of text inside the rectangle
            border=0, colorB=(255, 255, 255)  # Border thickness and color
        )

    # Write the frame to the output video file
    out.write(frame)
print("Frame count: ",frame_count)
# Release the video capture device, close the output video file, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
