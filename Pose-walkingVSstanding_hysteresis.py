import cv2
import numpy as np
import cvzone

# Load YOLOv8 model for pose estimation
from ultralytics import YOLO
model = YOLO('Models\yolov8m-pose.pt') 

# Open the video file
cap = cv2.VideoCapture(r"C:\Users\zeyad\OneDrive\Desktop\Yolov8\Media\uni4.MOV")

# Get the original dimensions of the video
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(r"C:\Users\zeyad\OneDrive\Desktop\MovingEval-PoseEstimation\uni-model.mp4", fourcc, 30, (original_width, original_height))

# Define a function to calculate the distance between two points
def calculate_distance(point1, point2):
    
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Initialize variables for hysteresis
prev_status = 'Not identified'
transition_counter = 0
transition_threshold = 3  # Adjust as needed

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    results = model.predict(frame, save=True)

    # Get the bounding box information in xyxy format
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    statuses = []

    # Get the keypoints data for all detected persons
    keypoints_data = results[0].keypoints.data
    # Iterate through the detected persons
    for i, keypoints in enumerate(keypoints_data):
        # Ensure keypoints are detected
        if keypoints.shape[0] > 0:
            # Calculate the left and right ankles keypoints
            left_ankle = keypoints[15][:2]
            right_ankle = keypoints[16][:2]
            # Check if the ankle keypoints are not detected
            if (keypoints[15][0].cpu().numpy().astype(int) ==0 & keypoints[15][1].cpu().numpy().astype(int) ==0) | \
                  (keypoints[16][0].cpu().numpy().astype(int) ==0 & keypoints[16][1].cpu().numpy().astype(int) ==0):
                status_to_display = 'Not identified'
            # Check if the keypoints are not clearly detected
            elif keypoints[15][2].cpu() <0.2 and keypoints[16][2].cpu() <0.2:
                status_to_display = 'Not identified'
            else:
                ankle_distance = calculate_distance(left_ankle, right_ankle)

                # Determine if the person is walking based on ankle distance
                if ankle_distance > 25:  # Adjust threshold as needed
                    status = 'Walking'
                else:
                    status = 'Standing'

                # Apply hysteresis to smooth transitions
                if status != prev_status:
                    transition_counter += 1
                    if transition_counter >= transition_threshold:
                        prev_status = status
                        transition_counter = 0
                else:
                    transition_counter = 0
                
            # Update the status to be displayed on the frame
            status_to_display = prev_status
            statuses.append(status_to_display)

    # Draw bounding boxes and statuses on the frame
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cvzone.putTextRect(
            frame, f"{statuses[i]}", (x1, y2 - 10),  # Image and starting position of the rectangle
            scale=3, thickness=3,  # Font scale and thickness
            colorT=(255, 255, 255), colorR=(255, 0, 255),  # Text color and Rectangle color
            font=cv2.FONT_HERSHEY_PLAIN,  # Font type
            offset=10,  # Offset of text inside the rectangle
            border=0, colorB=(0, 255, 0)  # Border thickness and color
        )

    # Write the frame to the output video file
    out.write(frame)

    # Exit the program if the user presses the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture device, close the output video file, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
