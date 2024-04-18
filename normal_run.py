from ultralytics import YOLO
model = YOLO("Models\yolov8m-pose.pt")  
results = model.track(source =r"Media\uni4.MOV" , conf=0.3 , save=True , persist=True , tracker='bytetrack.yaml')
print(results)


