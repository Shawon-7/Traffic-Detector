from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.track(source='traffic_vid.mp4', show=True, save=True, tracker='bytetrack.yaml')
