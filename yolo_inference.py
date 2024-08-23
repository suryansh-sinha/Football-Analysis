from ultralytics import YOLO

# Loading YOLO
model = YOLO('models/best.pt')

# Inference
results = model.predict('input_videos/08fd33_4.mp4', save=True)
print(results[0])   # Saving the result for the first frame.
print('===================================')
for box in results[0].boxes:
    print(box)
