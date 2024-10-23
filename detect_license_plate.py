from ultralytics import YOLO

# Load the model with the saved weights
model = YOLO("D:/Documents/Projects/YOLO/runs/detect/train12/weights/best.pt")


for imgae_no in range(901, 1111):
    image_path=f"D:/Documents/Projects/YOLO/test/{imgae_no}.jpg"
# image_path="D:/Documents/Projects/YOLO/test/904.jpg"

# Use the model for inference or further training
    results = model.predict(source=image_path, save=True)
