from ultralytics import YOLO
import easyocr
import cv2

# Load the model with the saved weights
model = YOLO("D:/Documents/Projects/YOLO/runs/detect/train12/weights/best.pt")

reader = easyocr.Reader(['ar'])

for imgae_no in range(901, 1111):
    image_path=f"D:/Documents/Projects/YOLO/test/{imgae_no}.jpg"
    # image_path="D:/Documents/Projects/YOLO/test/904.jpg"

    # Use the model for inference or further training
    results = model.predict(source=image_path, save=True)

    # Load the image using OpenCV
    img = cv2.imread(image_path)

    for result in results[0].boxes:
    # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, result.xyxy[0])
    
    # Crop the detected license plate from the image
        plate_img = img[y1:y2, x1:x2]

    # Use EasyOCR to extract text from the license plate
        ocr_result = reader.readtext(plate_img)
    
    # Extract and print the detected license plate text
        plate_text = ocr_result[0][-2] if ocr_result else ''
        print("Detected License Plate Text:", plate_text)

    # Optionally, draw the bounding box and the detected text on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Save or display the final image with the detected license plate and text
    output_image_path = "D:/Documents/Projects/YOLO/test_images/sample_with_easyocr_text.jpg"
    cv2.imwrite(output_image_path, img)