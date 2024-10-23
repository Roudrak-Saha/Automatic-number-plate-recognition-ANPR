import cv2
import pandas as pd
import os

# Load CSV file containing bounding box coordinates
bounding_boxes_df = pd.read_csv("Licplatesdetection_train.csv")

# Load image
for i in range(0,901):  
    filename=bounding_boxes_df['img_id'][i]
    
    # Extract bounding box coordinates from the CSV file
    # For example, assuming the CSV file has columns 'x_min', 'y_min', 'x_max', 'y_max'
    x_min = bounding_boxes_df['xmin'][i]
    y_min = bounding_boxes_df['ymin'][i]
    x_max = bounding_boxes_df['xmax'][i]
    y_max = bounding_boxes_df['ymax'][i]

    image_path = f"license_plates_detection_train/{filename}"  # Path to your single car image
    image = cv2.imread(image_path)

    # Draw bounding box on the image
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Extract the filename from the image path
    #filename = os.path.basename(image_path)  # Get filename with extension

    # Create the output image path with the same filename in the output folder
    output_path = os.path.join("train_final", filename)

    # Save the final image with bounding box
    cv2.imwrite(output_path, image)
'''
# Display the image with bounding box
cv2.imshow("Image with Bounding Box", image)
cv2.waitKey(0)
cv2.destroyAllWindows()'''