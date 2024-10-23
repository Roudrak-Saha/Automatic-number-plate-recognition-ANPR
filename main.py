from PIL import Image
import pandas as pd
import os

# Load CSV file containing bounding box coordinates
bounding_boxes_df = pd.read_csv("Licplatesdetection_train.csv")

# Load image
for i in range(0,900):  
    filename=bounding_boxes_df['img_id'][i]
    filename=filename[:-4]
    # Extract bounding box coordinates from the CSV file
    # For example, assuming the CSV file has columns 'x_min', 'y_min', 'x_max', 'y_max'
    x_min = bounding_boxes_df['xmin'][i]
    y_min = bounding_boxes_df['ymin'][i]
    x_max = bounding_boxes_df['xmax'][i]
    y_max = bounding_boxes_df['ymax'][i]

    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    width = x_max - x_min
    height = y_max - y_min

    image=Image.open(f"final_data/images/{filename}.jpg")
    image_width, image_height = image.size

    normalized_x_center = x_center / image_width
    normalized_y_center = y_center / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height

    output_text_path = os.path.join("final_data/labels", f"{filename}.txt")
    text_content = f"0 {normalized_x_center} {normalized_y_center} {normalized_width} {normalized_height}"

    # Save text file
    with open(f"final_data/labels/{filename}.txt", "w") as text_file:
        text_file.write(text_content)