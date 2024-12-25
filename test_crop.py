import os
import cv2
from tqdm import tqdm
import shutil

# Paths to your YOLO project folders
yolo_folder = 'output_test'
images_folder = f'{yolo_folder}/images'
labels_folder = f'{yolo_folder}/labels'
output_folder = 'test_crop'
classes_file = f'{yolo_folder}/classes.txt'
output_label_file = os.path.join(output_folder, 'label.txt')

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Load class names
with open(classes_file, 'r', encoding='utf-8') as f:
    class_names = f.read().strip().split('\n')

label_entries = []

# Iterate through label files
for label_file in tqdm(os.listdir(labels_folder)):
    if not label_file.endswith('.txt'):
        continue
    
    # Corresponding image file
    image_file = os.path.join(images_folder, label_file.replace('.txt', '.jpg'))
    
    if not os.path.exists(image_file):
        continue

    # Read image
    image = cv2.imread(image_file)
    h, w, _ = image.shape
    
    # Read label file
    with open(os.path.join(labels_folder, label_file), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        cls, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())
        
        # Convert YOLO coordinates to pixel coordinates
        x_center, y_center = x_center * w, y_center * h
        bbox_width, bbox_height = bbox_width * w, bbox_height * h
        
        x1 = int(x_center - bbox_width / 2)
        y1 = int(y_center - bbox_height / 2)
        x2 = int(x_center + bbox_width / 2)
        y2 = int(y_center + bbox_height / 2)
        
        # Crop image
        cropped_image = image[y1:y2, x1:x2]
        
        # Save cropped image
        class_name = class_names[int(cls)]
        save_name = f'{label_file.replace(".txt", "")}_{i}.jpg'
        save_path = os.path.join(output_folder, save_name)
        cv2.imwrite(save_path, cropped_image)
        
        # Add entry to label file
        label_entries.append(f'{save_name}\t{class_name}')

# Write label entries to file
with open(output_label_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(label_entries))
