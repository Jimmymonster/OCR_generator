import os
import json
from glob import glob

# Paths to YOLO dataset
yolo_path = 'output_test'
images_folder = f'{yolo_path}/images'
labels_folder = f'{yolo_path}/labels'
classes_file = f'{yolo_path}/classes.txt'
output_file = 'output.txt'

# Load class names
with open(classes_file, 'r', encoding='utf-8') as f:
    classes = f.read().strip().split('\n')

# Get image dimensions
def get_image_size(image_path):
    from PIL import Image
    with Image.open(image_path) as img:
        return img.size  # (width, height)

# Convert YOLO bbox to quadrilateral points
def yolo_to_points(x, y, w, h, img_w, img_h):
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y - h / 2) * img_h
    x3 = (x + w / 2) * img_w
    y3 = (y + h / 2) * img_h
    x4 = (x - w / 2) * img_w
    y4 = (y + h / 2) * img_h
    return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

# Process labels
lines = []
for label_path in glob(f'{labels_folder}/*.txt'):
    image_name = os.path.basename(label_path).replace('.txt', '.jpg')  # Assuming .jpg images
    image_path = os.path.join(images_folder, image_name)

    if not os.path.exists(image_path):
        continue

    img_w, img_h = get_image_size(image_path)

    with open(label_path, 'r') as f:
        annotations = []
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])

            points = yolo_to_points(x, y, w, h, img_w, img_h)
            transcription = classes[class_id]

            annotations.append({
                "points": points,
                "transcription": transcription
            })

        lines.append(f'{image_path}\t{json.dumps(annotations, ensure_ascii=False)}')

# Write to output file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f'Transformation complete. Results saved to {output_file}')
