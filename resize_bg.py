from PIL import Image
import os

# Set input folder and desired size
input_folder = 'C:/Jim_Storage/OCR_generator/background'  # Change to your folder path
resize_width = 600
resize_height = 600

# Resize function
def resize_image(image_path, output_size):
    with Image.open(image_path) as img:
        img = img.resize(output_size, Image.ANTIALIAS)
        img.save(image_path)  # Overwrite the image

# Process all images in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
        file_path = os.path.join(input_folder, filename)
        resize_image(file_path, (resize_width, resize_height))
        print(f'Resized {filename}')

print("All images have been resized.")
