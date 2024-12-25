from PIL import Image, ImageDraw, ImageFont

# Parameters
image_width = 4310
image_height = 200
background_color = "white"
text = 'นายสังฆภัณฑ์ เฮงพิทักษ์ฝั่ง ผู้เฒ่าซึ่งมีอาชีพเป็นฅนขายฃวด คำ น้ำ ถูกตำรวจปฏิบัติการจับฟ้องศาล ฐานลักนาฬิกาคุณหญิงฉัตรชฎา ฌานสมาธิ'
text_color = "black"
font_size = 64
# font_folder = "C:/Jim_Storage/OCR_generator/fonts"
# font_paths = [str(file) for file in font_folder.rglob("*")]
font_path = "C:/Jim_Storage/OCR_generator/fonts/ANGSA.ttf"

# Create an image with white background
image = Image.new("RGB", (image_width, image_height), color=background_color)

# Initialize drawing context
draw = ImageDraw.Draw(image)

# Load font
try:
    font = ImageFont.truetype(font_path, font_size)
except IOError:
    font = ImageFont.load_default()

# Get text size
text_width, text_height = draw.textsize(text, font=font)

# Calculate text position (centered)
text_x = (image_width - text_width) // 2
text_y = (image_height - text_height) // 2

# Add text to image
draw.text((text_x, text_y), text, fill=text_color, font=font)

image.save("test.jpg")