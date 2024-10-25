from PIL import Image, ImageDraw, ImageFont

def generate_image_from_text(text, font_path, font_size, text_color, outline_color=None, outline_thickness=0):
    # Load the font and set the font size
    font = ImageFont.truetype(font_path, font_size)
    
    # Create a temporary image to calculate text size
    temp_image = Image.new("RGBA", (1, 1))
    temp_draw = ImageDraw.Draw(temp_image)
    
    # Calculate text bounding box
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0] + 2 * outline_thickness
    text_height = bbox[3] - bbox[1] + 2 * outline_thickness
    
    # Create a new image with padding for the outline and transparent background
    image = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Draw the text outline first (if outline color and thickness are specified)
    if outline_color and outline_thickness > 0:
        for x_offset in range(-outline_thickness, outline_thickness + 1):
            for y_offset in range(-outline_thickness, outline_thickness + 1):
                if x_offset == 0 and y_offset == 0:
                    continue  # Skip the center (main text)
                draw.text((outline_thickness - bbox[0] + x_offset, 
                           outline_thickness - bbox[1] + y_offset), 
                          text, font=font, fill=outline_color)
    
    # Draw the main text on top of the outline
    draw.text((outline_thickness - bbox[0], outline_thickness - bbox[1]), text, fill=text_color, font=font)

    return image