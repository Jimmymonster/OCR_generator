from augmenter import Augmenter
from backgrounder import Blackgrounder
from utils.gen_text import generate_image_from_text
from utils.utils import insert_augmented_images, save_yolo_format, save_yolo_obbox_format
import os

augmenter = Augmenter()
augmenter.add_augmentation('rotation',angle_range=(-60,60))
backgrounder = Blackgrounder()
dict_name = 'name'
num_bg = num_augment = 20
backgrounder.add_dict(dict_name,"background","image")
backgrounds = backgrounder.get_background([dict_name],[num_bg],[None])

images = []
word_list = ["สวัสดี","ocr_gen","hello"]
font_list = os.listdir("fonts")

for word in word_list:
    for font in font_list:
        images.append(generate_image_from_text(word,os.path.join("fonts",font),font_size=100,text_color=(255,255,255),outline_color=(0,0,0),outline_thickness=3))
class_name="text"
augmenter.add_dict(class_name,images)

augmented_images ,oriented_bboxs = augmenter.augment(class_name,num_augment)

frames_with_augmentations, bounding_boxes, oriented_bounding_boxs = insert_augmented_images(backgrounds, augmented_images, oriented_bboxs, [0],padding_crop=False)
save_yolo_format(frames_with_augmentations, bounding_boxes, "output", ["text"])