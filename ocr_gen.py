from augmenter import Augmenter
from backgrounder import Blackgrounder
from utils.gen_text import generate_image_from_text
from utils.utils import insert_augmented_images, save_yolo_format, save_yolo_obbox_format
import os
import shutil

augmenter = Augmenter()
# augmenter.add_augmentation('rotation',angle_range=(-60,60))
backgrounder = Blackgrounder()
dict_name = 'name'
num_bg = num_augment = 18
output_path = "output"
if os.path.exists(output_path):
    shutil.rmtree(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
backgrounder.add_dict(dict_name,"background","image")
backgrounds = backgrounder.get_background([dict_name],[num_bg],[None])

augmented_images = []
oriented_bboxs = []
class_indices = []
font_indices = []
word_list = ["เป็นมนุษย์สุดประเสริฐเลิศคุณค่า กว่าบรรดาฝูงสัตว์เดรัจฉาน\nจงฝ่าฟันพัฒนาวิชาการ อย่าล้างผลาญฤๅเข่นฆ่าบีฑาใคร\nไม่ถือโทษโกรธแช่งซัดฮึดฮัดด่า หัดอภัยเหมือนกีฬาอัชฌาสัย\nปฏิบัติประพฤติกฎกำหนดใจ พูดจาให้จ๊ะๆ จ๋า น่าฟังเอยฯ","ocr_gen","hello"]
font_list = os.listdir("fonts")

for idx,word in enumerate(word_list):
    images = []
    for font in font_list:
        images.append(generate_image_from_text(word,os.path.join("fonts",font),font_size=20,text_color=(255,255,255),outline_color=(0,0,0),outline_thickness=3))
    augmenter.add_dict(word,images)
    num_class_images = num_augment // len(word_list)
    augmented_images_ ,oriented_bboxs_ = augmenter.augment(word,num_class_images)
    augmented_images.extend(augmented_images_)
    oriented_bboxs.extend(oriented_bboxs_)
    class_indices.extend([idx] * num_class_images)
    extended_font_indices = (font_list * (num_class_images // len(font_list) + 1))[:num_class_images]
    font_indices.extend(extended_font_indices)

frames_with_augmentations, bounding_boxes, oriented_bounding_boxs = insert_augmented_images(backgrounds, augmented_images, oriented_bboxs, class_indices,padding_crop=False)
save_yolo_format(frames_with_augmentations, bounding_boxes, output_path, word_list, font_indices)