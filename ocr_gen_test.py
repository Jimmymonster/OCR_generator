# -*- encoding: utf-8 -*-

from augmenter import Augmenter
from backgrounder import Blackgrounder
from utils.gen_text import generate_image_from_text
from utils.utils import insert_augmented_images, save
from utils.gen_random import generate_random_strings
import os
import shutil
import random

from trdg.generators import ( GeneratorFromDict )

augmenter = Augmenter()
# augmenter.add_augmentation('rotation',angle_range=(-60,60))
backgrounder = Blackgrounder()
dict_name = 'name'

output_path = "output_test"
if os.path.exists(output_path):
    shutil.rmtree(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
backgrounder.add_dict(dict_name,"background","image")
# backgrounder.add_dict(dict_name,"video/video_TNN.mp4","video")
# backgrounder.add_rgb_bg_dict(dict_name,(255,255,255),1000,300)

augmented_images = []
oriented_bboxs = []
class_indices = []
font_indices = []
word_list = []
# word_list.extend(["เป็นมนุษย์","สุดประเสริฐ","เลิศคุณค่า","กว่าบรรดา","ฝูงสัตว์เดรัจฉาน","จงฝ่าฟัน","พัฒนาวิชาการ","อย่าล้างผลาญฤๅ","เข่นฆ่าบีฑาใคร","ไม่ถือโทษโกรธ","แช่งซัดฮึดฮัดด่า","หัดอภัย","เหมือนกีฬา","อัชฌาสัย","ปฏิบัติ","ประพฤติกฎ","กำหนดใจ","พูดจาให้จ๊ะๆ จ๋า","น่าฟังเอยฯ"])
# word_list.extend(["The","quick","brown","fox","jumps","over","the","lazy","dog"])
# word_list.extend(generate_random_strings("thai",10,20))
# word_list.extend(["ยาย","กิน","ลำไย","น้ำลาย","ยาย","ไหล","ย้อย","ชาม","เขียว","คว่ำ","เช้า","วันพฤหัส","กตัญญูกตเวที","ทฤษฏี"])
# word_list.extend(generate_random_strings("english",7,14))
font_list = os.listdir("fonts")
generator = GeneratorFromDict(language="th",
                        fonts=["fonts/THSarabun.ttf"],
                        count=3000)
for img, lbl in generator:
    word_list.append(lbl)
# font_list = ["DSN ThaiRat Regular.ttf"]

# num_bg = num_augment = len(word_list)*len(font_list)
num_bg = num_augment = 3000

backgrounds = backgrounder.get_background([dict_name],[num_bg],[None])
images = []
for idx,word in enumerate(word_list):
    images.append(generate_image_from_text(word,os.path.join("fonts",random.choice(font_list)),font_size=30,text_color=(255,255,255),outline_color=(0,0,0),outline_thickness=1))
augmenter.add_dict("dict1",images)
num_class_images = num_augment
augmented_images_ ,oriented_bboxs_ = augmenter.augment("dict1",num_class_images)
augmented_images.extend(augmented_images_)
oriented_bboxs.extend(oriented_bboxs_)
class_indices.extend(i for i in range(num_augment))
extended_font_indices = (font_list * (num_class_images // len(font_list) + 1))[:num_class_images]
font_indices.extend(extended_font_indices)

frames_with_augmentations, bounding_boxes, oriented_bounding_boxs = insert_augmented_images(backgrounds, augmented_images, oriented_bboxs, class_indices,padding_crop=False)
save(frames_with_augmentations, bounding_boxes, output_path, word_list, font_indices)