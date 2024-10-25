import numpy as np
import os
import cv2
import random
from PIL import Image
# from ultralytics import YOLO

class Blackgrounder:
    # Note that every image is np array not PIL like augmenter
    def __init__(self, background_dict = None):
        # background dict contain list of folder path or video path
        self.background_dict = background_dict if background_dict is not None else {} 
        self.background_edit_settings_dict = {}
    def get_background(self, background_dict_list, num_bg_list, background_edit_setting_list):
        # iterate through background dict list to extract video to image and image according to numbg list, if num bg is higher than all possible extract image raise error
        all_backgrounds = []
        # check before do anything
        for i, dict_name in enumerate(background_dict_list):
            if dict_name not in self.background_dict:
                raise ValueError(f"Dictionary '{dict_name}' not found in background_dict.")
            if background_edit_setting_list[i] is not None and background_edit_setting_list[i] not in self.background_edit_settings_dict:
                raise ValueError(f"Dictionary '{dict_name}' not found in background_edit_settings_dict.")
            total_images=0
            for j,(type,path,max_frames) in enumerate(self.background_dict[dict_name]):
                if(type == "rgb"):
                    if max_frames is not None:
                        self.background_dict[dict_name][j][2] = max_frames # -1 mean infinity
                        total_images+=max_frames
                    else:
                        self.background_dict[dict_name][j][2] = -1 # -1 mean infinity
                        total_images+=num_bg_list[i]
                elif(type == "image"):
                    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")
                    # Count the number of image files
                    images = len([file for file in os.listdir(path) if file.lower().endswith(image_extensions)])
                    if max_frames is not None:
                        tmp = min(images,max_frames)
                        self.background_dict[dict_name][j][2] = tmp
                        total_images += tmp
                    else:
                        self.background_dict[dict_name][j][2] = images
                        total_images += images
                elif(type == "video"):
                    video = cv2.VideoCapture(path)
                    images = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    if max_frames is not None:
                        tmp = min(images,max_frames)
                        self.background_dict[dict_name][j][2] = tmp
                        total_images += tmp
                    else:
                        self.background_dict[dict_name][j][2] = images
                        total_images += images
            if total_images<num_bg_list[i]:
                raise ValueError(f"Requested more backgrounds ({num_bg_list[i]}) than available in '{dict_name}' ({total_images})")
        # extract image split evenly from every source
        for i, dict_name in enumerate(background_dict_list):
            backgrounds=[]
            total_images_needed = num_bg_list[i]
            image_per_source = [0 for _ in range(len(self.background_dict[dict_name]))]
            total_image_per_source = []
            leftover_images = total_images_needed
            leftover_source = len(self.background_dict[dict_name])
            

            for j, (_, _, num_images) in enumerate(self.background_dict[dict_name]):
                if(num_images<0):
                    total_image_per_source.append([total_images_needed,j])
                else:
                    total_image_per_source.append([num_images,j])

            # Distribute remaining images to sources with available images
            total_image_per_source = sorted(total_image_per_source)
            for minnow,j in total_image_per_source:
                useNow = leftover_source*minnow
                if(useNow<=leftover_images):
                    leftover_images -= minnow
                    image_per_source[j] = minnow
                    leftover_source-=1
                else:
                    useNow = leftover_images//leftover_source
                    isRemain = 1 if (leftover_images%leftover_source!=0) else 0
                    image_per_source[j] = useNow+isRemain
                    leftover_source-=1
                    leftover_images-=useNow+isRemain

                
            # Extract images from sources
            for j, (type, path, _) in enumerate(self.background_dict[dict_name]):
                if type == "rgb":
                    for _ in range(image_per_source[j]):
                        # Here you would add a blank image with the RGB color
                        backgrounds.append(path.copy())
                elif type == "image":
                    images = [file for file in os.listdir(path) if file.lower().endswith(image_extensions)]
                    selected_images = random.sample(images, image_per_source[j])
                    for image_name in selected_images:
                        image = cv2.imread(os.path.join(path, image_name))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        backgrounds.append(image)
                elif type == "video":
                    video = cv2.VideoCapture(path)
                    frame_indices = random.sample(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))), image_per_source[j])
                    for idx in frame_indices:
                        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = video.read()
                        if ret:
                            backgrounds.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # do apply setting to image
            if(background_edit_setting_list[i] is not None):
                all_backgrounds.extend(self._apply_settings(backgrounds,background_edit_setting_list[i]))
            else:
                all_backgrounds.extend(backgrounds)
        
        # do shuffle every images and save in list (optional)
        random.shuffle(all_backgrounds)
        # return list of background the total background eqaul sum of num bg list
        return all_backgrounds
    
    def add_rgb_bg_dict(self, name, rgb, width, height, max_frames:int = None):
        rgb_image = np.full((height, width, 3), rgb, dtype=np.uint8)
        if name in self.background_dict:
            self.background_dict[name].append(["rgb",rgb_image, max_frames])
        else:
            self.background_dict[name] = ["rgb",rgb_image, max_frames]
    def add_dict(self, name, path, type, max_frames:int = None):
        #there are two types first is image this type path will be directory, second is video this type path will be video file
        if name not in self.background_dict:
            self.background_dict[name] = []
        if type == 'image':
            if not os.path.isdir(path):
                raise ValueError(f"Path '{path}' is not a directory.")
            self.background_dict[name].append(["image",path, max_frames])
        elif type == 'video':
            if not os.path.isfile(path) or not path.lower().endswith(('.mp4', '.avi', '.mov')):
                raise ValueError(f"Path '{path}' is not a valid video file.")
            self.background_dict[name].append(["video",path, max_frames])  
    def remove_dict(self, name):
        if name in self.background_dict:
            del self.background_dict[name]
        else:
            raise ValueError(f"Background name '{name}' not found in background_dict.")
    def add_settings(self, setting_dict_name, setting_method_name, *args, image_range = None, **kwargs):
        if not hasattr(self, setting_method_name):
            raise ValueError(f"Augmentation method '{setting_method_name}' does not exist.")
        if setting_dict_name not in self.background_edit_settings_dict:
            self.background_edit_settings_dict[setting_dict_name] = []
        self.background_edit_settings_dict[setting_dict_name].append((getattr(self, setting_method_name), args, kwargs, image_range))

    def clear_settings(self, setting_name):
        if setting_name in self.background_edit_settings_dict:
            del self.background_edit_settings_dict[setting_name]
        else:
            raise ValueError(f"Setting name '{setting_name}' not found in background_edit_settings_dict.")
    def _apply_settings(self, images, setting_name):
        for setting,args,kwargs,image_range in self.background_edit_settings_dict[setting_name]:
            if image_range is None:
                for i in range(len(images)):
                    images[i] = setting(images[i], *args, **kwargs)
            else:
                start, end = image_range
                if(len(images)<start):
                    continue
                if(len(images)-1<end):
                    end = len(images)-1
                for i in range(start,end+1):
                    images[i] = setting(images[i], *args, **kwargs)
        return images
            

    #============================== Settings function ===============================
    def add_color_rectangle(self, image, position=(0.25,0.25,0.75,0.75), rgb = (255,255,255)):
        height, width = image.shape[:2]
        x1, y1 = int(position[0] * width), int(position[1] * height)
        x2, y2 = int(position[2] * width), int(position[3] * height)

        # Add the rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color=rgb, thickness=-1)
        return image
    def hsv_shift(self, image, position=(0.25,0.25,0.75,0.75),brightness_shift=0,contrast_shift=0,hue_shift=0,saturation_shift=0,gamma_shift=0):
        # Convert image to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Calculate the region of interest (ROI)
        height, width = image.shape[:2]
        x1, y1 = int(position[0] * width), int(position[1] * height)
        x2, y2 = int(position[2] * width), int(position[3] * height)
        # Extract ROI
        roi = hsv_image[y1:y2, x1:x2]
        # Apply shifts
        h, s, v = cv2.split(roi)
        # Shift Hue
        h = cv2.add(h, hue_shift)
        # Shift Saturation
        s = cv2.add(s, saturation_shift)
        # Shift Brightness (Value)
        v = cv2.add(v, brightness_shift)
        # Merge and apply ROI back to the image
        shifted_roi = cv2.merge([h, s, v])
        hsv_image[y1:y2, x1:x2] = shifted_roi
        # Convert back to BGR
        shifted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return shifted_image
    
    def rgb_shift(self, image, position=(0.25,0.25,0.75,0.75),red_shift=0,green_shift=0,blue_shift=0):
        # Calculate the region of interest (ROI)
        height, width = image.shape[:2]
        x1, y1 = int(position[0] * width), int(position[1] * height)
        x2, y2 = int(position[2] * width), int(position[3] * height)
        # Extract ROI
        roi = image[y1:y2, x1:x2]
        # Apply shifts
        r, g ,b= cv2.split(roi)
        # Shift Blue
        b = cv2.add(b, blue_shift)
        # Shift Green
        g = cv2.add(g, green_shift)
        # Shift Red
        r = cv2.add(r, red_shift)
        # Merge and apply ROI back to the image
        shifted_roi = cv2.merge([b, g, r])
        image[y1:y2, x1:x2] = shifted_roi
        return image
    
    # def remove_object_with_yolo_model(self, image, model, confident_level=0.5, target_class_name:list = None, rgb=(255,255,255), opacity = 1.0):
    #     # Run YOLO inference
    #     results = model(image, verbose=False)
    #     # results = model(image)
    #     for result in results:
    #         boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
    #         scores = result.boxes.conf.cpu().numpy()  # Confidence scores
    #         class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
    #         # Retrieve the class names from the YOLOv8 model
    #         class_names = result.names
    #         for box, score, class_id in zip(boxes, scores, class_ids):
    #             if score >= confident_level:
    #                 class_name = class_names[int(class_id)]
    #                 # Check if the detected class is in the target_class_name list (if provided)
    #                 if target_class_name is None or class_name in target_class_name:
    #                     x1, y1, x2, y2 = box
    #                     # Create a colored rectangle with the specified opacity
    #                     overlay = image.copy()
    #                     cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), rgb, -1)
    #                     image = cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)
    #     return image

    def add_object(self, image, object_image, num_object, position_range=(0.25, 0.25, 0.75, 0.75), scale_range = (1,1)):
        # Convert the np.array image to a PIL image for manipulation
        image_pil = Image.fromarray(image)
        
        # Get the dimensions of the base image
        image_width, image_height = image_pil.size
        scale = max(100 / image_width, 100 / image_height)
        object_image = object_image.resize((int(image_width*scale), int(image_height*scale)), Image.LANCZOS) #resize image to baze size

        for _ in range(num_object):
            # Generate random scaling factor within the scale range
            scale_factor = np.random.uniform(scale_range[0], scale_range[1])
            
            # Resize the object image according to the scaling factor
            new_object_width = int(object_image.width * scale_factor)
            new_object_height = int(object_image.height * scale_factor)
            scaled_object_image = object_image.resize((new_object_width, new_object_height), Image.LANCZOS)
            
            # Generate random positions within the specified range
            min_x = int(position_range[0] * image_width)
            max_x = int(position_range[2] * image_width) - new_object_width
            min_y = int(position_range[1] * image_height)
            max_y = int(position_range[3] * image_height) - new_object_height
            
            # Ensure the positions are valid
            if max_x <= min_x or max_y <= min_y:
                continue
            
            x_position = np.random.randint(min_x, max_x)
            y_position = np.random.randint(min_y, max_y)
            
            # Paste the scaled object image onto the base image at the random position
            image_pil.paste(scaled_object_image, (x_position, y_position), scaled_object_image)
        
        # Convert the image back to an np.array
        return np.array(image_pil)