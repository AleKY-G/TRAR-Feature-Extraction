import json
from PIL import Image
import os
from tqdm import tqdm
from coco_format_example import coco_example

class json_builder(coco_example):
    def __init__(self, file_path, save_path):
        super(coco_example, self).__init__()
        self.file_path = file_path  # where is the image
        self.save_path = save_path  # where to save the json file
        self.example = {}
        self.set_coco_example()
    #  get image-file name list
    def get_image_name_list(self, file_path):
        image_name_list = os.listdir(file_path)
        return image_name_list

    # create image info with coco format
    def create_image_info(self, file_path):
        image_name_list = self.get_image_name_list(file_path)
        images = []
        for i in tqdm(range(len(image_name_list))):
            try:
                coco_image_format = {}
                coco_image_format['license'] = 2
                coco_image_format['file_name'] = image_name_list[i]
                coco_image_format['coco_url'] = ''

                # open image and get its height and width
                img = Image.open(os.path.join(file_path, image_name_list[i]))
                coco_image_format['height'] = img.height
                coco_image_format['width'] = img.width

                coco_image_format['data_captured'] = '2020-11-3'
                coco_image_format['id'] = str(os.path.splitext(image_name_list[i])[0].split('_')[-1])
                images.append(coco_image_format)
            except:
                continue
        return images

    def create_coco_file(self):
        example = self.get_coco_example()
        coco_file = {}
        coco_file['info'] = example['info']
        coco_file['images'] = self.create_image_info(self.file_path)
        coco_file['licenses'] = example['licenses']
        coco_file['categories'] = example['categories']
        # sava as json file
        json_file = json.dumps(coco_file)
        with open(self.save_path, 'w') as json_writer:
            json_writer.write(json_file)

# Usage
# builder = json_builder(file_path='./dataset , save_path='./example.json')
# builder.create_coco_file()