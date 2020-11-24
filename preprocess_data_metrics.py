import pathlib
from get_coordinates import get_coordinates
from PIL import Image
import pandas as pd
from show_image import show_image_objects
import os
from kerasretinanet.keras_retinanet import models
from kerasretinanet.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from kerasretinanet.keras_retinanet.utils.visualization import draw_box, draw_caption
from kerasretinanet.keras_retinanet.utils.colors import label_color
import cv2
import matplotlib.pyplot as plt
import numpy as np

#some fixes so we can train model
import tensorflow.compat.v1 as tf1
config = tf1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf1.InteractiveSession(config=config)

#prepare test pictures and annotations
pic_list = [p for p in pathlib.Path('AWEForSegmentation/testannot_rect').iterdir() if p.is_file()]
dataset = dict()
dataset['img_name'] = list()
dataset['x_min'] = list()
dataset['y_min'] = list()
dataset['x_max'] = list()
dataset['y_max'] = list()
dataset['class_name'] = list()
counter = 1

for name in pic_list:
    img_name = str(name)[-8:]
    coordinates, object_n = get_coordinates(str(str(name)[-8:-4]), 'testannot_rect')
    img_dir = 'AWEForSegmentation/test/' + str(name)[-8:-4] + '.png'
    if object_n >= 2:
        continue
    for j in range(object_n):
        dataset['x_min'].append(coordinates[j][0])
        dataset['y_min'].append(coordinates[j][1])
        dataset['x_max'].append(coordinates[j][2])
        dataset['y_max'].append(coordinates[j][3])
        dataset['img_name'].append(f'test/ears_test_{counter}.jpeg')
        dataset['class_name'].append('ear')
    img = Image.open(img_dir).convert('RGB')
    #img = img.resize((200,200))
    img.save(f'test/ears_test_{counter}.jpeg', 'JPEG')
    counter += 1
df_test = pd.DataFrame(dataset)
df_test.to_csv('df_test_metrics.csv', index = False, header = ['img_name','x_min','y_min','x_max','y_max','class_name']) 
