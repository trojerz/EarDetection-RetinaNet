import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kerasretinanet.keras_retinanet import models
from kerasretinanet.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from kerasretinanet.keras_retinanet.utils.visualization import draw_box, draw_caption
from kerasretinanet.keras_retinanet.utils.colors import label_color

from keras import backend as K

from iou import iou


import tensorflow.compat.v1 as tf1
config = tf1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf1.InteractiveSession(config=config)

model_path = 'snapshots/resnet50_csv_10.h5'

model = models.load_model(model_path, backbone_name = 'resnet50')
model = models.convert_model(model)
labels_to_names = pd.read_csv('classes.csv', header=None).T.loc[0].to_dict()
df_test = pd.read_csv('df_test_metrics.csv')

iou_data = []
def show_detected_objects(image_row):

    print(image_row.img_name)

    THRES_SCORE = 0.3
    img_path = image_row.img_name
    true_box = [image_row.x_min, image_row.y_min, image_row.x_max, image_row.y_max]
    image = read_image_bgr(img_path)

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    image = preprocess_image(image)
    image, scale = resize_image(image)
    boxes, scores, _ = model.predict_on_batch(np.expand_dims(image, axis = 0))

    boxes /= scale

    #draw_box(draw, true_box, color = (255,255,0))

    
    best_pred = []
    for box, score in zip(boxes[0], scores[0]):
        if score < THRES_SCORE:
            break
        best_pred.append((box, score))
    best_pred = sorted(best_pred, key=lambda tup: tup[1],  reverse=True)

    box = best_pred[0][0]
    score = best_pred[0][1]
    
    

    b = box.astype(int)

    x_min, y_min, x_max, y_max = image_row.x_min, image_row.y_min, image_row.x_max, image_row.y_max
    x_min_p, y_min_p, x_max_p, y_max_p = b
    rect1 = (x_max, y_max, x_max - x_min, y_max - y_min)
    rect2 = (x_max_p, y_max_p, x_max_p - x_min_p, y_max_p - y_min_p)
    
    return(iou(rect1,rect2))

    #x_max, y_max, x_max - x_min, y_max - x_min
    #plt.axis('off')
    #plt.imshow(draw)
    #plt.show()

for k in range(300):
    try:
        m = show_detected_objects(df_test.iloc[k])
        iou_data.append(m)
        print(m)
        K.clear_session()
    except:
        pass

#print(show_detected_objects(df_test.iloc[0]))
print(f'Average IoU is {round(100*  (sum(iou_data) / len(iou_data)))} %. Maximum IoU: {round(100 * max(iou_data), 2)} %.')
