from kerasretinanet.keras_retinanet.utils.image import read_image_bgr
from kerasretinanet.keras_retinanet.utils.visualization import draw_box, draw_caption 
import cv2
import matplotlib.pyplot as plt


def show_image_objects(image):

    img_path = image.img_name
    box = [image.x_min, image.y_min, image.x_max, image.y_max]
    image = read_image_bgr(img_path)

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    draw_box(draw, box, color= (255,255,0))

    plt.axis('on')
    plt.imshow(draw)
    plt.show()
