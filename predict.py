from kerasretinanet.keras_retinanet import models
import pandas as pd

model_path = 'snapshots/resnet50_csv_05.h5'

model = models.load_model(model_path, backbone_name = 'resnet50')
model = models.convert_model(model)

labels_to_names = pd.read_csv('classes.csv', header=None).T.loc[0].to_dict()



def show_detected_objects(image_row):

    THRES_SCORE = 0.6
    img_path = image_row.image_name
    true_box = [image_row.x_min, image_row.y_min, image_row.x_max, image_row.y_max]
    image = read_image_bgr(img_path)

    draw = image.copy()
    draw = cv2.cvtColor(draw, cs2.COLOR.BGR2RGB)

    image = preprocess_image(image)
    image, scale = resize_image(image)
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis = 0))

    boxes /= scale

    draw_box(draq, true_box, color = (255,255,0))

    for box, socre, label in zip(boxes[0], scores[0], labels[0]):
        if score < THRES_SCORE:
            break
    
        color = label_color(label)
        b = box.astype(int)
        draw_box(draw, b, color = color)

        caption = "{}  {:.3f}".format(labels_to_names[label], score)
        draw_cation(draw, b. caption)

    plt.axis('off')
    plt.imgshow(draw)
    plt.show()

show_detected_objects(df_test.iloc[0])