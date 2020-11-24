### Overview

The code for this project was created by Pierluigi Ferrari in his Github repository [keras-retinanet](https://github.com/fizyr/keras-retinanet). The project was copied and adapted for this assignment.

RetinaNet was introduced in the paper [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).

### Dependencies

* Python 3.6
* TensorFlow 2.4.0
* Keras 2.4.3
* OpenCV

Exported environment: environment.yml

### Instructions

 * Install all dependencies from the `environment.yml` file (I did not use Jupyter Notebook due to issues with incompatibility of Tensorflow and Jupyter Notebook).
 * Download pretrained model [Google Drive](https://drive.google.com/drive/folders/1foIOYDwGeLxkyg1Em3kbR3_PEqtR7Kej?usp=sharing) and put it into main file.
 * Put files from [keras-retinanet](https://github.com/fizyr/keras-retinanet) to `kerasretinanet` file.
 * (optional) In a case you want to train model on your own dataset:
      + train model: `python kerasretinanet/keras_retinanet/bin/train.py --freeze-backbone --random-transform --weights snapshots/_pretrained_model.h5 --batch-size 8 --steps 500 --epochs 5 csv annotations.csv classes.csv`
      + convert model: `python kerasretinanet/keras_retinanet/bin/convert_model.py snapshots/resnet50_csv_10.h5 snapshots/inference/resnet50_csv_10.h5`
* Run `get_results.py` to get prediction on the image. You need to add your record to `df_test_metrics.csv`.
* Run `get_metrics.py` to get metrics for the model. I used IoU metric (see `iou.py` script for more information). 
* evaluate model: `python kerasretinanet/keras_retinanet/bin/evaluate.py csv df_test.csv classes.csv snapshots/inference/resnet50_csv_10.h5`
