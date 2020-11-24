

model.compile(
    loss={
        'regression'    : keras_retinanet.losses.smooth_l1(),
        'classification': keras_retinanet.losses.focal()
    },
    optimizer=keras.optimizers.Adam(lr=1e-5, clipnorm=0.001)
)

PRETRAINED_MODEL = './snapshots/_pretrained_model.h5'


#train model
python kerasretinanet/keras_retinanet/bin/train.py --freeze-backbone --random-transform --weights snapshots/_pretrained_model.h5 --batch-size 8 --steps 500 --epochs 5 csv annotations.csv classes.csv

#convert model
python kerasretinanet/keras_retinanet/bin/convert_model.py snapshots/resnet50_csv_10.h5 snapshots/inference/resnet50_csv_10.h5

#evaluate model
python kerasretinanet/keras_retinanet/bin/evaluate.py csv df_test.csv classes.csv snapshots/inference/resnet50_csv_10.h5
