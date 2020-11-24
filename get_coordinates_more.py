import pathlib
from get_coordinates import get_coordinates
from PIL import Image
import pandas as pd
from show_image import show_image_objects

pic_list = [p for p in pathlib.Path('AWEForSegmentation/testannot_rect').iterdir() if p.is_file()]


dataset = dict()
dataset['img_name'] = list()
dataset['x_min'] = list()
dataset['y_min'] = list()
dataset['x_max'] = list()
dataset['y_max'] = list()
dataset['class_name'] = list()

counter = 0

for name in pic_list[:5]:
    img_name = str(name)[-8:]

    coordinates, object_n = get_coordinates(str(str(name)[-8:-4]))
    img_dir = 'AWEForSegmentation/test/' + str(name)[-8:-4] + '.png'

    for j in range(object_n):
        dataset['x_min'].append(coordinates[j][0])
        dataset['y_min'].append(coordinates[j][1])
        dataset['x_max'].append(coordinates[j][2])
        dataset['y_max'].append(coordinates[j][3])
        dataset['img_name'].append(f'test/ears_test_{counter}.jpeg')
        dataset['class_name'].append('ear')

        #print(dataset)




    #dataset['x_min'].append(coordinates[0])
    #dataset['y_min'].append(coordinates[1])
    #dataset['x_max'].append(coordinates[2])
    #dataset['y_max'].append(coordinates[3])

    #img_dir = 'AWEForSegmentation/test/' + str(name)[-8:-4] + '.png'
    img = Image.open(img_dir).convert('RGB')
    img.save(f'test/ears_test_{counter}.jpeg', 'JPEG')
    #dataset['img_name'].append(f'test/ears_test_{counter}.jpeg')
    #dataset['class_name'].append('ear')
    counter += 1

#print(dataset)

df = pd.DataFrame(dataset)
#print(df.head(10))

show_image_objects(df.iloc[4])


