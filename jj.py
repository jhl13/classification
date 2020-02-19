import pandas as pd
import cv2
import os
import numpy as np
from utils import str_

train_labels = pd.read_csv("/home/luo13/workspace/datasets/baseline_food/ifood-2019-fgvc6/train_labels.csv")
train_dir = "ifood-2019-fgvc6/train_set"
train_labels['path'] = train_labels['img_name'].map(lambda x: os.path.join(train_dir, x))
print (train_labels)
print (len(train_labels))
print (train_labels['img_name'][0])

class_file_name = "/home/luo13/workspace/datasets/baseline_food/ifood-2019-fgvc6/class_list.txt"
with open(class_file_name, 'r') as data:
    names = {}
    for ID, name in enumerate(data):
        # names[ID] = name.strip('\n')
        label_ID, name = str_.split(name, ' ')
        names[name.strip('\n')] = int(label_ID)
print (names)