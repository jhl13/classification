# encoding = utf-8
import sys
import os
sys.path.append('../')
import numpy as numpy
import utils.io_
import utils.str_

def cal_category(data_file):
    category = {}
    instance_paths = []
    lines = utils.io_.read_lines(data_file)
    for line in lines:
        if len(line) == 0:
            continue
        line = utils.str_.remove_all(line, '\xef\xbb\xbf')
        line = utils.str_.remove_all(line, '\n')
        line_data = utils.str_.split(line, ',')
        instance_category = line_data[0]
        instance_path = line_data[1]
        instance_paths.append(instance_path)
        if instance_category not in category:
            category[instance_category] = 1
        else:
            category[instance_category] = category[instance_category] + 1
    return category, instance_paths

if __name__ == "__main__":
    root_dir = "/home/ljh/workspace/datasets/baseline_food"
    train_category, train_instance_paths = cal_category("../dataset/train.txt")
    test_category, test_instance_paths = cal_category("../dataset/test.txt")
    assert len(train_category.keys()) == len(test_category.keys())

    flag_equal = 1
    for i in train_category.keys():
        if i not in test_category.keys():
            print ("Warning!!! train category:", i, " is not in test category.")
            flag_equal = 0

    assert flag_equal == 1
    print ("train category is equal to test category.")

    flag_existance = 1
    for i in train_instance_paths:
        instance_path = os.path.join(root_dir, i)
        if os.path.exists(instance_path) is not True:
            print(instance_path, "is not existing")
            flag_existance = 0

    for i in test_instance_paths:
        instance_path = os.path.join(root_dir, i)
        if os.path.exists(instance_path) is not True:
            print(instance_path, "is not existing")
            flag_existance = 0
    
    assert flag_existance == 1
    print ("All files exist.")

    # f = open("./name.names", 'w')
    # for name in train_category.keys():
    #     f.write(name+"\n")
    # f.close()
