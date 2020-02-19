# encoding = utf-8
import os
from utils import str_

def get_absolute_path(p):
    if p.startswith('~'):
        p = os.path.expanduser(p)
    return os.path.abspath(p)

def read_lines(p):
    """return the text in a file in lines as a list """
    p = get_absolute_path(p)
    f = open(p,'r')
    return f.readlines()

def read_class_names_462(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            # names[ID] = name.strip('\n')
            names[name.strip('\n')] = ID
    return names

def read_class_names_251(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for _, name in enumerate(data):
            # names[ID] = name.strip('\n')
            label_ID, name = str_.split(name, ' ')
            names[name.strip('\n')] = int(label_ID)
    return names
