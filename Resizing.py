#!/usr/bin/python
from PIL import Image
import os, sys

path = "all_10_classes/"
dirs = os.listdir(path)

def resize():
    for folder in dirs:
        print(folder)
        if os.path.isdir(path + folder):
            items = os.listdir(path + folder + "/")
            for item in items:
                if os.path.isfile(path+folder+"/"+item):
                    try:
                        im = Image.open(path+folder+"/"+item).convert('RGB')
                    except OSError:
                        continue
                    f, e = os.path.splitext(path+folder+"/"+item)
                    im_resize = im.resize((100, 60), Image.ANTIALIAS)
                    im_resize.save(f + '2.jpg', 'JPEG', quality=90)
                    os.remove(path+folder+"/"+item)

resize()
