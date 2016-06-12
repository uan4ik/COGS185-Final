from PIL import Image
import os, sys
import scipy.ndimage
import numpy as np
import pickle
from scipy import stats



path = "all_10_classes/"
dirs = os.listdir(path)
num_labels = 2
val_size = 100 * num_labels
test_size = 100 * num_labels 
mini_size = 11
mini = 0 
out_file = str(num_labels) + "_classes_ImageNet.p"




label_dict = {}
data = []
labels = []
label_number = 0
for folder in dirs:
    if label_number == num_labels:
        break
    else:
        print("processing ", folder)
    if os.path.isdir(path + folder):
        items = os.listdir(path + folder + "/")
        count = 0
        folder_items = []
        for item in items:
            count += 1
            print(item)
            if count == mini_size and mini == 1:
                break
            if os.path.isfile(path + folder + "/" + item):
                image = scipy.ndimage.imread(path + folder + "/" + item)
                if image.shape == (60, 100, 3):
                    image = image.reshape(3, 100, 60)
                    folder_items.append((image - np.mean(image)).astype("float32"))
                    if folder not in label_dict:
                        label_dict[folder] = label_number
                        label_number += 1
                    labels.append(label_dict[folder])
        print("gets here ", folder_items[0].shape)
        data.extend(folder_items)



labels = np.array(labels)
data = np.array(data)


# shuffle data
dummy = np.arange(len(data))
np.random.shuffle(dummy)
data = data[dummy]
labels = labels[dummy]



if mini == 1:
    val_size = mini_size
    test_size = mini_size

X_train, X_val = data[:-val_size], data[-val_size:]
y_train, y_val = labels[:-val_size], labels[-val_size:]

X_train, X_test = X_train[:-test_size], X_train[-test_size:]
y_train, y_test = y_train[:-test_size], y_train[-test_size:]


pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), open(out_file, "wb"), protocol=4)
