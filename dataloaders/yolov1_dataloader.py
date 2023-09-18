import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import os
import sys


class YOLOV1DataLoader(tf.keras.utils.Sequence):
    """
    Data loader for YOLOV1 data set. Mostly based on https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/dataset.py 
    but done with TensorFlow instead of PyTorch.
    """
    def __init__(self, df, batch_size, image_shape, data_path, boxes=2, cells=7, classes=20):
        self.df = df
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.label_path = os.path.join(data_path, "labels")
        self.image_path = os.path.join(data_path, "images")
        self.cells = cells
        self.boxes = boxes
        self.classes = classes
        self.length = len(df)



    def __get_image(self, file_id):
        img = np.asarray(Image.open(file_id))
        img = np.resize(img, self.image_shape)
        # Add batch dimension before returning
        return np.expand_dims(img, axis=0)

    def __get_raw_label(self, label_file):
        labels = []
        with open(label_file, "r") as f:
            for val in f.readlines():
                val = val.replace("\n", "").split()
                class_label, x, y, width, height = [float(x) if float(x) != int(float(x)) else int(float(x)) for x in val]
                labels.append([class_label, x, y, width, height])
            return self.preprocess_labels(labels)


    def preprocess_labels(self, labels):
        label_matrix = np.zeros((self.cells, self.cells, self.classes + 5 * self.boxes))
        for label in labels:
            class_label, x, y,width,height = label
            
            i, j = int(self.cells * y), int(self.cells * x)
            x_cell, y_cell = self.cells * x - j, self.cells * y - i
            width_cell = width * self.cells
            height_cell = height * self.cells

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coordinates = tf.constant(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1
        label_matrix = tf.constant(label_matrix)
        return tf.expand_dims(label_matrix, axis=0)


    def __getitem__(self, idx):
        batch_x = self.df["image"][idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.df["label_file"][idx * self.batch_size:(idx + 1) * self.batch_size]
        x = [self.__get_image(os.path.join(self.image_path, file_id)) for file_id in batch_x] 
        y = [self.__get_raw_label(os.path.join(self.label_path, label_file)) for label_file in batch_y]
        
        # Convert to tensors and concat on batch axis to get the image and label tensors
        x = tf.concat([tf.constant(e) for e in x], axis=0) # N x 448 x 448 x 3
        y = tf.concat([tf.constant(e) for e in y], axis=0) # N x 7 x 7 x 30
        y = tf.cast(y, tf.float32)
        return x, y
    

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)


    def __len__(self):
        return self.length // self.batch_size

if __name__ == "__main__":
    input_folder = sys.argv[1]
    # For testing purposes. Give path to archive-folder as input to script.
    train_df = pd.read_csv(os.path.join(input_folder, "train.csv"), names=["image", "label_file"])
    c = YOLOV1DataLoader(train_df, 16, (448,448,3), input_folder)
    images, labels = c.__getitem__(0)
    print(images.shape, labels.shape)
