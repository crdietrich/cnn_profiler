"""Image Classifier for comparing predictions

2019 Colin Dietrich
"""

import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import imagenet_utils


class ImageClassifier:
    def __init__(self):

        self.model = None

        self.h = 128
        self.w = 128
        self.depth_multiplier = 1.0
        self.d = None
        self.df = None

        self.label_file = None
        self.labels = None

    @staticmethod
    def name_from_directory(dir_path):
        return dir_path.split(os.path.sep)[1].split("-")[1]

    def preprocess(self, file_path, to_array=False, expand=False, scale=False):
        """Load and convert JPG image file to Numpy Array

        Parameters
        ----------
        h : int, pixel height
        w : int, pixel width
        to_array : bool,
        expand : bool, expand dims (if multiple images sent)
        scale : bool, scale pixels to -1 to 1 (used by TF in Keras)

        Returns
        -------
        img_a : Numpy Array of shape (h, w, 3)
        """
        image_a = load_img(file_path, target_size=(self.h, self.w))
        if to_array:
            image_a = img_to_array(image_a)
        if expand:
            image_a = np.expand_dims(image_a, axis=0)
        if scale:
            image_a = imagenet_utils.preprocess_input(image_a, mode="tf")
        return image_a

    def predict_dataset(self, dataset_path):
        """Predict top 1 label for each image in directory_path

        Parameters
        ----------
        dataset_path : str, path to folder containing images
            assuming it contains subfolders for each class
            and that the folder is named for the class

        Returns
        -------
        dict of lists, where keys are the true class names, and
            the list is of class predictions for each image in
            directory_path
        """

        ddf = list(os.walk(os.path.normpath(dataset_path)))
        self.d = {}
        for dirpath, dirnames, filenames in ddf[1:]:
            name = self.name_from_directory(dirpath)
            predicted_labels = []
            for f_name in filenames:
                f_path = os.path.normpath(dirpath + os.path.sep + f_name)
                p_label = self.predict_file(f_path)
                predicted_labels.append(p_label)
            self.d[name] = predicted_labels

    def collate_predictions(self):
        """Collate predictions into a Pandas DataFrame
        and axis labels for a Confusion Matrix

        Parameters
        ----------
        d : dict of lists, where keys are the true class names, and
            the list is of class predictions for each image in
            directory_path

        Returns
        -------
        df : Pandas DataFrame, with one row per image and columns:
            y_true : true value of image being classified
            y_pred : predicted class of image
        d_label_ax : list of str, labels for confusion matrix axes
        """
        d_label = []
        d_pred = []
        for k, v in self.d.items():
            d_label += [k]*len(v)
            d_pred += v
        self.df = pd.DataFrame({"y_true":d_label, "y_pred":d_pred})

class ClassifyRegular(ImageClassifier):
    def __init__(self):
        super().__init__()

    def load_model(self, model_instance=False):
        """Load a pretrained model"""
        if not model_instance:

            from keras.applications import mobilenet_v2

            self.model = mobilenet_v2.MobileNetV2(
                             input_shape=(self.h, self.w, 3))
            #, depth_multiplier=self.depth_multiplier)

    def predict(self, image_a, top=1, score=False):
        p_n_label = self.model.predict(image_a)
        pred = imagenet_utils.decode_predictions(p_n_label, top=top)
        class_id, class_name, class_score = pred[0][0]
        if score:
            return class_name, class_score
        else:
            return class_name

    def predict_file(self, file_path, top=1):
        image_a = self.preprocess(file_path, expand=True, scale=True)
        class_name = self.predict(image_a, top=top)
        return class_name

class ClassifyTPU(ImageClassifier):
    def __init__(self):
        super().__init__()
        self.label_file = "/home/pi/Downloads/imagenet_labels.txt"
        self.model_file = "mobilenet_v2_1.0_224_quant_edgetpu.tflite"

    def load_model(self, label_file=None, model_file=None):
        """Load a pretrained model"""
        if label_file is not None:
            self.label_file = label_file
        self.labels = self.read_label_file(label_file)  # Prepared labels

        if model_file is not None:
            self.model_file = model_file

        from edgetpu.classification.engine import ClassificationEngine

        self.model = ClassificationEngine(model_file)  # Initialize TPU engine

    def read_label_file(self, file_path):
        """Function to read labels from text files"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        ret = {}
        for line in lines:
            pair = line.strip().split(maxsplit=1)
            ret[int(pair[0])] = pair[1].strip()
        return ret

    def predict(self, image_a, top=5, score=False):
        [(p_n_label, p_score)] = self.model.ClassifyWithImage(image_a)
        class_id = self.labels[p_n_label]
        if score:
            return class_id, p_score
        else:
            return class_id

    def predict_file(self, file_path, top=5):
        image_a = self.preprocess(file_path)
        class_id = self.predict(image_a, top=top)
        return class_id
