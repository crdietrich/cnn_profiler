"""Flow data from directory

Colin Dietrich 2019
"""


from keras.preprocessing.image import ImageDataGenerator


class ImageFlow:
    def __init__(self, d_directory, **kwargs):
        self.d_directory = d_directory
        self.img_height = 128
        self.img_width = 128
        self.shear_range = 0.2
        self.zoom_range = 0.2
        self.horizontal_flip = False
        self.vertical_flip = False
        self.validation_split = 0.2
        self.fillmode = 'nearest'

        self.seed = None
        self.batch_size = 10
        self.class_mode = 'categorical'

        train_datagen = ImageDataGenerator(rescale=1./255,
                                           shear_range=self.shear_range,
                                           zoom_range=self.zoom_range,
                                           horizontal_flip=self.horizontal_flip,
                                           vertical_flip=self.vertical_flip,
                                           validation_split=0.2,  # percent of images used for validation
                                           **kwargs)

        self.train_generator = train_datagen.flow_from_directory(
            self.d_directory,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            seed=self.seed,
            subset='training')  # set as training data

        self.validation_generator = train_datagen.flow_from_directory(
            self.d_directory,  # same directory as training data
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            seed=self.seed,
            subset='validation')  # set as validation data
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        self.test_generator = test_datagen.flow_from_directory(
            self.d_directory,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            seed=self.seed,
            shuffle=False)
        
    @staticmethod
    def display_images(img_generator, n=15, cols=5):
        import math
        import numpy as np
        import matplotlib.pyplot as plt
        class_mapper = {v: k for k, v in img_generator.class_indices.items()}

        rows = int(math.ceil(n / cols))
        plt.figure(figsize=(cols * 3, rows * 3))
        i = 0
        while True:
            a = next(img_generator)
            for j, image_array in enumerate(a[0]):  # image arrays
                if i == n:
                    return plt.show()
                plt.subplot(rows, cols, i + 1)
                plt.axis('off')
                plt.imshow(image_array)
                plt.title(class_mapper[np.argmax(a[1][j])])
                i += 1