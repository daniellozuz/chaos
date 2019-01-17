import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from constants import RESOLUTION


class Generator(Sequence):
    def __init__(self, path, batch_size=1, augmentation_parameters={}, shuffle=True, save_to_dir=None):
        self.path = path
        self.batch_size = batch_size
        self.augmentation_parameters = augmentation_parameters
        self.shuffle = shuffle
        self.save_to_dir = save_to_dir
        self.gen = self._generator()

    def _generator(self):
        for img, label in zip(self._image_generator(), self._label_generator()):
            yield self._adjust_data(img, label)

    def _image_generator(self):
        return self.__generator('image')

    def _label_generator(self):
        return self.__generator('label')

    def __generator(self, class_name):
        datagen = ImageDataGenerator(self.augmentation_parameters)
        generator = datagen.flow_from_directory(self.path,
                                                classes=[class_name],
                                                class_mode=None,
                                                color_mode='grayscale',
                                                target_size=RESOLUTION,
                                                batch_size=self.batch_size,
                                                save_to_dir=self.save_to_dir,
                                                save_prefix=class_name,
                                                shuffle=self.shuffle,
                                                seed=1)
        return generator

    def _adjust_data(self, img, label):
        img = img / 255
        label = label / 255
        label[label > 0.5] = 1.0
        label[label <= 0.5] = 0.0
        return img, label

    def __getitem__(self, index):
        return next(self.gen)

    def __len__(self):
        return len(os.listdir(os.path.join(self.path, 'image'))) // self.batch_size


class TrainGenerator(Generator):
    def __init__(self, batch_size, save_to_dir=None):
        augmentation_parameters = {
            # 'rotation_range': 0.2,
            # 'width_shift_range': 0.01,
            # 'height_shift_range': 0.01,
            # 'shear_range': 0.01,
            # 'zoom_range': 0.02,
            # 'fill_mode': 'nearest'
        }
        super().__init__('data/train',
                         batch_size=batch_size,
                         save_to_dir=save_to_dir,
                         augmentation_parameters=augmentation_parameters)


class ValidationGenerator(Generator):
    def __init__(self):
        super().__init__('data/test', shuffle=False)


class PredictionGenerator(Generator):
    def __init__(self):
        super().__init__('data/test', shuffle=False)

    def _generator(self):
        for img, label in zip(self._image_generator(), self._label_generator()):
            yield self._adjust_data(img, label)[0]
