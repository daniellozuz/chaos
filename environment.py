import pydicom
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import zipfile
import os
import skimage.transform as trans
import skimage.io as io
from constants import RESOLUTION, SIZE
import shutil


class Environment(object):
    DIRECTORIES = [
        'data/train/image',
        'data/train/label',
        'data/test/image',
        'data/test/label',
        'data/generated',
        'data',
        'weights',
    ]

    def __init__(self, training=[2], testing=[5], skip_blacks=True, testing_every=1, training_every=1):
        self.training = training
        self.testing = testing
        self.skip_blacks = skip_blacks
        self.testing_every = testing_every
        self.training_every = training_every

    def configure(self):
        self.unzip_images()
        self.remove_directories()
        self.create_directories()
        self.prepare_data()

    def unzip_images(self):
        for filename in ['chaos/CT_data_batch1.zip', 'chaos/MR_data_batch1.zip']:
            zip_ref = zipfile.ZipFile(filename, 'r')
            zip_ref.extractall('chaos')
            zip_ref.close()

    def remove_directories(self):
        for directory in self.DIRECTORIES:
            shutil.rmtree(directory, ignore_errors=True)

    def create_directories(self):
        for directory in self.DIRECTORIES:
            os.makedirs(directory, exist_ok=True)

    def convert_image(self, source, target):
        ds = (pydicom.dcmread(source).pixel_array / 10).astype(np.float32)  # XXX Some weird stuff, do it properly
        self.convert_and_save(ds, target)

    def convert_label(self, source, target):
        ds = plt.imread(source) * 255.1  # TODO does that even work?
        self.convert_and_save(ds, target)

    def convert_and_save(self, ds, target):
        ds = Image.fromarray(ds)
        if ds.mode != 'L':
            ds = ds.convert('L')
        ds = ds.resize(RESOLUTION)
        ds.save(target)

    def prepare_data(self):
        self.prepare_training()
        self.prepare_testing()

    def prepare_training(self):
        self.prepare(self.training, 'train', self.training_every)

    def prepare_testing(self):
        self.prepare(self.testing, 'test', self.testing_every)

    def prepare(self, patient, folder_name, step):
        for folder in patient:
            dicom_paths = sorted(list(x.path for x in os.scandir(f'chaos/CT_data_batch1/{folder}/DICOM_anon')))
            label_paths = sorted(list(x.path for x in os.scandir(f'chaos/CT_data_batch1/{folder}/Ground')))
            numbers = range(len(dicom_paths))
            for number, dicom_path, label_path in zip(numbers, dicom_paths, label_paths):
                if self.skip_blacks and np.all(plt.imread(label_path) == 0.0):
                    print(f'Skipping image (no liver): {folder}, {label_path}')
                    continue
                if number % step == 0:
                    self.convert_image(dicom_path, f'data/{folder_name}/image/{folder}_{number}.png')
                    self.convert_label(label_path, f'data/{folder_name}/label/{folder}_{number}.png')
