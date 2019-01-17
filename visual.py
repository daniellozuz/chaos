import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.image import AxesImage
import skimage.transform as trans
import numpy as np
import os
from metrics import Measurer
import skimage.io as io
from constants import RESOLUTION, SIZE


class Visualizer(animation.TimedAnimation):
    def __init__(self):
        self.epochs = self._get_epochs()
        self.resolution = RESOLUTION
        fig = plt.figure()
        fig.suptitle('x')
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        ax1.set_xlim(0, SIZE)
        ax1.set_ylim(0, SIZE)
        ax2.set_xlim(0, SIZE)
        ax2.set_ylim(0, SIZE)
        ax3.set_xlim(0, SIZE)
        ax3.set_ylim(0, SIZE)
        ax4.set_xlim(0, SIZE)
        ax4.set_ylim(0, SIZE)
        self.image1 = AxesImage(ax1, cmap='gray')
        self.image2 = AxesImage(ax2, cmap='gray')
        self.image3 = AxesImage(ax3, cmap='gray')
        self.image4 = AxesImage(ax4)
        ax1.add_image(self.image1)
        ax2.add_image(self.image2)
        ax3.add_image(self.image3)
        ax4.add_image(self.image4)
        self.gen = self.image_generator()
        self.tmp_init_gen = self.image_generator()
        animation.TimedAnimation.__init__(self, fig, interval=250, blit=True, repeat_delay=2000)

    def _get_epochs(self):
        epochs = []
        for prediction in sorted(os.listdir('data/test/')):
            if 'prediction' in prediction:
                epochs.append(prediction[-3:])
        return epochs

    def _init_draw(self):
        image, label, prediction, overlapped = next(self.tmp_init_gen)
        self.image1.set_data(image)
        self.image2.set_data(label)
        self.image3.set_data(prediction)
        self.image4.set_data(overlapped)

    def _draw_frame(self, framedata):
        image, label, prediction, overlapped = next(self.gen)
        self.image1.set_data(image)
        self.image2.set_data(label)
        self.image3.set_data(prediction)
        self.image4.set_data(overlapped)
        self._drawn_artists = [self.image1, self.image2, self.image3, self.image4]

    def new_frame_seq(self):
        total_number_of_images = len(os.listdir('data/test/image')) * len(self.epochs)
        return iter(range(total_number_of_images))

    def image_generator(self):
        while True:
            for epoch in self.epochs:
                for name in sorted(os.listdir('data/test/image')):
                    image = 255.0 * plt.imread(f'data/test/image/{name}')
                    label = 255.0 * plt.imread(f'data/test/label/{name}')
                    prediction = 255.0 * plt.imread(f'data/test/prediction{epoch}/{name}')
                    border = Measurer()._get_border(prediction / 255)
                    overlapped = trans.resize(label, self.resolution)
                    overlapped[border] = 128
                    yield image, label, prediction, overlapped

    def show_animation(self):
        plt.show()
