import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from skimage.morphology import binary_erosion, binary_dilation
from skimage.morphology.selem import square
from constants import RESOLUTION


class Measurer(object):
    class Decorators(object):
        THRESHOLD = 0.5

        @classmethod
        def logical(cls, decorated):
            def wrapped(measurer, *args):
                return decorated(measurer, *tuple(x > cls.THRESHOLD for x in args))
            return wrapped

    def __init__(self):
        self.epochs = self._get_epochs()

    def _get_epochs(self):
        return [folder_name[-3:] for folder_name in sorted(os.listdir('data/test/')) if 'prediction' in folder_name]

    def show_metrics(self):
        for epoch in self.epochs:
            print(f'Epoch: {int(epoch):>3}', end=' | ')
            print(f'VO (1): {self.volumetric_overlap(epoch):>4.2f}', end=' | ')
            print(f'RAVD (0): {self.relative_absolute_volume_difference(epoch):>5.2f}', end=' | ')
            print(f'ASSD (0): {self.average_symmetric_surface_distance(epoch):>5.2f}', end=' | ')
            print(f'RMSSSD (0): {self.rms_symmetric_surface_distance(epoch):>6.2f}', end=' | ')
            print(f'MSSD (0): {self.max_symmetric_surface_distance(epoch):>6.2f}')

    def volumetric_overlap(self, epoch):
        return np.mean([self._volumetric_overlap(_y_pred, _y_true) for _y_pred, _y_true in self._generator(epoch)])

    def relative_absolute_volume_difference(self, epoch):
        return np.mean([self._relative_absolute_volume_difference(_y_pred, _y_true) for _y_pred, _y_true in self._generator(epoch)])

    def average_symmetric_surface_distance(self, epoch):
        return np.mean([self._average_symmetric_surface_distance(_y_pred, _y_true) for _y_pred, _y_true in self._generator(epoch)])

    def rms_symmetric_surface_distance(self, epoch):
        return np.mean([self._rms_symmetric_surface_distance(_y_pred, _y_true) for _y_pred, _y_true in self._generator(epoch)])

    def max_symmetric_surface_distance(self, epoch):
        return np.mean([self._max_symmetric_surface_distance(_y_pred, _y_true) for _y_pred, _y_true in self._generator(epoch)])

    def _generator(self, epoch):
        for image_name in sorted(os.listdir('data/test/image')):
            prediction = io.imread(os.path.join(f'data/test/prediction{epoch}', image_name), as_gray=True)
            label = io.imread(os.path.join('data/test/label', image_name), as_gray=True)
            prediction = prediction / 255 ** 2
            label = label / 255
            prediction = trans.resize(prediction, RESOLUTION)
            label = trans.resize(label, RESOLUTION)
            yield prediction, label

    @Decorators.logical
    def _volumetric_overlap(self, a, b):
        return self._intersection_area(a, b) / self._union_area(a, b)

    @Decorators.logical
    def _relative_absolute_volume_difference(self, a, b):
        return abs(1 - self._area(a) / self._area(b))

    @Decorators.logical
    def _average_symmetric_surface_distance(self, a, b):
        return np.mean(self._symmetric_surface_distances(a, b))

    @Decorators.logical
    def _rms_symmetric_surface_distance(self, a, b):
        return np.mean(self._symmetric_surface_distances(a, b) ** 2) ** 0.5

    @Decorators.logical
    def _max_symmetric_surface_distance(self, a, b):
        return np.max(self._symmetric_surface_distances(a, b))

    @Decorators.logical
    def _symmetric_surface_distances(self, a, b):
        border_a = self._get_border(a)
        border_b = self._get_border(b)
        distances_of_a_to_b = self._min_distances_between_borders(border_a, border_b)
        distances_of_b_to_a = self._min_distances_between_borders(border_b, border_a)
        return np.concatenate((distances_of_a_to_b, distances_of_b_to_a), axis=None)

    @Decorators.logical
    def _get_border(self, a):
        pic_border = np.pad(np.zeros(tuple(map(lambda x: x - 2, a.shape))), 1, 'constant', constant_values=1)
        inner = binary_erosion(a, selem=square(3))
        inner_without_pic_border = np.logical_and(inner, np.logical_not(pic_border))
        return np.logical_and(a, np.logical_not(inner_without_pic_border))

    @Decorators.logical
    def _min_distances_between_borders(self, a, b):
        if not np.any(a) or not np.any(b):
            return np.array([np.inf])
        current_distance = 0
        distances = [current_distance for x in range(self._intersection_area(a, b))]
        while not np.all(a):
            current_distance += 1
            next_contour = np.logical_and(binary_dilation(a), np.logical_not(a))
            distances.extend(current_distance for x in range(self._intersection_area(next_contour, b)))
            a = binary_dilation(a)
        return np.array(distances)

    @Decorators.logical
    def _area(self, a):
        return np.sum(a)

    @Decorators.logical
    def _intersection_area(self, a, b):
        return np.sum(np.logical_and(a, b))

    @Decorators.logical
    def _union_area(self, a, b):
        return np.sum(np.logical_or(a, b))
