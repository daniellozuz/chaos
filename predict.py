import os
import skimage.io as io
from generator import PredictionGenerator
from model import Unet


class Predictor(object):
    def save(self):
        for weights in sorted(os.listdir('weights')):
            epoch = weights[:3]
            model = Unet(f'weights/{weights}').create()
            results = model.predict_generator(PredictionGenerator(), verbose=1)
            os.makedirs(f'data/test/prediction{epoch}', exist_ok=True)
            self.save_epoch_results(f'data/test/prediction{epoch}', results)

    def save_epoch_results(self, save_path, npyfile):
        for name, image in zip(sorted(os.listdir('data/test/image')), npyfile):
            io.imsave(os.path.join(save_path, name), image[:, :, 0])
