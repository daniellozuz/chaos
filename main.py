import os
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from environment import Environment
from generator import TrainGenerator, ValidationGenerator
from metrics import Measurer
from model import Unet
from predict import Predictor
from visual import Visualizer


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Environment(training=[2, 5, 6, 10, 14, 16, 19], testing=[1, 8, 18], training_every=10, testing_every=10).configure()

train_gene = TrainGenerator(batch_size=9)

model = Unet().create()
checkpoint = ModelCheckpoint('weights/{epoch:03d}.liver_weights.hdf5', monitor='loss', verbose=1, period=1)
plateau = ReduceLROnPlateau(factor=0.2, min_delta=1e-8, mode='min', monitor='loss', patience=10, cooldown=20, verbose=1)

model.fit_generator(train_gene, epochs=4, callbacks=[checkpoint, plateau], validation_data=ValidationGenerator())

Predictor().save()
Measurer().show_metrics()
Visualizer().show_animation()
