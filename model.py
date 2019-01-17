from keras.models import Input, Model
from keras.optimizers import Adam
from keras.layers import Concatenate, Conv2D, Dropout, MaxPooling2D, UpSampling2D
from keras import backend as keras  # TODO change it
from constants import SIZE


class Unet(object):
    class Conv2D(Conv2D):
        def __init__(self, *args, **kwargs):
            common_settings = {
                'activation': 'relu',
                'padding': 'same',
                'kernel_initializer': 'he_normal',
            }
            super().__init__(*args, **common_settings, **kwargs)

    def __init__(self, pretrained_weights=None):
        self.pretrained_weights = pretrained_weights
        self.input_size = (SIZE, SIZE, 1)

    @staticmethod
    def loss_function(y_true, y_pred):
        y_true = keras.flatten(y_true)
        y_pred = keras.flatten(y_pred)
        whites = keras.mean((y_true * (1.0 - y_pred)) ** 2)
        blacks = keras.mean(((1.0 - y_true) * y_pred) ** 2)
        white_area = keras.foldl(lambda acc, x: acc + x, y_true)  # TODO try keras.sum
        black_area = keras.foldl(lambda acc, x: acc + x, 1.0 - y_true)  # TODO try keras.sum
        return (whites * black_area + blacks * white_area) / (white_area + black_area)

    def create(self):
        inputs, outputs = self._structure()
        model = Model(input=inputs, output=outputs)
        model.compile(optimizer=Adam(lr=1e-4), loss=self.loss_function, metrics=['accuracy'])
        if self.pretrained_weights:
            model.load_weights(self.pretrained_weights)
        return model

    def _structure(self):
        inputs = Input(self.input_size)
        conv1 = Unet.Conv2D(64, 3)(Unet.Conv2D(64, 3)(inputs))
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Unet.Conv2D(128, 3)(Unet.Conv2D(128, 3)(pool1))
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Unet.Conv2D(256, 3)(Unet.Conv2D(256, 3)(pool2))
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Unet.Conv2D(512, 3)(Unet.Conv2D(512, 3)(pool3))
        drop4 = Dropout(0.0)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        conv5 = Unet.Conv2D(1024, 3)(Unet.Conv2D(1024, 3)(pool4))
        drop5 = Dropout(0.0)(conv5)
        up6 = Unet.Conv2D(512, 2)(UpSampling2D(size=(2, 2))(drop5))
        conv6 = Unet.Conv2D(512, 3)(Unet.Conv2D(512, 3)(Concatenate()([drop4, up6])))
        up7 = Unet.Conv2D(256, 2)(UpSampling2D(size=(2, 2))(conv6))
        conv7 = Unet.Conv2D(256, 3)(Unet.Conv2D(256, 3)(Concatenate()([conv3, up7])))
        up8 = Unet.Conv2D(128, 2)(UpSampling2D(size=(2, 2))(conv7))
        conv8 = Unet.Conv2D(128, 3)(Unet.Conv2D(128, 3)(Concatenate()([conv2, up8])))
        up9 = Unet.Conv2D(64, 2)(UpSampling2D(size=(2, 2))(conv8))
        conv9 = Unet.Conv2D(64, 3)(Unet.Conv2D(64, 3)(Concatenate()([conv1, up9])))
        outputs = Conv2D(1, 1, activation='sigmoid')(conv9)
        return inputs, outputs
