import keras
import tensorflow as tf
import tensorflow_addons as tfa
from keras import layers, initializers

from layers.ReflectionPadding import ReflectionPadding2D
from options import CycleGAN_Options

kernel_init = initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = initializers.RandomNormal(mean=0.0, stddev=0.02)


class Cycle(CycleGAN_Options):
    def __init__(self):
        super(Cycle, self).__init__()

    def residual_block(self, x, activation, kernel_initializer=kernel_init, kernel_size=(3, 3), strides=(1, 1),
                       padding='valid', gamma_initializer=gamma_init, use_bias=False):
        dim = x.shape[-1]
        input_tensor = x

        x = ReflectionPadding2D()(input_tensor)
        x = layers.Conv2D(filters=dim,
                          kernel_size=kernel_size,
                          strides=strides,
                          kernel_initializer=kernel_initializer,
                          padding=padding,
                          use_bias=use_bias)(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
        x = activation(x)

        x = ReflectionPadding2D()(x)
        x = layers.Conv2D(filters=dim,
                          kernel_size=kernel_size,
                          strides=strides,
                          kernel_initializer=kernel_initializer,
                          padding=padding,
                          use_bias=use_bias)(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
        x = layers.add([input_tensor, x])

        return x

    def downsample(self, x, filters, activation, kernel_initializer=kernel_init, kernel_size=(3, 3), strides=(2, 2),
                   padding='same', gamma_initializer=gamma_init, use_bias=False):

        x = layers.Conv2D(filters=filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          kernel_initializer=kernel_initializer,
                          padding=padding,
                          use_bias=use_bias)(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)

        if activation:
            x = activation(x)
        return x

    def upsample(self, x, filters, activation, kernel_initializer=kernel_init, kernel_size=(3, 3), strides=(2, 2),
                 padding='same', gamma_initializer=gamma_init, use_bias=False):

        x = layers.Conv2DTranspose(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   padding=padding,
                                   kernel_initializer=kernel_initializer,
                                   use_bias=use_bias)(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)

        if activation:
            x = activation(x)

        return x

    def gen(self, filters=64, num_downsampling_blocks=2, num_residual_blocks=9, num_upsample_blocks=2,
                             gamma_initializer=gamma_init, name=None):
        print(self.input_size)
        img_input = layers.Input(shape=self.input_size, name=name + "_img_input")
        x = ReflectionPadding2D(padding=(3, 3))(img_input)
        x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
        x = layers.Activation("relu")(x)

        # Downsampling
        for _ in range(num_downsampling_blocks):
            filters *= 2
            x = self.downsample(x, filters=filters, activation=layers.Activation("relu"))

        # Residual blocks
        for _ in range(num_residual_blocks):
            x = self.residual_block(x, activation=layers.Activation("relu"))

        # Upsampling
        for _ in range(num_upsample_blocks):
            filters //= 2
            x = self.upsample(x, filters, activation=layers.Activation("relu"))

        # Final block
        x = ReflectionPadding2D(padding=(3, 3))(x)
        x = layers.Conv2D(3, (7, 7), padding="valid")(x)
        x = layers.Activation("tanh")(x)

        return keras.models.Model(img_input, x, name=name)

    def disc(self, filters=64, kernel_initializer=kernel_init, num_downsampling=3, name=None):
        img_input = layers.Input(shape=self.input_size, name=f'{name}_img_input')
        x = layers.Conv2D(filters=filters,
                          kernel_size=(4, 4),
                          padding="same",
                          kernel_initializer=kernel_initializer)(img_input)
        x = layers.LeakyReLU(0.2)(x)

        num_filters = filters
        for num_downsample_block in range(num_downsampling):
            num_filters *= 2
            if num_downsample_block < 2:
                x = self.downsample(
                    x,
                    filters=num_filters,
                    activation=layers.LeakyReLU(0.2),
                    kernel_size=(4, 4),
                    strides=(2, 2))
            else:
                x = self.downsample(
                    x,
                    filters=num_filters,
                    activation=layers.LeakyReLU(0.2),
                    kernel_size=(4, 4),
                    strides=(1, 1))

        x = layers.Conv2D(1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer)(x)

        return keras.models.Model(inputs=img_input, outputs=x, name=name)
